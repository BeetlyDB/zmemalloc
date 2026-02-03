const std = @import("std");
const builtin = @import("builtin");
const assert = @import("util.zig").assert;
const types = @import("types.zig");
const page_mod = @import("page.zig");
const heap_mod = @import("heap.zig");
const segment_mod = @import("segment.zig");
const tld_mod = @import("tld.zig");
const os = @import("os.zig");
const os_alloc = @import("os_allocator.zig");

const Atomic = std.atomic.Value;
const Page = page_mod.Page;
const Block = page_mod.Block;
const Heap = heap_mod.Heap;
const Segment = segment_mod.Segment;
const TLD = tld_mod.TLD;

const RETIRE_CYCLES: u8 = 128; // Reduced for faster page reuse
const MAX_SCAN_PAGES = 8;
const COLLECT_INTERVAL: usize = 512; // Trigger collection every N allocations

// =============================================================================
// Thread-Local State
// =============================================================================

threadlocal var tld: TLD = .{};
threadlocal var tld_initialized: bool = false;
threadlocal var alloc_counter: usize = 0; // Counter for periodic collection

inline fn ensureTldInitialized() *TLD {
    if (!tld_initialized) {
        initTld(&tld);
        tld_initialized = true;
    }
    return &tld;
}

inline fn initTld(t: *TLD) void {
    // Initialize segments with os_allocator stored in TLD (so it doesn't go out of scope)
    t.segments = segment_mod.SegmentsTLD.init(t.os_allocator.allocator());
    t.heap_backing = &heap_mod.heap_main;
    heap_mod.heap_main.tld = t;
    heap_mod.heap_main.thread_id = std.Thread.getCurrentId();
}

// =============================================================================
// Fast Path Allocation
// =============================================================================

/// Fast path for small allocations (inlined)
inline fn mallocSmallFast(heap: *Heap, size: usize) ?[*]u8 {
    assert(size <= types.SMALL_OBJ_SIZE_MAX);

    // Get word size (rounded up)
    const wsize = (size + types.INTPTR_SIZE - 1) / types.INTPTR_SIZE;
    assert(wsize <= types.SMALL_WSIZE_MAX);

    // Direct page lookup for small sizes
    const page = heap.pages_free_direct[wsize] orelse return null;

    // Try to get block from page
    const block = page.popFreeBlock() orelse {
        heap.pages_free_direct[wsize] = null;
        return null;
    };
    // Reset retire counter - page is being used
    page.retire_expire = 0;

    return @ptrCast(block);
}

/// Medium allocation path
inline fn mallocMedium(heap: *Heap, size: usize) ?[*]u8 {
    const bin = page_mod.binFromSize(size);
    assert(bin <= types.BIN_HUGE);

    const pq = &heap.pages[bin];

    // Fast path: check only the first page (tail) - O(1)
    // All pages in this bin have the same block_size = blockSizeForBin(bin)
    if (pq.tail) |pg| {
        @branchHint(.likely);
        if (pg.popFreeBlock()) |block| {
            // Reset retire counter - page is being used
            pg.retire_expire = 0;
            // If page is now full, remove from queue and mark as full
            if (!pg.hasFree()) {
                pg.pageRemoveFromBin(heap, bin);
                pg.set_in_full(true);
                // Clear direct pointers since page is full
                clearDirectPointersForPage(heap, pg);
            }
            return @ptrCast(block);
        }
    }

    return null;
}

/// Set pages_free_direct for the block_size's wsize
/// The bin queue (via mallocMedium) handles other wsizes in the same bin
inline fn setDirectPointerForBlockSize(heap: *Heap, page: *Page) void {
    if (page.block_size == 0) return;

    const block_wsize = (page.block_size + types.INTPTR_SIZE - 1) / types.INTPTR_SIZE;
    if (block_wsize <= types.SMALL_WSIZE_MAX) {
        heap.pages_free_direct[block_wsize] = page;
    }
}

/// Lightweight collection - only processes retire counters (called periodically)
/// This is much faster than full collect() as it only decrements counters
inline fn collectRetired(heap: *Heap) void {
    // Only scan a subset of bins per call to keep it fast
    const start_bin = alloc_counter % (types.BIN_HUGE + 1);
    const end_bin = @min(start_bin + 16, types.BIN_HUGE + 1);

    for (start_bin..end_bin) |bin| {
        const pq = &heap.pages[bin];
        if (pq.tail) |pg| {
            // Only check the tail page (most likely to be empty)
            if (pg.used == 0 and pg.capacity > 0 and pg.retire_expire > 0) {
                pg.retire_expire -|= 1;
                if (pg.retire_expire == 0) {
                    // Time to retire this page
                    pg.pageRemoveFromBin(heap, bin);
                    clearDirectPointersForPage(heap, pg);
                    tld.segments.freePage(pg, false);
                }
            }
        }
    }
}

/// Generic allocation - slow path when fast path fails
inline fn mallocGeneric(heap: *Heap, size: usize, zero: bool) ?[*]u8 {
    const t = heap.tld orelse return null;

    // Round up to bin's block size - this allows page reuse across similar sizes
    const bin = page_mod.binFromSize(size);
    const block_size = page_mod.blockSizeForBin(bin);

    // Get or create a page
    const page = t.segments.allocPage(block_size) orelse return null;

    // Initialize page if not yet initialized (page_start is null)
    if (page.page_start == null) {
        const segment = Segment.fromPtr(page);
        assert(@intFromPtr(segment) % types.SEGMENT_SIZE == 0);
        var page_size: usize = undefined;
        const page_start = segment.pageStart(page, &page_size);
        page.init(block_size, page_start, page_size);
    }

    // Extend free list if needed
    if (page.free.empty() and page.reserved < page.capacity) {
        const extend = @min(16, page.capacity - page.reserved);
        page.extendFree(extend);
    }

    // Get block
    const block = page.popFreeBlock() orelse return null;
    const ptr: [*]u8 = @ptrCast(block);

    if (types.DEBUG) {
        const page_segment = Segment.fromPtr(page);
        const ptr_segment = Segment.fromPtr(ptr);
        assert(ptr_segment == page_segment);
    }

    // Zero if requested
    if (zero) {
        @memset(ptr[0..block_size], 0);
    }

    // Add page to heap queue if not already there
    // Use in_bin flag to track queue membership
    if (bin < heap.pages.len and !page.flags.page_flags.in_bin) {
        page.set_in_full(false);
        page.pagePushBin(heap, bin);
    }

    // Update direct pointer for the block_size's wsize
    // Other wsizes in the bin will use mallocMedium's bin queue lookup
    setDirectPointerForBlockSize(heap, page);

    return ptr;
}

/// Clear pages_free_direct entry for this page's block_size wsize
inline fn clearDirectPointersForPage(heap: *Heap, page: *Page) void {
    if (page.block_size == 0) return;

    const block_wsize = (page.block_size + types.INTPTR_SIZE - 1) / types.INTPTR_SIZE;
    if (block_wsize <= types.SMALL_WSIZE_MAX) {
        if (heap.pages_free_direct[block_wsize] == page) {
            heap.pages_free_direct[block_wsize] = null;
        }
    }
}

//TODO: make in collect
inline fn retirePage(page: *Page) void {
    const heap = &heap_mod.heap_main;
    const bin = page_mod.binFromSize(page.block_size);

    // remove from page queue
    if (bin < heap.pages.len) {
        page.pageRemoveFromBin(heap, bin);
    }

    // clear direct lookup for all wsizes in this bin
    clearDirectPointersForPage(heap, page);

    // free page to segment
    tld.segments.freePage(page, false);
}

/// Free memory
inline fn freeImpl(ptr: ?*anyopaque) void {
    const p = ptr orelse return;

    // Get segment and page from pointer
    const segment = Segment.fromPtr(p);
    const page = Segment.pageFromPtr(p);

    // Check if same thread
    const current_thread = std.Thread.getCurrentId();
    const page_thread = segment.thread_id.load(.acquire);

    const block: *Block = @ptrCast(@alignCast(p));

    if (page_thread == current_thread) {
        @branchHint(.likely);
        // Check if page was marked as full (not in queue)
        const was_full = page.is_in_full();

        // Same thread - fast path
        page.pushFreeBlock(block);

        // If page was full, add back to heap queue first
        if (was_full and tld_initialized and page.block_size > 0) {
            page.set_in_full(false);
            const heap = &heap_mod.heap_main;
            const bin = page_mod.binFromSize(page.block_size);
            if (bin < heap.pages.len) {
                page.pagePushBin(heap, bin);
                // Update direct pointer so fast path can use this page
                setDirectPointerForBlockSize(heap, page);
            }
        }

        // Check if page is completely empty - retire if there are other pages in the bin
        if (page.used == 0 and page.capacity > 0 and tld_initialized) {
            @branchHint(.unlikely);
            const heap = &heap_mod.heap_main;
            const bin = page_mod.binFromSize(page.block_size);

            if (bin < heap.pages.len) {
                const pq = &heap.pages[bin];
                // Only retire if there are OTHER pages in the queue
                // This keeps at least one page per bin for fast reallocation
                if (!(pq.tail == page and page.prev == null)) {
                    // Multiple pages in queue - retire this empty one immediately
                    retirePage(page);
                }
                // If sole page in queue, keep it - it will be reused on next alloc
                // The collectRetired will handle cleanup if it stays empty too long
            }
        }
    } else if (page_thread == 0) {
        @branchHint(.likely);
        // Abandoned segment - just push to local free
        page.pushFreeBlock(block);
    } else {
        @branchHint(.unlikely);
        // Different thread - use cross-thread free
        page.xthreadFree(block);
    }
}

pub const ZMemAllocator = struct {
    const Self = @This();

    pub fn allocator(self: *Self) std.mem.Allocator {
        return .{
            .ptr = @ptrCast(self),
            .vtable = &vtable,
        };
    }

    const vtable = std.mem.Allocator.VTable{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    };

    fn alloc(
        _: *anyopaque,
        len: usize,
        alignment: std.mem.Alignment,
        _: usize,
    ) ?[*]u8 {
        const align_val = alignment.toByteUnits();

        // For standard alignments, use normal allocation
        if (align_val <= types.MAX_ALIGN_GUARANTEE) {
            @branchHint(.likely);
            return malloc(len);
        }

        // For larger alignments, allocate extra and align manually
        const total = len + align_val;
        const ptr = malloc(total) orelse return null;
        const addr = @intFromPtr(ptr);
        const aligned = (addr + align_val - 1) & ~(align_val - 1);
        return @ptrFromInt(aligned);
    }

    fn resize(
        _: *anyopaque,
        buf: []u8,
        _: std.mem.Alignment,
        new_len: usize,
        _: usize,
    ) bool {
        // Check if resize fits within current block
        if (buf.len == 0) return false;

        const page = Segment.pageFromPtr(buf.ptr);
        const block_size = page.block_size;

        // Can resize if new size fits in block
        return new_len <= block_size;
    }

    fn remap(
        ctx: *anyopaque,
        memory: []u8,
        alignment: std.mem.Alignment,
        new_len: usize,
        return_address: usize,
    ) ?[*]u8 {
        // Try resize first
        if (resize(ctx, memory, alignment, new_len, return_address)) {
            return memory.ptr;
        }

        // Allocate new, copy, free old
        const new_ptr = alloc(ctx, new_len, alignment, return_address) orelse return null;
        const copy_len = @min(memory.len, new_len);
        @memcpy(new_ptr[0..copy_len], memory[0..copy_len]);
        free(ctx, memory, alignment, return_address);
        return new_ptr;
    }

    fn free(
        _: *anyopaque,
        buf: []u8,
        _: std.mem.Alignment,
        _: usize,
    ) void {
        freeImpl(buf.ptr);
    }
};

/// Maximum size for fast path (fits in direct page lookup)
const MAX_FAST_SIZE: usize = types.SMALL_WSIZE_MAX * types.INTPTR_SIZE;

/// Allocate memory
pub fn malloc(size: usize) ?[*]u8 {
    if (size == 0) {
        @branchHint(.cold);
        return null;
    }

    const t = ensureTldInitialized();
    const heap = t.heap_backing orelse return null;

    // Fast path for small allocations (only for sizes that fit in direct lookup)
    if (size <= MAX_FAST_SIZE) {
        if (mallocSmallFast(heap, size)) |ptr| {
            @branchHint(.likely);
            return ptr;
        }
        // Try medium path (bin queue) before falling back to generic
        if (mallocMedium(heap, size)) |ptr| {
            return ptr;
        }
        // Fall through to slow path
        return mallocGeneric(heap, size, false);
    }

    // Medium path (larger objects)
    if (size <= types.MEDIUM_OBJ_SIZE_MAX) {
        @branchHint(.likely);
        if (mallocMedium(heap, size)) |ptr| {
            return ptr;
        }
    }

    // Slow path
    return mallocGeneric(heap, size, false);
}

/// Allocate zeroed memory
pub fn zalloc(size: usize) ?[*]u8 {
    if (size == 0) return null;

    const t = ensureTldInitialized();
    const heap = t.heap_backing orelse return null;

    return mallocGeneric(heap, size, true);
}

/// Allocate array (with overflow check)
pub fn calloc(count: usize, size: usize) ?[*]u8 {
    // Check for overflow
    if (count != 0 and size > std.math.maxInt(usize) / count) {
        return null;
    }
    return zalloc(count * size);
}

/// Free memory
pub fn free_mem(ptr: ?*anyopaque) void {
    freeImpl(ptr);
}

/// Reallocate memory
pub fn realloc(ptr: ?*anyopaque, new_size: usize) ?[*]u8 {
    if (ptr == null) return malloc(new_size);
    if (new_size == 0) {
        freeImpl(ptr);
        return null;
    }

    const p: [*]u8 = @ptrCast(ptr.?);

    // Get current block size
    const page = Segment.pageFromPtr(ptr.?);
    const old_size = page.block_size;

    // Check if fits in current block
    if (new_size <= old_size) {
        return p;
    }

    // Allocate new, copy, free old
    const new_ptr = malloc(new_size) orelse return null;
    @memcpy(new_ptr[0..old_size], p[0..old_size]);
    freeImpl(ptr);
    return new_ptr;
}

var global_allocator_instance: ZMemAllocator = .{};

pub fn allocator() std.mem.Allocator {
    return global_allocator_instance.allocator();
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get usable size of allocation
pub fn usableSize(ptr: ?*const anyopaque) usize {
    if (ptr == null) return 0;
    const page = Segment.pageFromPtr(ptr.?);
    return page.block_size;
}

/// Check if pointer is from this allocator
pub fn isInHeapRegion(ptr: ?*const anyopaque) bool {
    if (ptr == null) return false;
    // Simple check: pointer should be segment-aligned block
    const addr = @intFromPtr(ptr.?);
    const segment_start = addr & ~types.SEGMENT_MASK;
    return segment_start != 0;
}

/// Collect and return unused memory to the OS
/// Call this periodically or when memory pressure is high
/// Returns number of pages purged
pub fn collect(force: bool) usize {
    if (!tld_initialized) return 0;

    var collected: usize = 0;

    // Collect unused memory from segments
    collected += tld.segments.collect(force);

    // Clean up heap's page queues - decrement retire_expire and retire when 0
    const heap = &heap_mod.heap_main;
    for (&heap.pages, 0..) |*pq, bin| {
        var page = pq.tail;
        while (page) |pg| {
            const next = pg.prev;
            if (pg.used == 0 and pg.capacity > 0) {
                if (force) {
                    // Force retire
                    pg.pageRemoveFromBin(heap, bin);
                    clearDirectPointersForPage(heap, pg);
                    tld.segments.freePage(pg, true);
                    collected += 1;
                } else if (pg.retire_expire > 0) {
                    // Decrement expire counter
                    pg.retire_expire -= 1;
                    if (pg.retire_expire == 0) {
                        // Time to retire
                        pg.pageRemoveFromBin(heap, bin);
                        clearDirectPointersForPage(heap, pg);
                        tld.segments.freePage(pg, false);
                        collected += 1;
                    }
                }
            }
            page = next;
        }
    }

    // Handle empty pages from direct pointers (might not be in queue)
    // Note: We iterate a copy to avoid issues during clearing
    var pages_to_check: [types.PAGES_DIRECT]?*Page = undefined;
    inline for (&heap.pages_free_direct, 0..) |direct, i| {
        @setEvalBranchQuota(10_000);
        pages_to_check[i] = direct;
    }

    for (pages_to_check) |maybe_pg| {
        const pg = maybe_pg orelse continue;
        if (pg.used == 0 and pg.capacity > 0) {
            if (force) {
                clearDirectPointersForPage(heap, pg);
                const bin = page_mod.binFromSize(pg.block_size);
                pg.pageRemoveFromBin(heap, bin);
                tld.segments.freePage(pg, true);
                collected += 1;
            } else if (pg.retire_expire > 0) {
                pg.retire_expire -= 1;
                if (pg.retire_expire == 0) {
                    clearDirectPointersForPage(heap, pg);
                    const bin = page_mod.binFromSize(pg.block_size);
                    pg.pageRemoveFromBin(heap, bin);
                    tld.segments.freePage(pg, false);
                    collected += 1;
                }
            }
        }
    }

    return collected;
}

/// Get memory statistics
pub fn getStats() struct {
    segments_count: usize,
    segments_size: usize,
    peak_segments: usize,
    peak_size: usize,
} {
    if (!tld_initialized) return .{
        .segments_count = 0,
        .segments_size = 0,
        .peak_segments = 0,
        .peak_size = 0,
    };

    const stats = tld.segments.getStats();
    return .{
        .segments_count = stats.count,
        .segments_size = stats.current_size,
        .peak_segments = stats.peak_count,
        .peak_size = stats.peak_size,
    };
}

/// Aligned allocation (internal)
fn alignedAlloc(alignment: usize, size: usize) ?*anyopaque {
    if (size == 0) return null;
    if (alignment == 0 or (alignment & (alignment - 1)) != 0) return null; // Must be power of 2

    // malloc guarantees alignment up to INTPTR_SIZE (8 bytes on 64-bit)
    // For larger alignments, we need to over-allocate
    const natural_align = types.INTPTR_SIZE;

    if (alignment <= natural_align) {
        return @ptrCast(malloc(size));
    }

    // For larger alignments, over-allocate and align manually
    const total = size + alignment;
    const ptr = malloc(total) orelse return null;
    const addr = @intFromPtr(ptr);
    const aligned_addr = (addr + alignment - 1) & ~(alignment - 1);

    // The segment-based free will work because we free the aligned pointer,
    // and Segment.fromPtr() will find the correct segment regardless of alignment
    return @ptrFromInt(aligned_addr);
}

/// Deinitialize thread-local state (call before thread exit)
pub fn deinitThread() void {
    if (tld_initialized) {
        // TODO: Abandon or free all thread-local allocations
        tld_initialized = false;
    }
}

// =============================================================================
// C ABI Exports (for use as libc malloc replacement)
// =============================================================================

/// C malloc - allocate memory
export fn c_malloc(size: usize) ?*anyopaque {
    return @ptrCast(malloc(size));
}

/// C free - free memory
export fn c_free(ptr: ?*anyopaque) void {
    freeImpl(ptr);
}

/// C realloc - reallocate memory
export fn c_realloc(ptr: ?*anyopaque, size: usize) ?*anyopaque {
    return @ptrCast(realloc(ptr, size));
}

/// C calloc - allocate zeroed array
export fn c_calloc(count: usize, size: usize) ?*anyopaque {
    return @ptrCast(calloc(count, size));
}

/// C aligned_alloc - allocate aligned memory (C11)
export fn c_aligned_alloc(alignment: usize, size: usize) ?*anyopaque {
    // C11: size must be multiple of alignment
    if (size % alignment != 0) return null;
    return alignedAlloc(alignment, size);
}

/// POSIX posix_memalign - allocate aligned memory
export fn c_posix_memalign(memptr: *?*anyopaque, alignment: usize, size: usize) c_int {
    // Alignment must be power of 2 and multiple of sizeof(void*)
    if (alignment < @sizeOf(*anyopaque)) return 22; // EINVAL
    if (alignment & (alignment - 1) != 0) return 22; // EINVAL

    const ptr = alignedAlloc(alignment, size);
    if (ptr == null and size != 0) return 12; // ENOMEM

    memptr.* = ptr;
    return 0;
}

/// Legacy memalign - allocate aligned memory
export fn c_memalign(alignment: usize, size: usize) ?*anyopaque {
    return alignedAlloc(alignment, size);
}

/// valloc - allocate page-aligned memory
export fn c_valloc(size: usize) ?*anyopaque {
    return alignedAlloc(std.heap.page_size_min, size);
}

/// pvalloc - allocate page-aligned memory, rounded up to page size
export fn c_pvalloc(size: usize) ?*anyopaque {
    const page_size = std.heap.page_size_min;
    const aligned_size = (size + page_size - 1) & ~(page_size - 1);
    return alignedAlloc(page_size, aligned_size);
}

/// malloc_usable_size - get usable size of allocation
export fn c_malloc_usable_size(ptr: ?*anyopaque) usize {
    return usableSize(ptr);
}

/// Deinitialize thread (call before thread exit for cleanup)
export fn zmem_thread_deinit() void {
    deinitThread();
}

/// Collect unused memory and return to OS
/// force=true to aggressively return all unused memory
export fn zmem_collect(force: bool) usize {
    return collect(force);
}

/// Get current memory usage in bytes
export fn zmem_get_memory_usage() usize {
    return getStats().segments_size;
}

// =============================================================================
// Tests
// =============================================================================

test "malloc and free basic" {
    const ptr = malloc(64);
    try std.testing.expect(ptr != null);

    if (ptr) |p| {
        // Write to memory
        @memset(p[0..64], 0xAB);
        try std.testing.expectEqual(@as(u8, 0xAB), p[0]);
        try std.testing.expectEqual(@as(u8, 0xAB), p[63]);

        free_mem(p);
    }
}

test "zalloc returns zeroed memory" {
    const ptr = zalloc(128);
    try std.testing.expect(ptr != null);

    if (ptr) |p| {
        // Should be zeroed
        for (p[0..128]) |byte| {
            try std.testing.expectEqual(@as(u8, 0), byte);
        }
        free_mem(p);
    }
}

test "calloc overflow check" {
    // This should fail due to overflow
    const huge_count = std.math.maxInt(usize);
    const ptr = calloc(huge_count, 2);
    try std.testing.expect(ptr == null);
}

test "realloc grows allocation" {
    const ptr1 = malloc(32);
    try std.testing.expect(ptr1 != null);

    if (ptr1) |p| {
        @memset(p[0..32], 0x55);

        const ptr2 = realloc(p, 64);
        try std.testing.expect(ptr2 != null);

        if (ptr2) |p2| {
            // Original data should be preserved
            for (p2[0..32]) |byte| {
                try std.testing.expectEqual(@as(u8, 0x55), byte);
            }
            free_mem(p2);
        }
    }
}

test "realloc with null is like malloc" {
    const ptr = realloc(null, 64);
    try std.testing.expect(ptr != null);
    if (ptr) |p| {
        free_mem(p);
    }
}

test "realloc with zero size frees" {
    const ptr = malloc(64);
    try std.testing.expect(ptr != null);

    const result = realloc(ptr, 0);
    try std.testing.expect(result == null);
    // ptr should be freed (no double-free test needed)
}

test "usableSize returns block size" {
    const ptr = malloc(48);
    try std.testing.expect(ptr != null);

    if (ptr) |p| {
        const usable = usableSize(p);
        // Usable size should be at least requested size
        try std.testing.expect(usable >= 48);
        free_mem(p);
    }
}

test "multiple allocations and frees" {
    var ptrs: [100]?[*]u8 = undefined;

    // Allocate many blocks
    for (&ptrs, 0..) |*p, i| {
        p.* = malloc(32 + i * 8);
        try std.testing.expect(p.* != null);
    }

    // Free in reverse order
    var i: usize = ptrs.len;
    while (i > 0) : (i -= 1) {
        free_mem(ptrs[i - 1]);
    }
}

test "std.mem.Allocator interface" {
    const alloc = allocator();

    const slice = try alloc.alloc(u8, 128);
    defer alloc.free(slice);

    @memset(slice, 0xCC);
    try std.testing.expectEqual(@as(u8, 0xCC), slice[0]);
    try std.testing.expectEqual(@as(u8, 0xCC), slice[127]);
}

// =============================================================================
// Data Correctness Tests
// =============================================================================

test "data integrity: write patterns preserved" {
    const sizes = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 };

    for (sizes) |size| {
        const ptr = malloc(size) orelse {
            try std.testing.expect(false); // Should not fail
            continue;
        };

        // Write unique pattern based on position
        for (0..size) |i| {
            ptr[i] = @truncate(i ^ (size >> 3));
        }

        // Verify pattern
        for (0..size) |i| {
            const expected: u8 = @truncate(i ^ (size >> 3));
            try std.testing.expectEqual(expected, ptr[i]);
        }

        free_mem(ptr);
    }
}

test "data integrity: multiple allocations don't overlap" {
    const num_allocs = 50;
    var ptrs: [num_allocs]?[*]u8 = undefined;
    var allocated_sizes: [num_allocs]usize = undefined;

    // Allocate with different sizes
    for (0..num_allocs) |i| {
        const size = 16 + i * 32;
        allocated_sizes[i] = size;
        ptrs[i] = malloc(size);
        try std.testing.expect(ptrs[i] != null);

        // Fill with unique pattern
        if (ptrs[i]) |p| {
            @memset(p[0..size], @truncate(i + 0x41)); // 'A', 'B', 'C', ...
        }
    }

    // Verify all patterns are intact (no overlap corruption)
    for (0..num_allocs) |i| {
        if (ptrs[i]) |p| {
            const size = allocated_sizes[i];
            const expected: u8 = @truncate(i + 0x41);
            for (p[0..size]) |byte| {
                try std.testing.expectEqual(expected, byte);
            }
        }
    }

    // Free all
    for (ptrs) |p| {
        free_mem(p);
    }
}

test "data integrity: realloc preserves data" {
    var ptr = malloc(64) orelse return error.OutOfMemory;

    // Write pattern
    for (0..64) |i| {
        ptr[i] = @truncate(i * 3);
    }

    // Grow allocation
    ptr = realloc(ptr, 128) orelse return error.OutOfMemory;

    // Verify original data preserved
    for (0..64) |i| {
        const expected: u8 = @truncate(i * 3);
        try std.testing.expectEqual(expected, ptr[i]);
    }

    // Write to new space
    for (64..128) |i| {
        ptr[i] = @truncate(i * 5);
    }

    // Verify all data
    for (0..64) |i| {
        const expected: u8 = @truncate(i * 3);
        try std.testing.expectEqual(expected, ptr[i]);
    }
    for (64..128) |i| {
        const expected: u8 = @truncate(i * 5);
        try std.testing.expectEqual(expected, ptr[i]);
    }

    free_mem(ptr);
}

test "data integrity: calloc returns zeroed memory" {
    const sizes = [_]usize{ 1, 10, 100, 1000 };
    const counts = [_]usize{ 1, 8, 64 };

    for (sizes) |size| {
        for (counts) |count| {
            const ptr = calloc(count, size) orelse continue;
            const total = count * size;

            // Verify all zeros
            for (ptr[0..total]) |byte| {
                try std.testing.expectEqual(@as(u8, 0), byte);
            }

            free_mem(ptr);
        }
    }
}

// =============================================================================
// Leak Detection Tests
// =============================================================================

test "leak test: alloc-free cycles" {
    // Perform many alloc/free cycles - should not leak
    const iterations = 10_000;

    for (0..iterations) |i| {
        const size = 16 + (i % 256) * 8;
        const ptr = malloc(size);
        try std.testing.expect(ptr != null);
        if (ptr) |p| {
            p[0] = 0xAA; // Touch memory
            free_mem(p);
        }
    }
}

test "leak test: varied sizes" {
    const sizes = [_]usize{ 1, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192 };

    for (0..100) |_| {
        for (sizes) |size| {
            const ptr = malloc(size);
            try std.testing.expect(ptr != null);
            free_mem(ptr);
        }
    }
}

test "leak test: realloc chains" {
    var ptr = malloc(16);
    try std.testing.expect(ptr != null);

    // Grow through multiple reallocations
    const growth_sizes = [_]usize{ 32, 64, 128, 256, 512, 1024, 2048, 4096 };
    for (growth_sizes) |new_size| {
        ptr = realloc(ptr, new_size);
        try std.testing.expect(ptr != null);
    }

    // Shrink back
    const shrink_sizes = [_]usize{ 2048, 1024, 512, 256, 128, 64, 32, 16 };
    for (shrink_sizes) |new_size| {
        ptr = realloc(ptr, new_size);
        try std.testing.expect(ptr != null);
    }

    free_mem(ptr);
}

test "leak test: interleaved alloc-free" {
    var ptrs: [100]?[*]u8 = [_]?[*]u8{null} ** 100;

    // Interleaved pattern: alloc odd indices, alloc even, free odd, free even
    for (0..100) |i| {
        if (i % 2 == 1) {
            ptrs[i] = malloc(64 + i * 4);
        }
    }

    for (0..100) |i| {
        if (i % 2 == 0) {
            ptrs[i] = malloc(64 + i * 4);
        }
    }

    // Free odd
    for (0..100) |i| {
        if (i % 2 == 1) {
            free_mem(ptrs[i]);
            ptrs[i] = null;
        }
    }

    // Free even
    for (0..100) |i| {
        if (i % 2 == 0) {
            free_mem(ptrs[i]);
            ptrs[i] = null;
        }
    }
}

test "leak test: stress with random-like pattern" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    var ptrs: [256]?[*]u8 = [_]?[*]u8{null} ** 256;

    // Simulate random alloc/free pattern
    for (0..5000) |_| {
        const idx = random.uintLessThan(usize, 256);

        if (ptrs[idx]) |p| {
            free_mem(p);
            ptrs[idx] = null;
        } else {
            const size = 8 + random.uintLessThan(usize, 4088);
            ptrs[idx] = malloc(size);
        }
    }

    // Free remaining
    for (&ptrs) |*p| {
        if (p.*) |ptr| {
            free_mem(ptr);
            p.* = null;
        }
    }
}

// =============================================================================
// C API Tests
// =============================================================================

test "C API: aligned_alloc" {
    const alignments = [_]usize{ 8, 16, 32, 64, 128, 256, 512, 1024, 4096 };

    for (alignments) |alignment| {
        // Size must be multiple of alignment for C11 aligned_alloc
        const size = alignment * 4;
        const ptr = c_aligned_alloc(alignment, size);

        if (alignment <= types.MAX_ALIGN_GUARANTEE or size > 0) {
            // Should succeed for reasonable alignments
            if (ptr) |p| {
                // Verify alignment
                const addr = @intFromPtr(p);
                try std.testing.expect(addr % alignment == 0);

                // Write and verify
                const bytes: [*]u8 = @ptrCast(p);
                @memset(bytes[0..size], 0xDD);
                try std.testing.expectEqual(@as(u8, 0xDD), bytes[0]);

                c_free(p);
            }
        }
    }
}

test "C API: posix_memalign" {
    var ptr: ?*anyopaque = null;

    // Valid alignment (power of 2, >= sizeof(void*))
    const result = c_posix_memalign(&ptr, 64, 256);
    try std.testing.expectEqual(@as(c_int, 0), result);
    try std.testing.expect(ptr != null);

    if (ptr) |p| {
        const addr = @intFromPtr(p);
        try std.testing.expect(addr % 64 == 0);
        c_free(p);
    }

    // Invalid alignment (not power of 2)
    var ptr2: ?*anyopaque = null;
    const result2 = c_posix_memalign(&ptr2, 65, 256);
    try std.testing.expectEqual(@as(c_int, 22), result2); // EINVAL
}

test "C API: malloc_usable_size" {
    const ptr = c_malloc(100);
    try std.testing.expect(ptr != null);

    if (ptr) |p| {
        const usable = c_malloc_usable_size(p);
        try std.testing.expect(usable >= 100);
        c_free(p);
    }

    // Null pointer should return 0
    try std.testing.expectEqual(@as(usize, 0), c_malloc_usable_size(null));
}

test "C API: valloc and pvalloc" {
    const page_size = std.heap.page_size_min;

    // valloc
    const ptr1 = c_valloc(100);
    if (ptr1) |p| {
        const addr = @intFromPtr(p);
        try std.testing.expect(addr % page_size == 0);
        c_free(p);
    }

    // pvalloc (size rounded up to page)
    const ptr2 = c_pvalloc(100);
    if (ptr2) |p| {
        const addr = @intFromPtr(p);
        try std.testing.expect(addr % page_size == 0);
        // Usable size should be at least one page
        const usable = c_malloc_usable_size(p);
        try std.testing.expect(usable >= page_size);
        c_free(p);
    }
}

// =============================================================================
// Data Race Tests (Thread Safety)
// =============================================================================

test "data race: concurrent allocation from multiple threads" {
    const num_threads = 8;
    const allocs_per_thread = 1000;

    const ThreadContext = struct {
        thread_id: usize,
        errors: std.atomic.Value(usize),

        fn worker(ctx: *@This()) void {
            var local_errors: usize = 0;

            for (0..allocs_per_thread) |i| {
                const size = 16 + (i % 256) * 8;
                const ptr = malloc(size) orelse {
                    local_errors += 1;
                    continue;
                };

                // Write thread-specific pattern
                const pattern: u8 = @truncate(ctx.thread_id + 1);
                @memset(ptr[0..size], pattern);

                // Verify pattern before free
                for (ptr[0..size]) |byte| {
                    if (byte != pattern) {
                        local_errors += 1;
                        break;
                    }
                }

                free_mem(ptr);
            }

            _ = ctx.errors.fetchAdd(local_errors, .seq_cst);
        }
    };

    var contexts: [num_threads]ThreadContext = undefined;
    var threads: [num_threads]std.Thread = undefined;

    // Start threads
    for (0..num_threads) |i| {
        contexts[i] = .{
            .thread_id = i,
            .errors = std.atomic.Value(usize).init(0),
        };
        threads[i] = std.Thread.spawn(.{}, ThreadContext.worker, .{&contexts[i]}) catch {
            // If we can't spawn, skip this test
            return;
        };
    }

    // Wait for all threads
    for (&threads) |*t| {
        t.join();
    }

    // Check for errors
    var total_errors: usize = 0;
    for (&contexts) |*ctx| {
        total_errors += ctx.errors.load(.seq_cst);
    }

    try std.testing.expectEqual(@as(usize, 0), total_errors);
}

test "data race: cross-thread free" {
    const num_allocs = 500;

    // Shared array of pointers - allocated by one thread, freed by another
    var shared_ptrs: [num_allocs]std.atomic.Value(?[*]u8) = undefined;
    for (&shared_ptrs) |*p| {
        p.* = std.atomic.Value(?[*]u8).init(null);
    }

    var allocators_done = std.atomic.Value(usize).init(0);
    var errors = std.atomic.Value(usize).init(0);

    const AllocatorThread = struct {
        fn run(ptrs: []std.atomic.Value(?[*]u8), done: *std.atomic.Value(usize), errs: *std.atomic.Value(usize), thread_id: usize) void {
            const start = (ptrs.len / 2) * thread_id / 2;
            const end = (ptrs.len / 2) * (thread_id + 1) / 2;

            for (start..end) |i| {
                const size = 64 + (i % 128) * 4;
                const ptr = malloc(size) orelse {
                    _ = errs.fetchAdd(1, .seq_cst);
                    continue;
                };

                // Write pattern
                const pattern: u8 = @truncate(i + 0xAA);
                @memset(ptr[0..size], pattern);

                ptrs[i].store(ptr, .release);
            }

            _ = done.fetchAdd(1, .seq_cst);
        }
    };

    const FreeThread = struct {
        fn run(ptrs: []std.atomic.Value(?[*]u8), done: *std.atomic.Value(usize), errs: *std.atomic.Value(usize), thread_id: usize) void {
            _ = thread_id;

            // Wait for allocators to finish
            while (done.load(.acquire) < 2) {
                std.atomic.spinLoopHint();
            }

            // Free all pointers
            for (ptrs, 0..) |*p, i| {
                const ptr = p.swap(null, .acq_rel) orelse continue;

                // Verify pattern before free
                const size = 64 + (i % 128) * 4;
                const pattern: u8 = @truncate(i + 0xAA);
                for (ptr[0..size]) |byte| {
                    if (byte != pattern) {
                        _ = errs.fetchAdd(1, .seq_cst);
                        break;
                    }
                }

                free_mem(ptr);
            }
        }
    };

    // Spawn allocator threads
    var alloc_threads: [2]std.Thread = undefined;
    for (0..2) |i| {
        alloc_threads[i] = std.Thread.spawn(.{}, AllocatorThread.run, .{ &shared_ptrs, &allocators_done, &errors, i }) catch return;
    }

    // Spawn free threads
    var free_threads: [2]std.Thread = undefined;
    for (0..2) |i| {
        free_threads[i] = std.Thread.spawn(.{}, FreeThread.run, .{ &shared_ptrs, &allocators_done, &errors, i }) catch return;
    }

    // Wait for all
    for (&alloc_threads) |*t| t.join();
    for (&free_threads) |*t| t.join();

    try std.testing.expectEqual(@as(usize, 0), errors.load(.seq_cst));
}

test "data race: producer-consumer pattern" {
    const queue_size = 256;
    const total_items = 2000;

    var queue: [queue_size]std.atomic.Value(?[*]u8) = undefined;
    for (&queue) |*p| {
        p.* = std.atomic.Value(?[*]u8).init(null);
    }

    var produced = std.atomic.Value(usize).init(0);
    var consumed = std.atomic.Value(usize).init(0);
    var errors = std.atomic.Value(usize).init(0);
    var done = std.atomic.Value(bool).init(false);

    const Producer = struct {
        fn run(q: []std.atomic.Value(?[*]u8), prod: *std.atomic.Value(usize), errs: *std.atomic.Value(usize)) void {
            var i: usize = 0;
            while (i < total_items) {
                const slot = i % q.len;

                // Wait for slot to be empty
                while (q[slot].load(.acquire) != null) {
                    std.atomic.spinLoopHint();
                }

                const size = 32 + (i % 64) * 8;
                const ptr = malloc(size) orelse {
                    _ = errs.fetchAdd(1, .seq_cst);
                    i += 1;
                    continue;
                };

                // Write item number as pattern
                const pattern: u8 = @truncate(i);
                @memset(ptr[0..size], pattern);

                q[slot].store(ptr, .release);
                _ = prod.fetchAdd(1, .seq_cst);
                i += 1;
            }
        }
    };

    const Consumer = struct {
        fn run(q: []std.atomic.Value(?[*]u8), cons: *std.atomic.Value(usize), errs: *std.atomic.Value(usize), is_done: *std.atomic.Value(bool)) void {
            var i: usize = 0;
            while (i < total_items) {
                const slot = i % q.len;

                // Wait for item
                const ptr = blk: {
                    var spins: usize = 0;
                    while (true) {
                        if (q[slot].swap(null, .acq_rel)) |p| break :blk p;
                        spins += 1;
                        if (spins > 1_000_000) {
                            // Timeout - producer might have failed
                            if (is_done.load(.acquire)) return;
                            spins = 0;
                        }
                        std.atomic.spinLoopHint();
                    }
                };

                // Verify pattern
                const size = 32 + (i % 64) * 8;
                const expected: u8 = @truncate(i);
                for (ptr[0..size]) |byte| {
                    if (byte != expected) {
                        _ = errs.fetchAdd(1, .seq_cst);
                        break;
                    }
                }

                free_mem(ptr);
                _ = cons.fetchAdd(1, .seq_cst);
                i += 1;
            }
        }
    };

    // Start producer and consumer
    const producer = std.Thread.spawn(.{}, Producer.run, .{ &queue, &produced, &errors }) catch return;
    const consumer = std.Thread.spawn(.{}, Consumer.run, .{ &queue, &consumed, &errors, &done }) catch return;

    producer.join();
    done.store(true, .release);
    consumer.join();

    try std.testing.expectEqual(@as(usize, 0), errors.load(.seq_cst));
    try std.testing.expectEqual(total_items, produced.load(.seq_cst));
}

test "data race: stress test with many threads" {
    const num_threads = 16;
    const ops_per_thread = 500;

    var errors = std.atomic.Value(usize).init(0);
    var total_allocs = std.atomic.Value(usize).init(0);
    var total_frees = std.atomic.Value(usize).init(0);

    const StressWorker = struct {
        fn run(errs: *std.atomic.Value(usize), allocs: *std.atomic.Value(usize), frees: *std.atomic.Value(usize), seed: u64) void {
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();

            var local_ptrs: [32]?[*]u8 = [_]?[*]u8{null} ** 32;
            var local_sizes: [32]usize = [_]usize{0} ** 32;

            for (0..ops_per_thread) |_| {
                const idx = random.uintLessThan(usize, 32);
                const op = random.uintLessThan(u8, 100);

                if (op < 60) {
                    // 60% chance: allocate
                    if (local_ptrs[idx] == null) {
                        const size = 8 + random.uintLessThan(usize, 2040);
                        const ptr = malloc(size) orelse continue;

                        // Write pattern
                        const pattern: u8 = @truncate(idx ^ size);
                        @memset(ptr[0..size], pattern);

                        local_ptrs[idx] = ptr;
                        local_sizes[idx] = size;
                        _ = allocs.fetchAdd(1, .seq_cst);
                    }
                } else if (op < 90) {
                    // 30% chance: free
                    if (local_ptrs[idx]) |ptr| {
                        const size = local_sizes[idx];
                        const pattern: u8 = @truncate(idx ^ size);

                        // Verify before free
                        for (ptr[0..size]) |byte| {
                            if (byte != pattern) {
                                _ = errs.fetchAdd(1, .seq_cst);
                                break;
                            }
                        }

                        free_mem(ptr);
                        local_ptrs[idx] = null;
                        _ = frees.fetchAdd(1, .seq_cst);
                    }
                } else {
                    // 10% chance: realloc
                    if (local_ptrs[idx]) |ptr| {
                        const old_size = local_sizes[idx];
                        const new_size = 8 + random.uintLessThan(usize, 2040);

                        const new_ptr = realloc(ptr, new_size) orelse continue;

                        // Verify old data preserved (up to min of old and new size)
                        const check_size = @min(old_size, new_size);
                        const pattern: u8 = @truncate(idx ^ old_size);
                        for (new_ptr[0..check_size]) |byte| {
                            if (byte != pattern) {
                                _ = errs.fetchAdd(1, .seq_cst);
                                break;
                            }
                        }

                        // Write new pattern
                        const new_pattern: u8 = @truncate(idx ^ new_size);
                        @memset(new_ptr[0..new_size], new_pattern);

                        local_ptrs[idx] = new_ptr;
                        local_sizes[idx] = new_size;
                    }
                }
            }

            // Cleanup remaining allocations
            for (&local_ptrs) |*p| {
                if (p.*) |ptr| {
                    free_mem(ptr);
                    p.* = null;
                    _ = frees.fetchAdd(1, .seq_cst);
                }
            }
        }
    };

    var threads: [num_threads]std.Thread = undefined;
    for (0..num_threads) |i| {
        threads[i] = std.Thread.spawn(.{}, StressWorker.run, .{ &errors, &total_allocs, &total_frees, @as(u64, i) * 12345 }) catch return;
    }

    for (&threads) |*t| {
        t.join();
    }

    const err_count = errors.load(.seq_cst);
    const alloc_count = total_allocs.load(.seq_cst);
    const free_count = total_frees.load(.seq_cst);

    std.debug.print("\n=== Data Race Stress Test Results ===\n", .{});
    std.debug.print("Threads: {}, Ops/thread: {}\n", .{ num_threads, ops_per_thread });
    std.debug.print("Total allocs: {}, Total frees: {}\n", .{ alloc_count, free_count });
    std.debug.print("Errors: {}\n\n", .{err_count});

    try std.testing.expectEqual(@as(usize, 0), err_count);
    try std.testing.expectEqual(alloc_count, free_count);
}

test "data race: rapid alloc-free ping-pong" {
    const num_pairs = 4;
    const iterations = 1000;

    var errors = std.atomic.Value(usize).init(0);

    const PingPong = struct {
        ptr: std.atomic.Value(?[*]u8) = std.atomic.Value(?[*]u8).init(null),
        size: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        turn: std.atomic.Value(bool) = std.atomic.Value(bool).init(false), // false = allocator's turn
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        fn allocator(self: *@This(), errs: *std.atomic.Value(usize), seed: u64) void {
            var prng = std.Random.DefaultPrng.init(seed);
            const random = prng.random();

            for (0..iterations) |i| {
                // Wait for our turn
                while (self.turn.load(.acquire) != false) {
                    if (self.done.load(.acquire)) return;
                    std.atomic.spinLoopHint();
                }

                const size = 32 + random.uintLessThan(usize, 480);
                const ptr = malloc(size) orelse {
                    _ = errs.fetchAdd(1, .seq_cst);
                    self.turn.store(true, .release);
                    continue;
                };

                // Write pattern
                const pattern: u8 = @truncate(i);
                @memset(ptr[0..size], pattern);

                self.size.store(size, .release);
                self.ptr.store(ptr, .release);
                self.turn.store(true, .release); // Signal freer
            }

            self.done.store(true, .release);
        }

        fn freer(self: *@This(), errs: *std.atomic.Value(usize)) void {
            for (0..iterations) |i| {
                // Wait for our turn
                while (self.turn.load(.acquire) != true) {
                    if (self.done.load(.acquire)) return;
                    std.atomic.spinLoopHint();
                }

                const ptr = self.ptr.swap(null, .acq_rel) orelse {
                    self.turn.store(false, .release);
                    continue;
                };
                const size = self.size.load(.acquire);

                // Verify pattern
                const pattern: u8 = @truncate(i);
                for (ptr[0..size]) |byte| {
                    if (byte != pattern) {
                        _ = errs.fetchAdd(1, .seq_cst);
                        break;
                    }
                }

                free_mem(ptr);
                self.turn.store(false, .release); // Signal allocator
            }
        }
    };

    var pairs: [num_pairs]PingPong = [_]PingPong{.{}} ** num_pairs;
    var alloc_threads: [num_pairs]std.Thread = undefined;
    var free_threads: [num_pairs]std.Thread = undefined;

    // Start all threads
    for (0..num_pairs) |i| {
        alloc_threads[i] = std.Thread.spawn(.{}, PingPong.allocator, .{ &pairs[i], &errors, @as(u64, i) * 9999 }) catch return;
        free_threads[i] = std.Thread.spawn(.{}, PingPong.freer, .{ &pairs[i], &errors }) catch return;
    }

    // Wait for all
    for (&alloc_threads) |*t| t.join();
    for (&free_threads) |*t| t.join();

    try std.testing.expectEqual(@as(usize, 0), errors.load(.seq_cst));
}

// =============================================================================
// Benchmark: zmemalloc vs smp_allocator
// =============================================================================

test "benchmark: zmemalloc vs smp_allocator" {
    const iterations = 100_000;
    const sizes = [_]usize{
        16,
        64,
        256,
        1024,
        4096,
        8192,
        16384,
        16384 * 2,
        32768 * 2,
    };

    std.debug.print("\n\n=== Allocator Benchmark ({} iterations per size) ===\n", .{iterations});
    std.debug.print("{s:>10} | {s:>15} | {s:>15} | {s:>10}\n", .{ "Size", "zmemalloc", "smp_allocator", "Speedup" });
    std.debug.print("{s:-^10}-+-{s:-^15}-+-{s:-^15}-+-{s:-^10}\n", .{ "", "", "", "" });

    for (sizes) |size| {
        // Benchmark zmemalloc
        const zmem_time = blk: {
            var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
            for (0..iterations) |_| {
                const ptr = malloc(size) orelse continue;
                free_mem(ptr);
            }
            break :blk timer.read();
        };

        // Benchmark smp_allocator
        const smp_time = blk: {
            const smp = std.heap.smp_allocator;
            var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
            for (0..iterations) |_| {
                const slice = smp.alloc(u8, size) catch continue;
                smp.free(slice);
            }
            break :blk timer.read();
        };

        const zmem_ns = zmem_time / iterations;
        const smp_ns = smp_time / iterations;
        const speedup = if (zmem_ns > 0) @as(f64, @floatFromInt(smp_ns)) / @as(f64, @floatFromInt(zmem_ns)) else 0.0;

        std.debug.print("{d:>10} | {d:>12} ns | {d:>12} ns | {d:>9.2}x\n", .{ size, zmem_ns, smp_ns, speedup });
    }

    std.debug.print("\n", .{});
}

test "benchmark: mixed workload" {
    const num_ptrs = 10000;
    const iterations = 100;

    std.debug.print("\n=== Mixed Workload Benchmark ===\n", .{});
    std.debug.print("Pattern: alloc {}, free random half, alloc again, free all\n", .{num_ptrs});
    std.debug.print("Iterations: {}\n\n", .{iterations});

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    // zmemalloc
    const zmem_time = blk: {
        var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
        for (0..iterations) |_| {
            var ptrs: [num_ptrs]?[*]u8 = undefined;

            // Allocate all
            for (&ptrs) |*p| {
                const size = 16 + random.uintLessThan(usize, 1024);
                p.* = malloc(size);
            }

            // Free random half
            for (&ptrs) |*p| {
                if (random.boolean()) {
                    if (p.*) |ptr| free_mem(ptr);
                    p.* = null;
                }
            }

            // Allocate again into freed slots
            for (&ptrs) |*p| {
                if (p.* == null) {
                    const size = 16 + random.uintLessThan(usize, 1024);
                    p.* = malloc(size);
                }
            }

            // Free all
            for (ptrs) |p| {
                if (p) |ptr| free_mem(ptr);
            }
        }

        // Collect at the end
        break :blk timer.read();
    };

    // smp_allocator
    const smp_time = blk: {
        const smp = std.heap.smp_allocator;
        prng = std.Random.DefaultPrng.init(12345); // Reset PRNG
        var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
        for (0..iterations) |_| {
            var ptrs: [num_ptrs]?[]u8 = undefined;
            @memset(&ptrs, null);

            // Allocate all
            for (&ptrs) |*p| {
                const size = 16 + random.uintLessThan(usize, 1024);
                p.* = smp.alloc(u8, size) catch null;
            }

            // Free random half
            for (&ptrs) |*p| {
                if (random.boolean()) {
                    if (p.*) |slice| smp.free(slice);
                    p.* = null;
                }
            }

            // Allocate again into freed slots
            for (&ptrs) |*p| {
                if (p.* == null) {
                    const size = 16 + random.uintLessThan(usize, 1024);
                    p.* = smp.alloc(u8, size) catch null;
                }
            }

            // Free all
            for (ptrs) |p| {
                if (p) |slice| smp.free(slice);
            }
        }
        break :blk timer.read();
    };

    const zmem_ms = @as(f64, @floatFromInt(zmem_time)) / 1_000_000.0;
    const smp_ms = @as(f64, @floatFromInt(smp_time)) / 1_000_000.0;
    const speedup = if (zmem_ms > 0) smp_ms / zmem_ms else 0.0;

    std.debug.print("{s:<15}: {d:>10.2} ms\n", .{ "zmemalloc", zmem_ms });
    std.debug.print("{s:<15}: {d:>10.2} ms\n", .{ "smp_allocator", smp_ms });
    std.debug.print("{s:<15}: {d:>10.2}x\n\n", .{ "Speedup", speedup });
}
