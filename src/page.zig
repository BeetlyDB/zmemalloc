//! # Page and Block Management
//!
//! Pages are the core allocation unit in zmemalloc. Each page contains
//! fixed-size blocks of a single size class.
//!
//! ## Memory Layout
//!
//! ```
//! Segment (32MB)
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Segment Header │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N    │
//! └─────────────────────────────────────────────────────────────┘
//!                      │
//!                      ▼
//!                  Page (64KB/512KB)
//!                  ┌─────────────────────────────────────────┐
//!                  │ Block │ Block │ Block │ ... │ Block     │
//!                  └─────────────────────────────────────────┘
//! ```
//!
//! ## Three-Level Free List Sharding
//!
//! For optimal multi-threaded performance, each page has three free lists:
//!
//! 1. **`free`**: Main free list for allocation (LIFO stack)
//! 2. **`local_free`**: Same-thread frees (merged into `free` when empty)
//! 3. **`xthread_free`**: Cross-thread frees (atomic, lock-free)
//!
//! This design minimizes contention: allocations only touch `free`,
//! same-thread frees go to `local_free`, and only cross-thread frees
//! require atomic operations.
//!
//! ## Allocation Strategy
//!
//! 1. Pop from `free` list (hot path)
//! 2. Bump pointer for fresh pages (sequential memory access)
//! 3. Merge `local_free` into `free` (when `free` empty)
//! 4. Collect `xthread_free` (atomic swap, cold path)
//!
//! ## Block Structure
//!
//! ```
//! Allocated:  [=========== user data ===========]
//! Free:       [next_ptr][====== unused ========]
//! ```
//!
//! The same memory serves as user data when allocated, or as a linked
//! list node when free. No separate metadata needed.

const std = @import("std");
const list = @import("queue.zig");
const types = @import("types.zig");
const Atomic = std.atomic.Value;
const builtin = @import("builtin");
const assert = @import("util.zig").assert;
const heap = @import("heap.zig");

/// Global counter of pending cross-thread frees
/// When this exceeds a threshold, triggers collection in malloc path
pub var pending_xthread_free: Atomic(usize) = .init(0);

/// Intrusive free block node
/// When memory is free, the first 8 bytes store a pointer to the next free block.
/// When allocated, user owns the entire block including these bytes.
pub const Block = struct {
    link: list.IntrusiveLifo(Block).Link = .{},
};

/// Page structure - exactly 64 bytes to fit in single cache line.
///
/// Memory layout optimized for allocation hot path:
/// - First 32 bytes: hot fields accessed on every alloc (free list, block_size, counters)
/// - Next 32 bytes: queue management and cross-thread state
///
/// Uses bump pointer allocation for fresh pages (sequential memory access),
/// then recycles from free lists for subsequent allocations.
pub const Page = struct {
    const Self = @This();

    // === Hot path fields (32 bytes) ===
    /// Main free list head pointer (IntrusiveLifoLink*)
    free_head: ?*anyopaque = null,
    /// Start of allocatable memory area within segment
    page_start: ?[*]u8 = null,
    /// Size of each block in this page (all blocks same size)
    block_size: usize = 0,
    /// Maximum number of blocks that fit in this page
    capacity: u16 = 0,
    /// Bump pointer position - blocks [0..reserved) have been allocated at least once
    reserved: u16 = 0,
    /// Number of blocks currently in use by application
    used: u16 = 0,
    /// Packed: segment_idx (10 bits) + flags (6 bits)
    flags_and_idx: u16 = 0,

    // === Queue fields (32 bytes) ===
    /// Same-thread freed blocks head pointer
    local_free_head: ?*anyopaque = null,
    /// Cross-thread freed blocks - atomic lock-free list
    xthread_free: Atomic(?*Block) = .init(null),
    /// Next page in heap's bin queue
    next: ?*Page = null,
    /// Previous page in heap's bin queue
    prev: ?*Page = null,

    comptime {
        std.debug.assert(@sizeOf(Page) == 64);
    }

    /// Bit-packed page metadata - fits in single u16.
    /// Flags in low 6 bits, segment index in high 10 bits.
    pub const PackedInfo = packed struct(u16) {
        /// Page is full (all blocks used), removed from bin queue
        in_full: bool = false,
        /// Page is in heap's bin queue for its size class
        in_bin: bool = false,
        /// This is a huge allocation (> 16MB, custom segment size)
        is_huge: bool = false,
        /// Page is claimed and in use by segment
        segment_in_use: bool = false,
        /// Physical memory is committed (not just reserved)
        is_commited: bool = false,
        /// Memory is zero-initialized (for calloc optimization)
        is_zero_init: bool = false,
        /// Index of this page within segment.pages[] array (max 1024)
        segment_idx: u10 = 0,
    };

    /// Doubly-linked list for bin queues
    pub const Queue = list.DoublyLinkedListType(Page, .next, .prev);

    /// Page size category - determines page size within segment
    pub const PageKind = enum {
        PAGE_SMALL, // 64KB pages, blocks 8-1024 bytes
        PAGE_MEDIUM, // 512KB pages, blocks 1KB-128KB
        PAGE_LARGE, // Full segment, blocks 128KB-16MB
        PAGE_HUGE, // Variable segment, blocks > 16MB
    };

    inline fn info(self: *const Self) PackedInfo {
        return @bitCast(self.flags_and_idx);
    }

    inline fn setInfo(self: *Self, i: PackedInfo) void {
        self.flags_and_idx = @bitCast(i);
    }

    /// Get page's index within segment.pages[] array
    pub inline fn segment_idx(self: *const Self) u10 {
        return self.info().segment_idx;
    }

    /// Set page's index within segment.pages[] array
    pub inline fn set_segment_idx(self: *Self, idx: u10) void {
        var i = self.info();
        i.segment_idx = idx;
        self.setInfo(i);
    }

    pub inline fn is_in_full(self: *const Self) bool {
        return self.info().in_full;
    }

    pub inline fn set_in_full(self: *Self, val: bool) void {
        var i = self.info();
        i.in_full = val;
        self.setInfo(i);
    }

    pub inline fn is_in_bin(self: *const Self) bool {
        return self.info().in_bin;
    }

    pub inline fn set_in_bin(self: *Self, val: bool) void {
        var i = self.info();
        i.in_bin = val;
        self.setInfo(i);
    }

    pub inline fn is_huge(self: *const Self) bool {
        return self.info().is_huge;
    }

    pub inline fn set_huge(self: *Self, val: bool) void {
        var i = self.info();
        i.is_huge = val;
        self.setInfo(i);
    }

    pub inline fn is_segment_in_use(self: *const Self) bool {
        return self.info().segment_in_use;
    }

    pub inline fn set_segment_in_use(self: *Self, val: bool) void {
        var i = self.info();
        i.segment_in_use = val;
        self.setInfo(i);
    }

    pub inline fn is_commited(self: *const Self) bool {
        return self.info().is_commited;
    }

    pub inline fn set_commited(self: *Self, val: bool) void {
        var i = self.info();
        i.is_commited = val;
        self.setInfo(i);
    }

    pub inline fn is_zero_init(self: *const Self) bool {
        return self.info().is_zero_init;
    }

    pub inline fn set_zero_init(self: *Self, val: bool) void {
        var i = self.info();
        i.is_zero_init = val;
        self.setInfo(i);
    }

    // =========================================================================
    // Bin Queue Management
    // =========================================================================

    /// Add page to heap's bin queue for its size class.
    /// If already in bin, removes first to avoid duplicates.
    pub inline fn pagePushBin(self: *Self, hp: *heap.Heap, bin: usize) void {
        @setEvalBranchQuota(10_000);
        if (self.is_in_bin()) {
            hp.pages[bin].remove(self);
        }
        hp.pages[bin].push(self);
        self.set_in_bin(true);
    }

    /// Remove page from heap's bin queue.
    /// Called when page becomes full or is being retired.
    pub inline fn pageRemoveFromBin(self: *Self, hp: *heap.Heap, bin: usize) void {
        if (self.is_in_bin()) {
            hp.pages[bin].remove(self);
            self.set_in_bin(false);
        }
    }

    /// Initialize page for use with given block size
    pub fn init(self: *Self, block_size: usize, page_start: [*]u8, page_size: usize) void {
        self.block_size = block_size;
        self.page_start = page_start;
        self.used = 0;
        self.free_head = null;
        self.local_free_head = null;
        self.xthread_free.store(null, .release);

        // Calculate capacity
        if (block_size > 0 and page_size >= block_size) {
            self.capacity = @intCast(page_size / block_size);
            self.reserved = 0; // Bump pointer starts at 0
        } else {
            self.capacity = 0;
            self.reserved = 0;
        }

        // Bump pointer handles initial allocation - no pre-extension needed
    }

    // =========================================================================
    // Free List Management
    // =========================================================================

    /// Check if free list is empty
    inline fn freeEmpty(self: *const Self) bool {
        return self.free_head == null;
    }

    /// Check if local_free list is empty
    inline fn localFreeEmpty(self: *const Self) bool {
        return self.local_free_head == null;
    }

    /// Check if page has any free blocks (including bump pointer room)
    pub inline fn hasFree(self: *const Self) bool {
        // Bump pointer has room (cheapest check - register comparison)
        if (self.reserved < self.capacity) return true;
        // Free list (recycled + same-thread freed blocks)
        if (!self.freeEmpty()) return true;
        // Cross-thread free list (cold path)
        return self.xthread_free.load(.monotonic) != null;
    }

    /// Quick check without atomic xthread load — for hot allocation path.
    /// Cross-thread freed blocks will be collected on next popFreeBlockSlow().
    pub inline fn hasFreeQuick(self: *const Self) bool {
        return self.reserved < self.capacity or !self.freeEmpty();
    }

    /// Check if all blocks are used
    pub inline fn allUsed(self: *const Self) bool {
        return self.used >= self.capacity;
    }

    /// Check if page is mostly empty (for retirement)
    pub inline fn isMostlyEmpty(self: *const Self) bool {
        if (self.capacity == 0) return true;
        // Consider page empty if less than 1/8 used
        return self.used <= self.capacity / 8;
    }

    /// Collect cross-thread free list (atomic) - batch optimized
    /// Instead of pushing blocks one-by-one, splice the entire list at once
    pub inline fn collectXthreadFree(self: *Self) usize {
        const head = self.xthread_free.swap(null, .acquire) orelse return 0;

        // Find tail and count in single pass
        var tail: *Block = head;
        var count: usize = 1;
        while (tail.link.next) |next_link| {
            tail = @alignCast(@fieldParentPtr("link", next_link));
            count += 1;
        }

        // Splice: tail.next = free.head, free.head = xthread_head
        // This prepends entire xthread list to free in O(1) after the walk
        tail.link.next = @ptrCast(@alignCast(self.free_head));
        self.free_head = &head.link;

        self.used -|= @intCast(@min(count, self.used));

        // Decrement global pending counter
        _ = pending_xthread_free.fetchSub(count, .monotonic);

        return count;
    }

    /// Free a block from another thread (lock-free)
    pub inline fn xthreadFree(self: *Self, block: *Block) void {
        while (true) {
            const old_head = self.xthread_free.load(.acquire);
            // Store old_head's link pointer (convert Block* to Link*)
            block.link.next = if (old_head) |h| &h.link else null;

            if (self.xthread_free.cmpxchgWeak(
                old_head,
                block,
                .release,
                .acquire,
            ) == null) {
                // Success - increment global pending counter
                _ = pending_xthread_free.fetchAdd(1, .monotonic);
                break;
            }
        }
    }

    /// Pop from free list - simple linked list, no branching
    inline fn popFromFreeList(self: *Self) ?[*]u8 {
        const link: ?*list.IntrusiveLifoLink = @ptrCast(@alignCast(self.free_head));
        if (link) |l| {
            self.free_head = l.next;
            if (l.next) |next| {
                @prefetch(next, .{ .cache = .data, .locality = 3, .rw = .read });
            }
            return @ptrCast(l);
        }
        return null;
    }

    /// Push to free list - simple linked list
    inline fn pushToFreeList(self: *Self, block: [*]u8) void {
        const link: *list.IntrusiveLifoLink = @ptrCast(@alignCast(block));
        link.next = @ptrCast(@alignCast(self.free_head));
        self.free_head = link;
    }

    /// Push to local_free list - simple linked list
    inline fn pushToLocalFreeList(self: *Self, block: [*]u8) void {
        const link: *list.IntrusiveLifoLink = @ptrCast(@alignCast(block));
        link.next = @ptrCast(@alignCast(self.local_free_head));
        self.local_free_head = link;
    }

    /// Pop from local_free list
    inline fn popFromLocalFreeList(self: *Self) ?[*]u8 {
        const link: ?*list.IntrusiveLifoLink = @ptrCast(@alignCast(self.local_free_head));
        if (link) |l| {
            self.local_free_head = l.next;
            return @ptrCast(l);
        }
        return null;
    }

    /// Swap free and local_free lists
    inline fn swapFreeLists(self: *Self) void {
        const temp = self.free_head;
        self.free_head = self.local_free_head;
        self.local_free_head = temp;
    }

    /// Pop a free block - hot path with bump pointer
    pub inline fn popFreeBlock(self: *Self) ?*Block {
        // Hot path 1: pop from free list (recycled blocks)
        if (self.popFromFreeList()) |block_ptr| {
            @branchHint(.likely);
            self.used += 1;
            return @ptrCast(@alignCast(block_ptr));
        }

        // Hot path 2: bump allocate fresh contiguous block (no cache miss)
        if (self.reserved < self.capacity) {
            @branchHint(.likely);
            const start = self.page_start orelse return self.popFreeBlockSlow();
            const next_offset = self.reserved * self.block_size;
            @prefetch(start + next_offset, .{ .cache = .data, .locality = 3, .rw = .write });
            const block: *Block = @ptrCast(@alignCast(start + @as(usize, self.reserved) * self.block_size));
            self.reserved += 1;
            self.used += 1;
            return block;
        }

        // Cold path: collect local_free/xthread_free
        return self.popFreeBlockSlow();
    }

    inline fn popFreeBlockSlow(self: *Self) ?*Block {
        @branchHint(.unlikely);
        // Move local_free to free
        if (!self.localFreeEmpty()) {
            @branchHint(.likely);
            // Swap the lists - local_free becomes new free
            self.swapFreeLists();
            if (self.popFromFreeList()) |block_ptr| {
                @branchHint(.likely);
                self.used += 1;
                return @ptrCast(@alignCast(block_ptr));
            }
        }

        // Try to collect cross-thread free blocks
        if (self.collectXthreadFree() > 0) {
            if (self.popFromFreeList()) |block_ptr| {
                @branchHint(.likely);
                self.used += 1;
                return @ptrCast(@alignCast(block_ptr));
            }
        }

        // Fallback: bump allocate directly
        if (self.reserved < self.capacity) {
            @branchHint(.unlikely);
            const start = self.page_start orelse return null;
            const block: *Block = @ptrCast(@alignCast(start + @as(usize, self.reserved) * self.block_size));
            self.reserved += 1;
            self.used += 1;
            return block;
        }

        return null;
    }

    /// Push freed block directly to local_free list
    /// Note: uses regular decrement (not saturating) since used is always >= 1 when freeing
    pub inline fn pushFreeBlock(self: *Self, block: *Block) void {
        @setRuntimeSafety(false);
        self.pushToLocalFreeList(@ptrCast(block));
        self.used -= 1;
    }

    /// Reset page to initial state
    pub inline fn reset(self: *Self) void {
        self.used = 0;
        self.free_head = null;
        self.local_free_head = null;
        self.xthread_free.store(null, .release);
        self.reserved = 0;
        // Clear all flags but keep segment_idx
        const idx = self.segment_idx();
        self.setInfo(.{ .segment_idx = idx });
    }

    /// Get utilization ratio (0.0 to 1.0)
    pub inline fn utilization(self: *const Self) f32 {
        if (self.capacity == 0) return 0.0;
        return @as(f32, @floatFromInt(self.used)) / @as(f32, @floatFromInt(self.capacity));
    }
};

/// Calculate bin index for a given size in words
pub inline fn binFromWsize(wsize: usize) usize {
    if (wsize <= 1) return 1;
    if (wsize <= 8) return wsize;

    // For sizes > 8 words, use logarithmic binning
    const w = wsize - 1;
    const b = @as(usize, types.INTPTR_BITS - 1) - @clz(w);

    // b is in range [3, INTPTR_BITS)
    // bin = ((b << 2) | ((w >> (b - 2)) & 3)) - 3
    const shift: u6 = @intCast(b - 2);
    const bin = ((b << 2) | ((w >> shift) & 3)) - 3;

    return @min(bin, types.BIN_HUGE);
}

/// Calculate bin index for a given size in bytes
pub inline fn binFromSize(size: usize) usize {
    const wsize = (size + types.INTPTR_SIZE - 1) / types.INTPTR_SIZE;
    return binFromWsize(wsize);
}

/// Get block size for a bin index
/// Returns the MAXIMUM size that maps to this bin, ensuring block_size >= any size in the bin
pub fn blockSizeForBin(bin: usize) usize {
    if (bin <= 1) return types.INTPTR_SIZE;
    if (bin <= 8) return bin * types.INTPTR_SIZE;

    // Reverse the binning formula to get maximum wsize for this bin
    // In binFromWsize: bin + 3 = (b << 2) | ((w >> (b-2)) & 3)
    // where w = wsize - 1, and b = number of significant bits in w minus 1
    const b = (bin + 3) >> 2;
    const rem = (bin + 3) & 3;
    const shift_b: u6 = @intCast(b);
    const shift_b2: u6 = @intCast(b - 2);

    // Maximum w with b+1 bits where bits (b-1, b-2) = rem:
    // Set bit b (MSB), set bits (b-1, b-2) to rem, set all lower bits to 1
    const max_w = (@as(usize, 1) << shift_b) | (rem << shift_b2) | ((@as(usize, 1) << shift_b2) - 1);
    const wsize = max_w + 1;

    return wsize * types.INTPTR_SIZE;
}

/// Check if size qualifies for small allocation
pub inline fn isSmallSize(size: usize) bool {
    return size <= types.SMALL_OBJ_SIZE_MAX;
}

/// Check if size qualifies for medium allocation
pub inline fn isMediumSize(size: usize) bool {
    return size > types.SMALL_OBJ_SIZE_MAX and size <= types.MEDIUM_OBJ_SIZE_MAX;
}

/// Check if size qualifies for large allocation
pub inline fn isLargeSize(size: usize) bool {
    return size > types.MEDIUM_OBJ_SIZE_MAX and size <= types.LARGE_OBJ_SIZE_MAX;
}

/// Check if size qualifies for huge allocation
pub inline fn isHugeSize(size: usize) bool {
    return size > types.LARGE_OBJ_SIZE_MAX;
}

/// Get the page kind for a given block size
pub fn pageKindForSize(size: usize) Page.PageKind {
    if (isSmallSize(size)) return .PAGE_SMALL;
    if (isMediumSize(size)) return .PAGE_MEDIUM;
    if (isLargeSize(size)) return .PAGE_LARGE;
    return .PAGE_HUGE;
}

/// Find first page with free blocks in queue (iterates from tail via prev)
pub inline fn queueFindFree(queue: *Page.Queue) ?*Page {
    var current = queue.tail;
    while (current) |page| {
        if (page.hasFree()) {
            return page;
        }
        current = page.prev;
    }
    return null;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Page: basic initialization" {
    var page: Page = .{};
    var buffer: [1024]u8 align(64) = undefined;

    page.init(64, &buffer, 1024);

    try testing.expectEqual(@as(usize, 64), page.block_size);
    try testing.expectEqual(@as(u16, 16), page.capacity); // 1024 / 64
    try testing.expectEqual(@as(u16, 0), page.reserved); // starts at 0, extended lazily
    try testing.expectEqual(@as(u16, 0), page.used);
}

test "Page: flag accessors" {
    var page: Page = .{};

    // Test all flag accessors
    try testing.expect(!page.is_in_full());
    page.set_in_full(true);
    try testing.expect(page.is_in_full());

    try testing.expect(!page.is_huge());
    page.set_huge(true);
    try testing.expect(page.is_huge());

    try testing.expect(!page.is_commited());
    page.set_commited(true);
    try testing.expect(page.is_commited());

    // Test segment_idx packing
    page.set_segment_idx(100);
    try testing.expectEqual(@as(u10, 100), page.segment_idx());
    // Flags should still be set
    try testing.expect(page.is_in_full());
    try testing.expect(page.is_huge());
}

test "Page: popFreeBlock and pushFreeBlock" {
    var page: Page = .{};
    var buffer: [256]u8 align(16) = undefined;

    page.init(32, &buffer, 256);

    // Pop blocks via bump allocation
    const b1 = page.popFreeBlock();
    try testing.expect(b1 != null);
    try testing.expectEqual(@as(u16, 1), page.used);

    const b2 = page.popFreeBlock();
    try testing.expect(b2 != null);
    try testing.expectEqual(@as(u16, 2), page.used);

    // Push back (goes to local_free)
    page.pushFreeBlock(b1.?);
    try testing.expectEqual(@as(u16, 1), page.used);

    // Pop again (from local_free via slow path)
    const b3 = page.popFreeBlock();
    try testing.expect(b3 != null);
    try testing.expectEqual(@as(u16, 2), page.used);
}

test "Page: hasFree and allUsed" {
    var page: Page = .{};
    var buffer: [128]u8 align(8) = undefined;

    page.init(32, &buffer, 128); // small blocks, linked list mode
    // capacity=4 (128/32), reserved=0, so bump has room
    try testing.expect(page.hasFree());

    // Use all blocks via bump allocation
    _ = page.popFreeBlock();
    _ = page.popFreeBlock();
    _ = page.popFreeBlock();
    _ = page.popFreeBlock();

    try testing.expect(!page.hasFree());
    try testing.expect(page.allUsed());
}

test "Page: isMostlyEmpty" {
    var page: Page = .{};

    page.capacity = 64;
    page.used = 0;
    try testing.expect(page.isMostlyEmpty()); // 0/64 < 1/8

    page.used = 8;
    try testing.expect(page.isMostlyEmpty()); // 8/64 = 1/8

    page.used = 9;
    try testing.expect(!page.isMostlyEmpty()); // 9/64 > 1/8

    page.used = 32;
    try testing.expect(!page.isMostlyEmpty());
}

test "Page: utilization" {
    var page: Page = .{};

    page.capacity = 0;
    try testing.expectEqual(@as(f32, 0.0), page.utilization());

    page.capacity = 100;
    page.used = 0;
    try testing.expectEqual(@as(f32, 0.0), page.utilization());

    page.used = 50;
    try testing.expectEqual(@as(f32, 0.5), page.utilization());

    page.used = 100;
    try testing.expectEqual(@as(f32, 1.0), page.utilization());
}

test "binFromWsize: small sizes" {
    try testing.expectEqual(@as(usize, 1), binFromWsize(0));
    try testing.expectEqual(@as(usize, 1), binFromWsize(1));
    try testing.expectEqual(@as(usize, 2), binFromWsize(2));
    try testing.expectEqual(@as(usize, 8), binFromWsize(8));
}

test "binFromWsize: larger sizes" {
    // Sizes > 8 use logarithmic binning
    const bin9 = binFromWsize(9);
    const bin16 = binFromWsize(16);
    const bin32 = binFromWsize(32);

    try testing.expect(bin9 > 8);
    try testing.expect(bin16 > bin9);
    try testing.expect(bin32 > bin16);
    try testing.expect(bin32 <= types.BIN_HUGE);
}

test "isSmallSize, isMediumSize, isLargeSize" {
    try testing.expect(isSmallSize(64));
    try testing.expect(isSmallSize(types.SMALL_OBJ_SIZE_MAX));
    try testing.expect(!isSmallSize(types.SMALL_OBJ_SIZE_MAX + 1));

    try testing.expect(!isMediumSize(types.SMALL_OBJ_SIZE_MAX));
    try testing.expect(isMediumSize(types.SMALL_OBJ_SIZE_MAX + 1));
    try testing.expect(isMediumSize(types.MEDIUM_OBJ_SIZE_MAX));
    try testing.expect(!isMediumSize(types.MEDIUM_OBJ_SIZE_MAX + 1));

    try testing.expect(!isLargeSize(types.MEDIUM_OBJ_SIZE_MAX));
    try testing.expect(isLargeSize(types.MEDIUM_OBJ_SIZE_MAX + 1));
    try testing.expect(isLargeSize(types.LARGE_OBJ_SIZE_MAX));
}

test "pageKindForSize" {
    try testing.expectEqual(Page.PageKind.PAGE_SMALL, pageKindForSize(64));
    try testing.expectEqual(Page.PageKind.PAGE_MEDIUM, pageKindForSize(types.SMALL_OBJ_SIZE_MAX + 1));
    try testing.expectEqual(Page.PageKind.PAGE_LARGE, pageKindForSize(types.MEDIUM_OBJ_SIZE_MAX + 1));
    try testing.expectEqual(Page.PageKind.PAGE_HUGE, pageKindForSize(types.LARGE_OBJ_SIZE_MAX + 1));
}

test "Page.Queue: basic operations" {
    var queue: Page.Queue = .{};

    var p1: Page = .{};
    var p2: Page = .{};
    var p3: Page = .{};

    try testing.expect(queue.empty());

    queue.push(&p1);
    try testing.expect(!queue.empty());
    try testing.expectEqual(&p1, queue.tail);

    queue.push(&p2);
    try testing.expectEqual(&p2, queue.tail);

    queue.push(&p3);
    try testing.expectEqual(&p3, queue.tail);

    const popped = queue.pop();
    try testing.expectEqual(&p3, popped);
    try testing.expectEqual(&p2, queue.tail);
}

test "queueFindFree" {
    var queue: Page.Queue = .{};
    var buffer: [64]u8 align(8) = undefined;

    var p1: Page = .{ .capacity = 10, .used = 10, .reserved = 10, .block_size = 8 }; // Full (bump exhausted)
    var p2: Page = .{}; // Has free space
    p2.init(8, &buffer, 64);
    // p2 has reserved=0 < capacity, so hasFree() is true

    queue.push(&p1);
    queue.push(&p2);

    // p2 is pushed last so it's at tail, we iterate from tail via prev
    // p2 has free blocks so it should be found
    const found = queueFindFree(&queue);
    try testing.expectEqual(&p2, found);
}
