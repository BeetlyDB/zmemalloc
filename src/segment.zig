//! # Segment Management
//!
//! Segments are 32MB memory regions that contain multiple pages.
//! They are the unit of OS memory allocation and thread ownership.
//!
//! ## Memory Hierarchy
//!
//! ```
//! Arena (optional, 256MB+)
//!   └── Segment (32MB, aligned)
//!         ├── Segment Header (metadata, pages array)
//!         ├── Page 0 (64KB for small, 512KB for medium)
//!         ├── Page 1
//!         └── ...
//! ```
//!
//! ## Segment Types by Page Size
//!
//! - **Small**: 64KB pages, for blocks 8-1024 bytes
//! - **Medium**: 512KB pages, for blocks 1KB-128KB
//! - **Large**: Single page spanning full segment, for blocks 128KB-16MB
//! - **Huge**: Variable size segment, for blocks > 16MB
//!
//! ## Thread Ownership
//!
//! Each segment is owned by a single thread (stored in `thread_id`).
//! When a thread exits, its segments are "abandoned" and can be
//! reclaimed by other threads via the global abandoned queue.
//!
//! ## Pointer to Segment Lookup
//!
//! For standard segments (32MB aligned), finding the segment from a pointer
//! is a simple bit mask: `ptr & ~SEGMENT_MASK`.
//!
//! For huge segments (> 32MB), a global registry is used since the
//! segment boundary can't be determined from the pointer alone.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const queue = @import("queue.zig");
const page_mod = @import("page.zig");
const MemID = @import("mem.zig").MemID;
const OsAllocator = @import("os_allocator.zig");
const arena_mod = @import("arena.zig");
const util = @import("util.zig");
const os_mod = @import("os.zig");
const assert = util.assert;
const subproc_m = @import("subproc.zig");
const Mutex = @import("mutex.zig").Mutex;

const Atomic = std.atomic.Value;
const Page = page_mod.Page;

/// Registry for tracking huge segments (> 32MB).
/// This allows fromPtr to find the correct segment for pointers > 32MB into an allocation.
/// TODO: handle mor huge segments and use map instead of linier search of array
pub const HugeSegmentRegistry = struct {
    const MAX_HUGE_SEGMENTS: usize = 512;

    entries: [MAX_HUGE_SEGMENTS]Entry = [_]Entry{.{}} ** MAX_HUGE_SEGMENTS,
    count: Atomic(usize) = Atomic(usize).init(0),

    lock: Mutex = .{},

    const Entry = struct {
        start: usize = 0, // Segment start address
        end: usize = 0, // Segment end address (start + segment_size)
        segment: ?*Segment = null,
    };

    /// Register a huge segment
    pub fn register(self: *HugeSegmentRegistry, segment: *Segment) void {
        self.lock.lock();
        defer self.lock.unlock();

        const idx = self.count.load(.acquire);
        if (idx >= MAX_HUGE_SEGMENTS) return; // Registry full

        const start = @intFromPtr(segment);
        self.entries[idx] = .{
            .start = start,
            .end = start + segment.segment_size,
            .segment = segment,
        };
        self.count.store(idx + 1, .release);
    }

    /// Unregister a huge segment
    pub fn unregister(self: *HugeSegmentRegistry, segment: *Segment) void {
        self.lock.lock();
        defer self.lock.unlock();

        const target = @intFromPtr(segment);
        const cnt = self.count.load(.acquire);
        for (0..cnt) |i| {
            if (self.entries[i].start == target) {
                // Move last entry to this slot
                if (i < cnt - 1) {
                    self.entries[i] = self.entries[cnt - 1];
                }
                self.entries[cnt - 1] = .{};
                self.count.store(cnt - 1, .release);
                return;
            }
        }
    }

    /// Find segment containing the given address
    pub fn find(self: *HugeSegmentRegistry, addr: usize) ?*Segment {
        const cnt = self.count.load(.acquire);
        for (0..cnt) |i| {
            const entry = self.entries[i];
            if (addr >= entry.start and addr < entry.end) {
                return entry.segment;
            }
        }
        return null;
    }
};

/// Global registry for huge segments
var huge_registry: HugeSegmentRegistry = .{};

// =============================================================================
// Segment Structure
// =============================================================================

pub const Segment = struct {
    const Self = @This();

    // Constant fields (set at allocation)
    memid: MemID = .{},
    allow_decommit: bool = false,
    allow_purge: bool = false,
    segment_size: usize = 0,
    subproc: ?*anyopaque = null,

    // Intrusive doubly-linked list for segment queues
    next: ?*Self = null,
    prev: ?*Self = null,

    // Link for QueueType (spans list)
    link: queue.IntrusiveLifo(Self).Link = .{},

    // Link for all_segments list (tracks ALL segments owned by thread)
    all_next: ?*Self = null,
    all_prev: ?*Self = null,

    // Segment state
    was_reclaimed: bool = false,
    dont_free: bool = false,
    in_free_queue: bool = false, // Track queue membership (avoids O(n) hasItem)

    abandoned: usize = 0,
    abandoned_visits: usize = 0,

    used: usize = 0,
    capacity: usize = 0,
    segment_info_size: usize = 0,

    // For abandoned segments outside arenas
    abandoned_os_next: ?*Self = null,
    abandoned_os_prev: ?*Self = null,

    // Layout optimized for free() access - cache line aligned
    // Use usize for thread_id to allow TLD address (faster than syscall)
    thread_id: Atomic(usize) align(std.atomic.cache_line) = .init(0),
    /// Owning thread's heap - atomic for safe cross-thread access in xthreadFree
    heap: Atomic(?*@import("heap.zig").Heap) = .init(null),
    page_shift: usize = 0,
    page_kind: PageKind = .small,
    header_slices: usize = 0, // Pre-computed for fast pageFromPtr
    first_free_page: u16 = 0, // Hint for next free page (avoids O(n) scan)
    pages: [types.SLICES_PER_SEGMENT]Page = undefined,

    pub const PageKind = enum(u8) {
        small, // 64 KiB pages
        medium, // 512 KiB pages
        large, // full segment page
        huge, // variable size, 1 per segment
    };

    // =========================================================================
    // Size Calculations
    // =========================================================================

    pub inline fn pageSize(self: *const Self) usize {
        return if (self.capacity > 1)
            @as(usize, 1) << @intCast(self.page_shift)
        else
            self.segment_size;
    }

    pub inline fn rawPageSize(self: *const Self) usize {
        return if (self.page_kind == .huge)
            self.segment_size
        else
            @as(usize, 1) << @intCast(self.page_shift);
    }

    /// Get raw start of page memory
    pub inline fn rawPageStart(self: *const Self, pg: *const Page, page_size_out: ?*usize) [*]u8 {
        const base: [*]u8 = @ptrFromInt(@intFromPtr(self));

        // For large/huge pages (capacity=1), data starts right after aligned header
        if (self.capacity == 1) {
            const psize = self.segment_size - self.segment_info_size;
            if (page_size_out) |ps| ps.* = psize;
            return base + self.segment_info_size;
        }

        // Small/medium: multiple pages per segment
        const psize_raw = @as(usize, 1) << @intCast(self.page_shift);

        // Calculate how many page-sized slices the segment header uses
        const info_slices = (self.segment_info_size + psize_raw - 1) / psize_raw;

        // Actual page index accounting for header slices
        const effective_idx = pg.segment_idx() + info_slices;
        const p = base + effective_idx * psize_raw;

        if (page_size_out) |ps| ps.* = psize_raw;
        return p;
    }

    /// Get start of page memory (adjusted for block alignment)
    pub inline fn pageStart(self: *const Self, pg: *const Page, page_size_out: ?*usize) [*]u8 {
        var psize: usize = undefined;
        var p = self.rawPageStart(pg, &psize);

        const block_size = pg.block_size;
        if (block_size > 0 and block_size <= types.MAX_ALIGN_GUARANTEE) {
            const adjust = block_size - (@intFromPtr(p) % block_size);
            if (adjust < block_size and psize >= block_size + adjust) {
                p += adjust;
                psize -= adjust;
            }
        }

        if (page_size_out) |ps| ps.* = psize;
        return p;
    }

    // =========================================================================
    // Pointer to Segment/Page
    // =========================================================================

    /// Get segment from any pointer within it.
    /// For huge segments (> 32MB), uses a global registry to find the correct segment.
    pub inline fn fromPtr(p: *const anyopaque) *Self {
        const addr = @intFromPtr(p);
        const masked = addr & ~types.SEGMENT_MASK;
        const candidate: *Self = @ptrFromInt(masked);

        // Check if this segment contains our pointer
        // For non-huge segments, segment_size == SEGMENT_SIZE and this always works
        // For huge segments, we need to verify the pointer is within bounds
        if (masked + candidate.segment_size > addr) {
            @branchHint(.likely);
            return candidate;
        }

        // Slow path: pointer is > 32MB into a huge segment
        // Use the global registry to find the correct segment
        if (huge_registry.find(addr)) |segment| {
            @branchHint(.cold);
            return segment;
        }

        return candidate;
    }

    /// Get page from pointer (within segment's data area) - optimized
    pub inline fn pageFromPtr(p: *const anyopaque) *Page {
        return pageFromPtrWithSegment(fromPtr(p), p);
    }

    /// Get page when segment is already known - optimized
    /// Uses comptime shifts for small/medium (most common), avoids page_shift load
    pub inline fn pageFromPtrWithSegment(segment: *Self, p: *const anyopaque) *Page {
        // Large/huge pages: capacity == 1, always page 0
        if (segment.capacity == 1) {
            @branchHint(.unlikely);
            return &segment.pages[0];
        }

        const diff = @intFromPtr(p) - @intFromPtr(segment);

        // Use comptime constant shifts based on page_kind (avoids memory load)
        const idx = switch (segment.page_kind) {
            .small => diff >> types.SMALL_PAGE_SHIFT,
            .medium => diff >> types.MEDIUM_PAGE_SHIFT,
            .large, .huge => 0, // Should not reach here due to capacity check above
        };

        const page_idx = idx -| segment.header_slices;
        return &segment.pages[page_idx];
    }

    pub inline fn hasFree(self: *const Self) bool {
        return self.used < self.capacity;
    }

    // =========================================================================
    // Page Operations
    // =========================================================================

    pub inline fn pagePurge(self: *Self, pg: *Page) void {
        if (!self.allow_purge) return;
        if (!pg.is_commited()) return;

        var psize: usize = undefined;
        const start = self.rawPageStart(pg, &psize);

        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(start);
        std.posix.madvise(ptr, psize, std.posix.MADV.FREE) catch {};
        pg.set_commited(false);
    }

    pub inline fn pageEnsureCommitted(self: *Self, pg: *Page) bool {
        if (pg.is_commited()) return true;
        if (!self.allow_decommit) return true;

        // Memory is already mapped, just mark as committed
        pg.set_commited(true);
        pg.used = 0;
        // pg.free = .init();
        // pg.local_free = .init();
        pg.set_zero_init(false);
        return true;
    }

    pub inline fn pageClear(self: *Self, pg: *Page) void {
        const page_idx = pg.segment_idx();
        // Clear all flags, keep segment_idx
        pg.flags_and_idx = @bitCast(Page.PackedInfo{ .segment_idx = page_idx });
        pg.xthread_free.store(null, .release);
        pg.free_head = null;
        pg.local_free_head = null;
        pg.block_size = 0;
        pg.page_start = null;
        pg.capacity = 0;
        pg.reserved = 0;
        pg.used = 0;
        pg.next = null;
        pg.prev = null;

        // Update hint if this page comes before current hint
        if (page_idx < self.first_free_page) {
            self.first_free_page = @intCast(page_idx);
        }

        if (self.allow_purge) {
            self.pagePurge(pg);
        }
    }

    pub fn pageClaim(self: *Self, pg: *Page) bool {
        if (!self.pageEnsureCommitted(pg)) return false;

        pg.set_segment_in_use(true);
        self.used += 1;
        return true;
    }

    pub fn findFreePage(self: *Self) ?*Page {
        const cap = self.capacity;
        if (cap == 0) return null;

        // Start search from hint (first_free_page)
        var i: usize = self.first_free_page;
        var checked: usize = 0;

        while (checked < cap) : ({
            i = if (i + 1 >= cap) 0 else i + 1;
            checked += 1;
        }) {
            const pg = &self.pages[i];
            @prefetch(pg, .{ .cache = .data, .locality = 3, .rw = .read });
            if (!pg.is_segment_in_use()) {
                const next_i = if (i + 1 >= self.capacity) 0 else i + 1;
                @prefetch(&self.pages[next_i], .{ .cache = .data, .locality = 3 });
                if (self.pageClaim(pg)) {
                    // Update hint to next position
                    self.first_free_page = @intCast(if (i + 1 >= cap) 0 else i + 1);
                    return pg;
                }
            }
        }
        return null;
    }

    pub inline fn calculateSizes(capacity: usize, required: usize) struct { segment_size: usize, info_size: usize } {
        _ = capacity; // Capacity is already accounted for in @sizeOf(Segment)
        // Segment struct includes pages: [512]Page array, so just use struct size
        const minsize = @sizeOf(Self) + 16;
        const info_size = util.alignUp(minsize, 16 * types.MAX_ALIGN_SIZE);

        const segment_size = if (required == 0)
            types.SEGMENT_SIZE
        else
            util.alignUp(required + info_size, 256 * types.KiB);

        return .{ .segment_size = segment_size, .info_size = info_size };
    }

    pub inline fn capacityForKind(kind: PageKind, page_shift: usize) usize {
        const total_slices = types.SEGMENT_SIZE >> @intCast(page_shift);

        // Reserve slices for segment header
        // Header size is approximately @sizeOf(Segment) aligned up
        const page_size = @as(usize, 1) << @intCast(page_shift);
        const header_size = @sizeOf(Self);
        const header_slices = (header_size + page_size - 1) / page_size;

        return switch (kind) {
            .huge, .large => 1,
            .medium, .small => total_slices -| header_slices,
        };
    }

    // =========================================================================
    // Abandonment
    // =========================================================================

    /// Check if segment is abandoned
    pub inline fn isAbandoned(self: *const Self) bool {
        return self.thread_id.load(.acquire) == 0;
    }

    /// Mark segment as abandoned (called when thread exits)
    pub fn markAbandoned(self: *Self) void {
        self.thread_id.store(0, .release);
        self.abandoned_visits = 0;
    }

    /// Try to reclaim this segment for the current thread
    pub inline fn tryReclaim(self: *Self, thread_id: usize) bool {
        // Try to atomically claim the segment
        const old = self.thread_id.cmpxchgStrong(
            0, // expected: abandoned
            thread_id,
            .acq_rel,
            .acquire,
        );
        return old == null; // success if was 0
    }

    /// Check if segment can be reclaimed (has free pages)
    pub inline fn canReclaim(self: *const Self) bool {
        return self.isAbandoned() and self.hasFree();
    }

    // =========================================================================
    // Purge Operations (simplified - no delayed scheduling)
    // =========================================================================

    /// Purge all unused pages in segment
    pub inline fn purgeUnusedPages(self: *Self) usize {
        if (!self.allow_purge) return 0;
        var purged: usize = 0;
        for (0..self.capacity) |i| {
            const pg = &self.pages[i];
            if (!pg.is_segment_in_use() and pg.is_commited()) {
                self.pagePurge(pg);
                purged += 1;
            }
        }
        return purged;
    }

    // =========================================================================
    // Visit/Inspection
    // =========================================================================

    /// Visit all blocks in a page
    pub fn visitPageBlocks(
        self: *const Self,
        pg: *const Page,
        visitor: *const fn (block: *anyopaque, block_size: usize, ctx: ?*anyopaque) bool,
        ctx: ?*anyopaque,
    ) bool {
        if (pg.block_size == 0) return true;

        var psize: usize = undefined;
        const start = self.pageStart(pg, &psize);
        const block_size = pg.block_size;

        if (block_size == 0 or psize < block_size) return true;

        const num_blocks = psize / block_size;
        var block_ptr = start;
        for (0..num_blocks) |_| {
            if (!visitor(block_ptr, block_size, ctx)) {
                return false;
            }
            block_ptr += block_size;
        }
        return true;
    }

    /// Visit all pages in segment
    pub fn visitPages(
        self: *const Self,
        visitor: *const fn (pg: *const Page, ctx: ?*anyopaque) bool,
        ctx: ?*anyopaque,
    ) bool {
        for (0..self.capacity) |i| {
            const pg = &self.pages[i];
            if (pg.is_segment_in_use()) {
                if (!visitor(pg, ctx)) {
                    return false;
                }
            }
        }
        return true;
    }
    //DEBUG
    /// Check segment validity (for debugging)
    pub fn isValid(self: *const Self) bool {
        if (self.capacity == 0) return false;
        if (self.capacity > types.SLICES_PER_SEGMENT) return false;
        if (self.used > self.capacity) return false;
        if (self.segment_size == 0) return false;

        var used_count: usize = 0;
        for (0..self.capacity) |i| {
            const pg = &self.pages[i];
            if (pg.is_segment_in_use()) {
                used_count += 1;
            }
        }
        return used_count == self.used;
    }
};

// Type aliases - defined outside struct to avoid circular dependency
pub const SegmentList = queue.IntrusiveLifo(Segment);
pub const SegmentAbandonedQueue = queue.Intrusive(Segment, .abandoned_os_next, .abandoned_os_prev);
pub const SegmentQueue = queue.DoublyLinkedListType(Segment, .next, .prev);
pub const AllSegmentsList = queue.DoublyLinkedListType(Segment, .all_next, .all_prev);

// =============================================================================
// Segments Thread-Local Data
// =============================================================================

pub const SegmentsTLD = struct {
    const Self = @This();

    // Free segment queues for reuse (by page kind)
    small_free: SegmentQueue = .{},
    medium_free: SegmentQueue = .{},
    large_free: SegmentQueue = .{}, // Cache for large segments (one page per segment)

    // List of ALL segments owned by this thread (for xthread_free collection)
    all_segments: AllSegmentsList = .{},

    // spans per size bin
    spans: [types.SEGMENTS_BIN_MAX + 1]SegmentList = [_]SegmentList{SegmentList.init()} ** (types.SEGMENTS_BIN_MAX + 1),

    count: usize = 0,
    peak_count: usize = 0,
    current_size: usize = 0,
    peak_size: usize = 0,
    reclaim_count: usize = 0,
    subproc: ?*subproc_m.Subproc = null,
    // stats: ?*Stats = null,

    os_alloc: std.mem.Allocator = undefined,

    // Owning heap (for xthread_free collection)
    heap: ?*@import("heap.zig").Heap = null,

    // Fast thread ID (TLD address) - set during init, avoids syscall
    thread_id: usize = 0,

    // Counter for empty segment tracking (avoids O(n) counting)
    empty_segment_count: usize = 0,

    // =========================================================================
    // Initialization
    // =========================================================================

    pub fn init(os_alloc: std.mem.Allocator) Self {
        return .{ .os_alloc = os_alloc };
    }

    // =========================================================================
    // Queue Management
    // =========================================================================

    pub inline fn freeQueueOfKind(self: *Self, kind: Segment.PageKind) ?*SegmentQueue {
        return switch (kind) {
            .small => &self.small_free,
            .medium => &self.medium_free,
            .large => &self.large_free,
            .huge => null, // Huge segments are unique-sized, not cacheable
        };
    }

    pub inline fn freeQueue(self: *Self, segment: *const Segment) ?*SegmentQueue {
        return self.freeQueueOfKind(segment.page_kind);
    }

    pub inline fn removeFromFreeQueue(self: *Self, segment: *Segment) void {
        if (!segment.in_free_queue) return; // O(1) check instead of O(n) hasItem
        const q = self.freeQueue(segment) orelse return;
        q.remove(segment);
        segment.in_free_queue = false;
    }

    pub inline fn insertInFreeQueue(self: *Self, segment: *Segment) void {
        if (segment.in_free_queue) return; // Already in queue
        const q = self.freeQueue(segment) orelse return;
        q.push(segment);
        segment.in_free_queue = true;
    }

    // =========================================================================
    // Tracking
    // =========================================================================

    pub inline fn trackSize(self: *Self, segment_size: isize) void {
        if (segment_size >= 0) {
            self.count += 1;
            self.current_size += @intCast(segment_size);
        } else {
            self.count -|= 1;
            self.current_size -|= @intCast(-segment_size);
        }
        if (self.count > self.peak_count) self.peak_count = self.count;
        if (self.current_size > self.peak_size) self.peak_size = self.current_size;
    }

    // =========================================================================
    // Segment Allocation/Free
    // =========================================================================

    pub fn allocSegment(
        self: *Self,
        required: usize,
        page_kind: Segment.PageKind,
        page_shift: usize,
    ) ?*Segment {
        const capacity = Segment.capacityForKind(page_kind, page_shift);
        const sizes = Segment.calculateSizes(capacity, required);

        var mem_id: MemID = .{};

        // For huge segments, use direct OS allocation to avoid arena complexity
        // Arena blocks are 32MB which may not handle huge segments well
        if (page_kind == .huge) {
            const alignment = std.mem.Alignment.fromByteUnits(types.SEGMENT_ALIGN);
            const os_ptr = self.os_alloc.rawAlloc(sizes.segment_size, alignment, @returnAddress()) orelse return null;
            mem_id = MemID.create_os(
                os_ptr[0..sizes.segment_size],
                true, // committed
                true, // zero
                false, // large
            );
            return self.initSegment(os_ptr, mem_id, sizes.segment_size, sizes.info_size, page_kind, page_shift, capacity);
        }

        // Try to allocate from arenas first for non-huge segments
        const raw_ptr = self.allocFromArena(sizes.segment_size, &mem_id) orelse {
            // Fallback to direct OS allocation if arena allocation fails
            const alignment = std.mem.Alignment.fromByteUnits(types.SEGMENT_ALIGN);
            const os_ptr = self.os_alloc.rawAlloc(sizes.segment_size, alignment, @returnAddress()) orelse return null;
            mem_id = MemID.create_os(
                os_ptr[0..sizes.segment_size],
                true, // committed
                true, // zero
                false, // large
            );
            return self.initSegment(os_ptr, mem_id, sizes.segment_size, sizes.info_size, page_kind, page_shift, capacity);
        };

        return self.initSegment(raw_ptr, mem_id, sizes.segment_size, sizes.info_size, page_kind, page_shift, capacity);
    }

    /// Try to allocate segment memory from arena
    inline fn allocFromArena(self: *Self, size: usize, mem_id: *MemID) ?[*]u8 {
        _ = self;
        const arenas = arena_mod.globalArenas();

        // Try existing arenas first
        if (arenas.tryAlloc(
            size,
            types.SEGMENT_ALIGN,
            true, // commit
            false, // allow_large
            null, // any arena
            mem_id,
        )) |ptr| {
            return ptr;
        }

        // No space in existing arenas - try to create a new one
        // Reserve a large arena (e.g. 256 MiB = 8 segments worth)
        const arena_size = @max(size, 8 * types.SEGMENT_SIZE);
        const arena = arena_mod.reserve(
            arena_size,
            true, // commit
            false, // allow_large
            false, // exclusive
            -1, // any NUMA node
        ) catch return null;

        // Now try to allocate from the new arena
        return arena.tryAlloc(
            arena_mod.blockCountForSize(size),
            true,
            false,
            mem_id,
        );
    }

    /// Initialize segment structure
    fn initSegment(
        self: *Self,
        raw_ptr: [*]u8,
        mem_id: MemID,
        segment_size: usize,
        info_size: usize,
        page_kind: Segment.PageKind,
        page_shift: usize,
        capacity: usize,
    ) *Segment {
        const segment: *Segment = @ptrCast(@alignCast(raw_ptr));
        // Pre-compute header_slices for fast pageFromPtr
        const page_size_local = @as(usize, 1) << @as(u6, @intCast(page_shift));
        const header_slices = (info_size + page_size_local - 1) / page_size_local;

        segment.* = .{
            .memid = mem_id,
            .allow_decommit = !mem_id.flags.is_pinned,
            .allow_purge = !mem_id.flags.is_pinned,
            .segment_size = segment_size,
            .segment_info_size = info_size,
            .page_kind = page_kind,
            .page_shift = page_shift,
            .header_slices = header_slices,
            .capacity = capacity,
        };

        segment.thread_id.store(self.thread_id, .release);

        // Initialize pages - explicitly reset all fields to avoid garbage values
        for (0..capacity) |i| {
            var pg = &segment.pages[i];
            pg.* = .{
                .free_head = null,
                .page_start = null,
                .block_size = 0,
                .capacity = 0,
                .reserved = 0,
                .used = 0,
                .flags_and_idx = 0,
                .local_free_head = null,
                .xthread_free = .init(null),
                .next = null,
                .prev = null,
            };
            // Set segment_idx and initial flags
            pg.set_segment_idx(@intCast(i));
            pg.set_commited(mem_id.flags.initially_committed);
            pg.set_zero_init(mem_id.flags.initially_zero);
        }

        // Register huge segments (> 32MB) in the global registry
        if (page_kind == .huge and segment_size > types.SEGMENT_SIZE) {
            huge_registry.register(segment);
        }

        // Set heap pointer for xthread_free collection
        segment.heap.store(self.heap, .release);

        // Add to all_segments list for iteration during collection
        self.all_segments.push(segment);

        self.trackSize(@intCast(segment_size));
        return segment;
    }

    pub inline fn freeSegment(self: *Self, segment: *Segment, force: bool) void {
        // Unregister from huge registry if applicable
        if (segment.page_kind == .huge and segment.segment_size > types.SEGMENT_SIZE) {
            huge_registry.unregister(segment);
        }

        // Update empty segment counter
        if (segment.used == 0) {
            self.empty_segment_count -|= 1;
        }

        self.removeFromFreeQueue(segment);

        // Remove from all_segments list
        self.all_segments.remove(segment);

        // Purge all uncommitted pages
        if (force and !segment.memid.flags.is_pinned) {
            for (0..segment.capacity) |i| {
                const pg = &segment.pages[i];
                if (!pg.is_segment_in_use() and pg.is_commited()) {
                    segment.pagePurge(pg);
                }
            }
        }

        const segment_size = segment.segment_size;
        segment.thread_id.store(0, .release);
        self.trackSize(-@as(isize, @intCast(segment_size)));

        if (segment.was_reclaimed) {
            self.reclaim_count -|= 1;
        }

        // Free based on memory kind
        if (segment.memid.memkind == .MEM_ARENA) {
            // Free back to arena
            const arena_info = segment.memid.arena_info();
            const arenas = arena_mod.globalArenas();
            if (arenas.get(@intCast(arena_info.id))) |arena| {
                const block_count = arena_mod.blockCountForSize(segment_size);
                arena.free(arena_info.block_index, block_count, segment.memid.flags.initially_committed);
            }
        } else {
            // Direct OS free
            const ptr: [*]align(std.heap.page_size_min) u8 = @ptrCast(@alignCast(segment));
            self.os_alloc.free(ptr[0..segment_size]);
        }
    }

    // =========================================================================
    // Page Allocation
    // =========================================================================

    /// Threshold for segment count before we start aggressively freeing empty segments.
    /// Each segment is 32MB, so 8 segments = 256MB of cached memory.
    const SEGMENT_CACHE_THRESHOLD: usize = 8;

    /// How many empty segments to keep cached per queue type.
    const EMPTY_SEGMENT_CACHE_MAX: usize = 2;

    /// Purge delay in milliseconds (mimalloc style)
    const PURGE_DELAY_MS: i64 = 10;

    pub inline fn freePage(self: *Self, pg: *Page, force: bool) void {
        const segment = Segment.fromPtr(pg);
        const was_used = segment.used;
        segment.pageClear(pg);
        segment.used -= 1;

        if (segment.used == 0) {
            if (force) {
                // Only free segment if explicitly forced
                self.freeSegment(segment, true);
            } else if (self.empty_segment_count >= EMPTY_SEGMENT_CACHE_MAX * 3 and self.count > SEGMENT_CACHE_THRESHOLD) {
                // Too many empty segments cached, free this one immediately
                self.freeSegment(segment, true);
            } else {
                // Cache empty segment for reuse
                self.empty_segment_count += 1;
                self.insertInFreeQueue(segment);
            }
        } else if (was_used == segment.capacity) {
            // Segment was full, now has free space - add to queue
            self.insertInFreeQueue(segment);
        }
    }

    inline fn tryAllocInQueue(self: *Self, kind: Segment.PageKind) ?*Page {
        const q = self.freeQueueOfKind(kind) orelse return null;

        var segment = q.tail;
        while (segment) |seg| {
            const next_seg = seg.prev;
            if (next_seg) |ns| {
                @prefetch(ns, .{ .cache = .data, .locality = 3, .rw = .read });
                // Prefetch pages array area
                @prefetch(@as([*]u8, @ptrCast(ns)) + @offsetOf(Segment, "pages"), .{ .cache = .data, .locality = 3 });
            }
            @prefetch(seg, .{ .cache = .data, .locality = 3, .rw = .read });
            if (seg.hasFree()) {
                if (seg.findFreePage()) |pg| {
                    @setEvalBranchQuota(100_000);
                    if (!seg.hasFree()) {
                        self.removeFromFreeQueue(seg);
                    }
                    return pg;
                }
            }
            segment = seg.prev;
        }
        return null;
    }

    inline fn allocPageInKind(
        self: *Self,
        block_size: usize,
        kind: Segment.PageKind,
        page_shift: usize,
    ) ?*Page {
        // Try existing segments first
        // Note: tryAllocInQueue -> findFreePage -> pageClaim already claims the page
        // and removes segment from free queue if it becomes full
        if (self.tryAllocInQueue(kind)) |pg| {
            // Page is already claimed by findFreePage, just reinitialize for new block size
            pg.block_size = block_size;
            pg.page_start = null; // Will be set on first alloc
            pg.capacity = 0;
            pg.reserved = 0;
            pg.used = 0;
            pg.free_head = null;
            pg.local_free_head = null;
            pg.xthread_free.store(null, .release);
            pg.next = null;
            pg.prev = null;
            // Reset flags but keep segment_idx and segment_in_use
            pg.set_in_full(false);
            pg.set_in_bin(false);

            return pg;
        }

        // Allocate new segment
        const segment = self.allocSegment(0, kind, page_shift) orelse return null;
        self.insertInFreeQueue(segment);

        // findFreePage already calls pageClaim internally
        const pg = segment.findFreePage() orelse {
            self.freeSegment(segment, true);
            return null;
        };

        if (!segment.hasFree()) {
            self.removeFromFreeQueue(segment);
        }

        pg.block_size = block_size;
        return pg;
    }

    pub inline fn allocSmallPage(self: *Self, block_size: usize) ?*Page {
        return self.allocPageInKind(block_size, .small, types.SMALL_PAGE_SHIFT);
    }

    pub inline fn allocMediumPage(self: *Self, block_size: usize) ?*Page {
        return self.allocPageInKind(block_size, .medium, types.MEDIUM_PAGE_SHIFT);
    }

    pub inline fn allocLargePage(self: *Self, block_size: usize) ?*Page {
        // Try to reuse a cached large segment first
        if (self.large_free.pop()) |seg| {
            seg.in_free_queue = false;
            const pg = &seg.pages[0];
            // Reinitialize the page
            pg.set_segment_in_use(true);
            pg.block_size = block_size;
            pg.page_start = null; // Will be set on first alloc
            pg.capacity = 0;
            pg.reserved = 0;
            pg.used = 0;
            pg.free_head = null;
            pg.local_free_head = null;
            pg.xthread_free.store(null, .release);
            pg.next = null;
            pg.prev = null;
            pg.set_in_full(false);
            pg.set_in_bin(false);
            seg.used = 1;
            return pg;
        }

        // Allocate new segment
        const segment = self.allocSegment(0, .large, types.SEGMENT_SHIFT) orelse return null;

        const pg = &segment.pages[0];
        if (!segment.pageClaim(pg)) {
            self.freeSegment(segment, true);
            return null;
        }

        pg.block_size = block_size;
        return pg;
    }

    pub fn allocHugePage(self: *Self, size: usize) ?*Page {
        const segment = self.allocSegment(size, .huge, types.SEGMENT_SHIFT) orelse return null;

        const pg = &segment.pages[0];
        if (!segment.pageClaim(pg)) {
            self.freeSegment(segment, true);
            return null;
        }

        pg.block_size = size;
        pg.set_huge(true);
        return pg;
    }

    /// Main entry point for page allocation
    pub inline fn allocPage(self: *Self, block_size: usize) ?*Page {
        if (block_size <= types.SMALL_OBJ_SIZE_MAX) {
            return self.allocSmallPage(block_size);
        } else if (block_size <= types.MEDIUM_OBJ_SIZE_MAX) {
            return self.allocMediumPage(block_size);
        } else if (block_size <= types.LARGE_OBJ_SIZE_MAX) {
            return self.allocLargePage(block_size);
        } else {
            return self.allocHugePage(block_size);
        }
    }

    /// Abandon a segment (when thread exits or heap is destroyed)
    pub fn abandonSegment(self: *Self, segment: *Segment) void {
        self.removeFromFreeQueue(segment);
        segment.markAbandoned();
        segment.abandoned += 1;

        //TODO: Add to subproc abandoned list if available
        // (handled by caller with proper locking)
    }

    /// Try to reclaim an abandoned segment
    pub fn tryReclaimSegment(self: *Self, segment: *Segment) bool {
        if (!segment.tryReclaim(self.thread_id)) {
            return false;
        }

        // Successfully reclaimed
        segment.was_reclaimed = true;
        self.reclaim_count += 1;

        if (segment.hasFree()) {
            self.insertInFreeQueue(segment);
        }

        return true;
    }

    /// Try to reclaim abandoned segments from a queue
    pub fn tryReclaimFromQueue(self: *Self, abandoned_queue: *SegmentAbandonedQueue, max_count: usize) usize {
        var reclaimed: usize = 0;
        var segment = abandoned_queue.head;

        while (segment) |seg| {
            const next = seg.abandoned_os_next;

            if (seg.canReclaim()) {
                abandoned_queue.remove(seg);
                if (self.tryReclaimSegment(seg)) {
                    reclaimed += 1;
                    if (reclaimed >= max_count) break;
                }
            }

            segment = next;
        }

        return reclaimed;
    }

    /// Try to reclaim or allocate a page
    pub fn reclaimOrAllocPage(
        self: *Self,
        block_size: usize,
        abandoned_queue: ?*SegmentAbandonedQueue,
    ) ?*Page {
        // First try to reclaim abandoned segments
        if (abandoned_queue) |aq| {
            _ = self.tryReclaimFromQueue(aq, 8);
        }

        // Then try normal allocation
        return self.allocPage(block_size);
    }

    /// Collect and purge pages across all segments.
    /// When force=true, also frees empty segments back to the OS.
    pub inline fn collect(self: *Self, force: bool) usize {
        var collected: usize = 0;
        const now = std.time.milliTimestamp();

        // Collect segments to free (can't modify queue while iterating)
        var segments_to_free: [128]*Segment = undefined;
        var free_count: usize = 0;

        // Iterate through free queues and purge expired pages
        inline for ([_]*SegmentQueue{ &self.small_free, &self.medium_free, &self.large_free }) |q| {
            var segment = q.tail;
            while (segment) |seg| {
                const next = seg.prev;
                if (force) {
                    // If segment is completely empty, mark it for freeing
                    if (seg.used == 0 and free_count < segments_to_free.len) {
                        segments_to_free[free_count] = seg;
                        free_count += 1;
                    } else {
                        // Purge unused pages in non-empty segments
                        for (0..seg.capacity) |i| {
                            const pg = &seg.pages[i];
                            if (!pg.is_segment_in_use() and pg.is_commited()) {
                                seg.pagePurge(pg);
                                collected += 1;
                            }
                        }
                    }
                } else {
                    // Time-based purge check removed - simplified purge
                    _ = now;
                }
                segment = next;
            }
        }

        // Free empty segments (outside of iteration to avoid queue corruption)
        for (segments_to_free[0..free_count]) |seg| {
            self.freeSegment(seg, true);
            collected += 1;
        }

        return collected;
    }

    /// Collect xthread_free from ALL pages in ALL segments owned by this thread.
    /// This is critical for reclaiming memory freed by other threads.
    /// Returns total number of blocks collected.
    pub fn collectAllXthreadFree(self: *Self) usize {
        var total_collected: usize = 0;

        var segment = self.all_segments.tail;
        while (segment) |seg| {
            const next = seg.all_prev;

            for (0..seg.capacity) |i| {
                const pg = &seg.pages[i];
                if (pg.is_segment_in_use()) {
                    // Check if there are pending xthread frees
                    if (pg.xthread_free.load(.monotonic) != null) {
                        total_collected += pg.collectXthreadFree();
                    }
                }
            }

            segment = next;
        }

        return total_collected;
    }

    /// Schedule a page for delayed purging
    pub fn schedulePurge(self: *Self, pg: *Page, delay_ms: i64) void {
        const segment = Segment.fromPtr(pg);
        const expire = std.time.milliTimestamp() + delay_ms;
        segment.schedulePurge(pg, expire);
        _ = self;
    }

    /// Get current segment count within target
    pub inline fn isWithinTarget(self: *const Self, target: usize) bool {
        return self.count <= target;
    }

    /// Reduce memory usage by freeing empty segments
    pub inline fn reduceMemory(self: *Self, target_count: usize) usize {
        var freed: usize = 0;

        // Free empty segments from all queues
        inline for ([_]*SegmentQueue{ &self.small_free, &self.medium_free, &self.large_free }) |q| {
            while (!self.isWithinTarget(target_count)) {
                const seg = q.pop() orelse break;
                seg.in_free_queue = false;
                if (seg.used == 0) {
                    // Note: freeSegment already decrements empty_segment_count
                    self.freeSegment(seg, true);
                    freed += 1;
                } else {
                    seg.in_free_queue = true;
                    q.push(seg);
                    break;
                }
            }
        }

        return freed;
    }

    /// Free a huge page
    pub inline fn freeHugePage(self: *Self, pg: *Page) void {
        assert(pg.is_huge());
        const segment = Segment.fromPtr(pg);
        self.freeSegment(segment, true);
    }

    /// Reset huge page memory (decommit/recommit for zeroing)
    pub inline fn resetHugePage(self: *Self, pg: *Page) void {
        _ = self;
        assert(pg.is_huge());
        const segment = Segment.fromPtr(pg);

        var psize: usize = undefined;
        const start = segment.rawPageStart(pg, &psize);

        // Use madvise to reset memory
        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(start);
        std.posix.madvise(ptr, psize, std.posix.MADV.FREE) catch {};
        pg.set_zero_init(true);
    }

    // pub fn getStats(self: *const Self) struct {
    //     count: usize,
    //     peak_count: usize,
    //     current_size: usize,
    //     peak_size: usize,
    //     reclaim_count: usize,
    // } {
    //     return .{
    //         .count = self.count,
    //         .peak_count = self.peak_count,
    //         .current_size = self.current_size,
    //         .peak_size = self.peak_size,
    //         .reclaim_count = self.reclaim_count,
    //     };
    // }
};

// =============================================================================
// Tests
// =============================================================================

test "Segment: size calculations" {
    const testing = std.testing;

    if (types.INTPTR_SIZE == 8) {
        try testing.expectEqual(@as(usize, 32 * types.MiB), types.SEGMENT_SIZE);
        try testing.expectEqual(@as(usize, 64 * types.KiB), types.SEGMENT_SLICE_SIZE);
    }
}

test "Segment: fromPtr" {
    const testing = std.testing;

    // Test segment mask calculation
    const mask = types.SEGMENT_MASK;
    try testing.expect(mask > 0);
    try testing.expectEqual(types.SEGMENT_SIZE - 1, mask);
}

test "Segment: calculateSizes" {
    const testing = std.testing;

    // Standard segment (required = 0)
    const sizes1 = Segment.calculateSizes(64, 0);
    try testing.expectEqual(types.SEGMENT_SIZE, sizes1.segment_size);
    try testing.expect(sizes1.info_size > 0);

    // Huge segment
    const huge_required = 64 * types.MiB;
    const sizes2 = Segment.calculateSizes(1, huge_required);
    try testing.expect(sizes2.segment_size >= huge_required);
    try testing.expect(sizes2.segment_size % (256 * types.KiB) == 0);
}

test "Segment: PageKind enum" {
    const testing = std.testing;

    try testing.expectEqual(@as(u8, 0), @intFromEnum(Segment.PageKind.small));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(Segment.PageKind.medium));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(Segment.PageKind.large));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(Segment.PageKind.huge));
}

test "Segment: capacityForKind" {
    const testing = std.testing;

    try testing.expectEqual(@as(usize, 1), Segment.capacityForKind(.huge, types.SEGMENT_SHIFT));
    try testing.expectEqual(@as(usize, 1), Segment.capacityForKind(.large, types.SEGMENT_SHIFT));

    const small_capacity = Segment.capacityForKind(.small, types.SMALL_PAGE_SHIFT);
    try testing.expect(small_capacity > 1);
}

test "Segment: queue operations with Intrusive" {
    const testing = std.testing;

    var small_free: SegmentQueue = .{};
    var medium_free: SegmentQueue = .{};

    // Queue should be empty initially
    try testing.expect(small_free.empty());
    try testing.expect(medium_free.empty());

    // Create mock segments
    var seg1: Segment = .{};
    var seg2: Segment = .{};

    // Push to queue
    small_free.push(&seg1);
    try testing.expect(!small_free.empty());
    try testing.expect(small_free.hasItem(&seg1));
    try testing.expect(!small_free.hasItem(&seg2));

    small_free.push(&seg2);
    try testing.expect(small_free.hasItem(&seg2));

    // Pop (LIFO order)
    const popped1 = small_free.pop();
    try testing.expectEqual(&seg2, popped1);

    const popped2 = small_free.pop();
    try testing.expectEqual(&seg1, popped2);

    try testing.expect(small_free.empty());
}

test "Segment: Segment structure size" {
    const testing = std.testing;

    // Segment should be reasonably sized
    const segment_size = @sizeOf(Segment);
    try testing.expect(segment_size > 0);

    // Segment header + pages array should be smaller than a few slices
    // This allows room for data in the segment
    try testing.expect(segment_size < 4 * types.SEGMENT_SLICE_SIZE);
}

test "Segment: hasFree" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.capacity = 10;
    segment.used = 0;

    try testing.expect(segment.hasFree());

    segment.used = 5;
    try testing.expect(segment.hasFree());

    segment.used = 10;
    try testing.expect(!segment.hasFree());

    segment.used = 11; // overflow case
    try testing.expect(!segment.hasFree());
}

test "Segment: pageClaim and used counter" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.capacity = 4;
    segment.used = 0;
    segment.allow_decommit = false;

    // Initialize pages
    for (0..4) |i| {
        var pg = &segment.pages[i];
        pg.* = .{};
        pg.set_segment_idx(@intCast(i));
        pg.set_commited(true);
    }

    // Claim first page
    try testing.expect(segment.pageClaim(&segment.pages[0]));
    try testing.expectEqual(@as(usize, 1), segment.used);
    try testing.expect(segment.pages[0].is_segment_in_use());

    // Claim second page
    try testing.expect(segment.pageClaim(&segment.pages[1]));
    try testing.expectEqual(@as(usize, 2), segment.used);

    // Still has free pages
    try testing.expect(segment.hasFree());

    // Claim remaining pages
    try testing.expect(segment.pageClaim(&segment.pages[2]));
    try testing.expect(segment.pageClaim(&segment.pages[3]));
    try testing.expectEqual(@as(usize, 4), segment.used);

    // No more free pages
    try testing.expect(!segment.hasFree());
}

test "Segment: findFreePage" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.capacity = 3;
    segment.used = 0;
    segment.allow_decommit = false;

    // Initialize pages
    for (0..3) |i| {
        var pg = &segment.pages[i];
        pg.* = .{};
        pg.set_segment_idx(@intCast(i));
        pg.set_commited(true);
    }

    // Find first free page
    const pg1 = segment.findFreePage();
    try testing.expect(pg1 != null);
    try testing.expectEqual(@as(u10, 0), pg1.?.segment_idx());
    try testing.expectEqual(@as(usize, 1), segment.used);

    // Find second free page
    const pg2 = segment.findFreePage();
    try testing.expect(pg2 != null);
    try testing.expectEqual(@as(u10, 1), pg2.?.segment_idx());
    try testing.expectEqual(@as(usize, 2), segment.used);

    // Find third free page
    const pg3 = segment.findFreePage();
    try testing.expect(pg3 != null);
    try testing.expectEqual(@as(u10, 2), pg3.?.segment_idx());
    try testing.expectEqual(@as(usize, 3), segment.used);

    // No more free pages
    const pg4 = segment.findFreePage();
    try testing.expect(pg4 == null);
}

test "Segment: pageClear" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.allow_purge = false; // Don't actually purge in test

    var page: Page = .{
        .used = 10,
    };
    page.set_segment_idx(0);
    page.set_segment_in_use(true);
    page.set_commited(true);
    page.set_in_full(true);

    segment.pageClear(&page);

    try testing.expect(!page.is_segment_in_use());
    try testing.expectEqual(@as(u16, 0), page.used);
    try testing.expect(!page.is_in_full());
}

test "SegmentsTLD: trackSize" {
    const testing = std.testing;

    var tld = SegmentsTLD{};

    // Initial state
    try testing.expectEqual(@as(usize, 0), tld.count);
    try testing.expectEqual(@as(usize, 0), tld.current_size);
    try testing.expectEqual(@as(usize, 0), tld.peak_count);
    try testing.expectEqual(@as(usize, 0), tld.peak_size);

    // Add segment
    tld.trackSize(1000);
    try testing.expectEqual(@as(usize, 1), tld.count);
    try testing.expectEqual(@as(usize, 1000), tld.current_size);
    try testing.expectEqual(@as(usize, 1), tld.peak_count);
    try testing.expectEqual(@as(usize, 1000), tld.peak_size);

    // Add another segment
    tld.trackSize(2000);
    try testing.expectEqual(@as(usize, 2), tld.count);
    try testing.expectEqual(@as(usize, 3000), tld.current_size);
    try testing.expectEqual(@as(usize, 2), tld.peak_count);
    try testing.expectEqual(@as(usize, 3000), tld.peak_size);

    // Remove a segment
    tld.trackSize(-1000);
    try testing.expectEqual(@as(usize, 1), tld.count);
    try testing.expectEqual(@as(usize, 2000), tld.current_size);

    // Peak should not decrease
    try testing.expectEqual(@as(usize, 2), tld.peak_count);
    try testing.expectEqual(@as(usize, 3000), tld.peak_size);

    // Remove all segments
    tld.trackSize(-2000);
    try testing.expectEqual(@as(usize, 0), tld.count);
    try testing.expectEqual(@as(usize, 0), tld.current_size);
}

test "SegmentsTLD: freeQueueOfKind" {
    const testing = std.testing;

    var tld = SegmentsTLD{};

    // Small, medium, and large have queues
    try testing.expect(tld.freeQueueOfKind(.small) != null);
    try testing.expect(tld.freeQueueOfKind(.medium) != null);
    try testing.expect(tld.freeQueueOfKind(.large) != null);

    // Huge don't use queues (unique sized)
    try testing.expect(tld.freeQueueOfKind(.huge) == null);

    // Verify they return correct queue pointers
    try testing.expectEqual(&tld.small_free, tld.freeQueueOfKind(.small).?);
    try testing.expectEqual(&tld.medium_free, tld.freeQueueOfKind(.medium).?);
    try testing.expectEqual(&tld.large_free, tld.freeQueueOfKind(.large).?);
}

test "SegmentsTLD: queue management" {
    const testing = std.testing;

    var tld = SegmentsTLD{};

    var seg1: Segment = .{ .page_kind = .small };
    var seg2: Segment = .{ .page_kind = .small };
    var seg3: Segment = .{ .page_kind = .medium };

    // Insert segments
    tld.insertInFreeQueue(&seg1);
    tld.insertInFreeQueue(&seg2);
    tld.insertInFreeQueue(&seg3);

    try testing.expect(tld.small_free.hasItem(&seg1));
    try testing.expect(tld.small_free.hasItem(&seg2));
    try testing.expect(tld.medium_free.hasItem(&seg3));

    // Remove segment
    tld.removeFromFreeQueue(&seg1);
    try testing.expect(!tld.small_free.hasItem(&seg1));
    try testing.expect(tld.small_free.hasItem(&seg2));

    // Remove non-existent segment (should not crash)
    tld.removeFromFreeQueue(&seg1);
}

test "SegmentsTLD: freeQueue by segment" {
    const testing = std.testing;

    var tld = SegmentsTLD{};

    var small_seg: Segment = .{ .page_kind = .small };
    var medium_seg: Segment = .{ .page_kind = .medium };
    var large_seg: Segment = .{ .page_kind = .large };

    try testing.expectEqual(&tld.small_free, tld.freeQueue(&small_seg).?);
    try testing.expectEqual(&tld.medium_free, tld.freeQueue(&medium_seg).?);
    try testing.expectEqual(&tld.large_free, tld.freeQueue(&large_seg).?);
}

test "Segment: calculateSizes alignment" {
    const testing = std.testing;

    // Test info_size is properly aligned
    const sizes = Segment.calculateSizes(512, 0);
    try testing.expect(sizes.info_size % (16 * types.MAX_ALIGN_SIZE) == 0);

    // Huge segment size should be aligned to 256 KiB
    const huge_sizes = Segment.calculateSizes(1, 100 * types.MiB);
    try testing.expect(huge_sizes.segment_size % (256 * types.KiB) == 0);
    try testing.expect(huge_sizes.segment_size >= 100 * types.MiB);
}

test "Segment: capacityForKind values" {
    const testing = std.testing;

    // Small pages: total slices minus header slices
    const small_cap = Segment.capacityForKind(.small, types.SMALL_PAGE_SHIFT);
    // Header takes some slices, so capacity is less than total
    try testing.expect(small_cap > 0);
    try testing.expect(small_cap < types.SLICES_PER_SEGMENT);

    // Medium pages: also reduced by header
    const medium_cap = Segment.capacityForKind(.medium, types.MEDIUM_PAGE_SHIFT);
    try testing.expect(medium_cap > 0);
    try testing.expect(medium_cap < types.SLICES_PER_SEGMENT / 8);

    // Large and huge always have capacity 1
    try testing.expectEqual(@as(usize, 1), Segment.capacityForKind(.large, types.SEGMENT_SHIFT));
    try testing.expectEqual(@as(usize, 1), Segment.capacityForKind(.huge, types.SEGMENT_SHIFT));
}

test "SegmentList: basic operations" {
    const testing = std.testing;

    var list = SegmentList.init();

    try testing.expect(list.empty());
    try testing.expectEqual(@as(u64, 0), list.count());

    var seg1: Segment = .{};
    var seg2: Segment = .{};

    list.push(&seg1);
    try testing.expect(!list.empty());
    try testing.expectEqual(@as(u64, 1), list.count());

    list.push(&seg2);
    try testing.expectEqual(@as(u64, 2), list.count());

    const p1 = list.pop();
    try testing.expectEqual(&seg2, p1.?);
    try testing.expectEqual(@as(u64, 1), list.count());

    const p2 = list.pop();
    try testing.expectEqual(&seg1, p2.?);
    try testing.expect(list.empty());
}

test "Segment: Page structure size" {
    const testing = std.testing;

    const page_size = @sizeOf(Page);
    try testing.expect(page_size > 0);

    // Page should be relatively small
    try testing.expect(page_size < 256);
}

test "Segment: thread_id alignment" {
    const testing = std.testing;

    // thread_id should be cache line aligned
    const field_offset = @offsetOf(Segment, "thread_id");
    try testing.expect(field_offset % std.atomic.cache_line == 0);
}

test "Segment: abandonment" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.thread_id.store(12345, .release);

    try testing.expect(!segment.isAbandoned());

    segment.markAbandoned();
    try testing.expect(segment.isAbandoned());
    try testing.expectEqual(@as(usize, 0), segment.abandoned_visits);
}

test "Segment: tryReclaim" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.thread_id.store(0, .release); // abandoned

    const thread_id = std.Thread.getCurrentId();

    // Should succeed on abandoned segment
    try testing.expect(segment.tryReclaim(thread_id));
    try testing.expectEqual(thread_id, segment.thread_id.load(.acquire));

    // Second attempt should fail (not abandoned anymore)
    try testing.expect(!segment.tryReclaim(thread_id + 1));
}

test "Segment: canReclaim" {
    const testing = std.testing;

    var segment: Segment = .{};
    segment.capacity = 4;
    segment.used = 2;
    segment.thread_id.store(0, .release); // abandoned

    // Has free and is abandoned
    try testing.expect(segment.canReclaim());

    // Not abandoned
    segment.thread_id.store(12345, .release);
    try testing.expect(!segment.canReclaim());

    // Abandoned but no free pages
    segment.thread_id.store(0, .release);
    segment.used = 4;
    try testing.expect(!segment.canReclaim());
}

test "Segment: page flags" {
    const testing = std.testing;

    var page: Page = .{};

    // Test flag accessors
    try testing.expect(!page.is_in_full());
    page.set_in_full(true);
    try testing.expect(page.is_in_full());
    page.set_in_full(false);
    try testing.expect(!page.is_in_full());

    // Test segment_idx
    try testing.expectEqual(@as(u10, 0), page.segment_idx());
    page.set_segment_idx(42);
    try testing.expectEqual(@as(u10, 42), page.segment_idx());

    // Test flags don't interfere with segment_idx
    page.set_in_full(true);
    page.set_in_bin(true);
    try testing.expectEqual(@as(u10, 42), page.segment_idx());
    try testing.expect(page.is_in_full());
    try testing.expect(page.is_in_bin());
}

test "Segment: isValid" {
    const testing = std.testing;

    var segment: Segment = .{};

    // Invalid: zero capacity
    try testing.expect(!segment.isValid());

    segment.capacity = 3;
    segment.used = 0;
    segment.segment_size = types.SEGMENT_SIZE;

    // Initialize pages
    for (0..3) |i| {
        var pg = &segment.pages[i];
        pg.* = .{};
        pg.set_segment_idx(@intCast(i));
        pg.set_segment_in_use(false);
    }

    // Valid: used count matches
    try testing.expect(segment.isValid());

    // Mark one page in use
    segment.pages[0].set_segment_in_use(true);
    segment.used = 1;
    try testing.expect(segment.isValid());

    // Invalid: mismatch
    segment.used = 2;
    try testing.expect(!segment.isValid());
}

// test "SegmentsTLD: getStats" {
//     const testing = std.testing;
//
//     var tld = SegmentsTLD{};
//
//     tld.trackSize(1000);
//     tld.trackSize(2000);
//
//     const stats = tld.getStats();
//     try testing.expectEqual(@as(usize, 2), stats.count);
//     try testing.expectEqual(@as(usize, 3000), stats.current_size);
//     try testing.expectEqual(@as(usize, 2), stats.peak_count);
//     try testing.expectEqual(@as(usize, 3000), stats.peak_size);
// }

test "SegmentsTLD: isWithinTarget" {
    const testing = std.testing;

    var tld = SegmentsTLD{};
    tld.count = 5;

    try testing.expect(tld.isWithinTarget(5));
    try testing.expect(tld.isWithinTarget(10));
    try testing.expect(!tld.isWithinTarget(4));
}
