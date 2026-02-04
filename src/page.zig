const std = @import("std");
const list = @import("queue.zig");
const types = @import("types.zig");
const Atomic = std.atomic.Value;
const builtin = @import("builtin");
const bounded_array = @import("bounded_array.zig");
const assert = @import("util.zig").assert;
const heap = @import("heap.zig");

/// Free block - overlays on free memory, uses intrusive linked list
//------------------------------------------------------------------------------------
//
// Allocated memory (64 bytes):
// ┌────────────────────────────────────────────────────────────────┐
// │                          Data                                  │
// └────────────────────────────────────────────────────────────────┘
//
// Free memory (same 64 байта):
// ┌──────────┬─────────────────────────────────────────────────────┐
// │  *Block  │              not used                               │
// │  .link   │                                                     │
// └──────────┴─────────────────────────────────────────────────────┘
//      ↓
//    next free block
//
// How used:
//
// const Block = struct {
//     link: queue.QueueType(Block).Link = .{},
// };
//
// // free  - cast to block
// pub fn free(ptr: [*]u8) void {
//     const block: *Block = @ptrCast(@alignCast(ptr));
//
//     // add in free list
//     page.free.push(block);
// }
//
// // Allocate memory - take block from free list
// pub fn alloc() ?[*]u8 {
//     const block = page.free.pop() orelse return null;
//
//
//     return @ptrCast(block);
// }
//
// I:
// - *Block and [*]u8 point to same address
// - When memory freed — first 8  bytes its link.next
// - When memory allocated — user uses all memory
//
//

//------------------------------------------------------------------------------------

pub const Block = struct {
    link: list.IntrusiveLifo(Block).Link = .{},
};

pub const Page = struct {
    const Self = @This();

    // Segment info
    segment_idx: u32 = 0, // index within segment
    slice_count: u32 = 0, // number of slices this page uses
    slice_offset: u32 = 0, // offset from segment start in slices

    // Page metadata
    capacity: u16 = 0, // max number of blocks
    reserved: u16 = 0, // reserved blocks
    used: u16 = 0, // blocks in use

    flags: Flags = .{},
    retire_expire: u8 = 0,
    block_size_shift: u8 = 0,
    heap_tag: u8 = 0,

    // Free lists
    free: list.IntrusiveLifo(Block) = list.IntrusiveLifo(Block).init(), // free blocks in this page
    local_free: list.IntrusiveLifo(Block) = list.IntrusiveLifo(Block).init(), // thread-local free list

    block_size: usize = 0,
    page_start: ?[*]u8 = null,

    xthread_free: Atomic(?*Block) = .init(null), // cross-thread free list (atomic)

    // intrusive linked list for page queues
    next: ?*Page = null, // next page owned by this thread with the same `block_size`
    prev: ?*Page = null,

    // Segment usage flag
    segment_in_use: bool = false,

    // Purge scheduling
    expire: i64 = 0, // expiration time for delayed purge

    pub const Flags = packed struct(u8) {
        free_is_zero: bool = false,
        is_commited: bool = false,
        is_zero_init: bool = false,
        is_huge: bool = false,
        page_flags: PageFlags = .{},
    };

    pub const Queue = list.DoublyLinkedListType(Page, .next, .prev); //lifo

    pub const PageKind = union(enum) {
        PAGE_SMALL, // small blocks go into 64KiB pages inside a segment
        PAGE_MEDIUM, // medium blocks go into 512KiB pages inside a segment
        PAGE_LARGE, // larger blocks go into a single page spanning a whole segment
        PAGE_HUGE, // a huge page is a single page in a segment of variable size
        // used for blocks `> LARGE_OBJ_SIZE_MAX` or an aligment `> BLOCK_ALIGNMENT_MAX`.
    };

    pub const PageFlags = packed struct(u4) {
        in_full: bool = false,
        has_aligned: bool = false,
        in_purge_queue: bool = false,
        in_bin: bool = false,
    };

    pub inline fn is_in_full(self: *const Self) bool {
        return self.flags.page_flags.in_full;
    }

    pub inline fn pagePushBin(self: *Self, hp: *heap.Heap, bin: usize) void {
        if (self.flags.page_flags.in_bin) {
            hp.pages[bin].remove(self);
        }

        hp.pages[bin].push(self);
        self.flags.page_flags.in_bin = true;
    }

    pub inline fn pageRemoveFromBin(self: *Self, hp: *heap.Heap, bin: usize) void {
        if (self.flags.page_flags.in_bin) {
            hp.pages[bin].remove(self);
            self.flags.page_flags.in_bin = false;
        }
    }

    pub inline fn set_in_full(self: *Self, in_full: bool) void {
        self.flags.page_flags.in_full = in_full;
    }

    pub inline fn is_aligned(self: *const Self) bool {
        return self.flags.page_flags.has_aligned;
    }

    pub inline fn set_aligned(self: *Self, has_aligned: bool) void {
        self.flags.page_flags.has_aligned = has_aligned;
    }

    /// Initialize page for use with given block size
    pub fn init(self: *Self, block_size: usize, page_start: [*]u8, page_size: usize) void {
        self.block_size = block_size;
        self.page_start = page_start;
        self.used = 0;
        self.free = list.IntrusiveLifo(Block).init();
        self.local_free = list.IntrusiveLifo(Block).init();
        self.xthread_free.store(null, .release);

        // Calculate capacity
        if (block_size > 0 and page_size >= block_size) {
            self.capacity = @intCast(page_size / block_size);
            self.reserved = 0; // Start with 0 reserved, will be extended as needed
        } else {
            self.capacity = 0;
            self.reserved = 0;
        }

        // Calculate shift for power-of-2 sizes
        if (block_size > 0 and (block_size & (block_size - 1)) == 0) {
            self.block_size_shift = @intCast(@ctz(block_size));
        } else {
            self.block_size_shift = 0;
        }

        self.flags.free_is_zero = self.flags.is_zero_init;
        self.retire_expire = 0;
    }

    // =========================================================================
    // Free List Management
    // =========================================================================

    /// Get block at index within page
    pub inline fn blockAt(self: *const Self, idx: usize) ?*Block {
        if (idx >= self.capacity) return null;
        const start = self.page_start orelse return null;
        return @ptrCast(@alignCast(start + idx * self.block_size));
    }

    /// Extend free list by adding more blocks
    pub fn extendFree(self: *Self, extend_count: usize) void {
        if (extend_count == 0) return;
        const start = self.page_start orelse return;
        const bsize = self.block_size;
        if (bsize == 0) return;

        const reserved = self.reserved;
        const capacity = self.capacity;
        if (reserved >= capacity) return;

        const actual_extend = @min(extend_count, capacity - reserved);
        const block_start = start + reserved * bsize;

        // Add blocks to free list in reverse order for better locality
        var i: usize = actual_extend;
        while (i > 0) : (i -= 1) {
            const block: *Block = @ptrCast(@alignCast(block_start + (i - 1) * bsize));
            // Initialize link before pushing (memory may contain garbage)
            // block.link = .{};
            self.free.push(block);
        }

        self.reserved += @intCast(actual_extend);
    }

    /// Check if page has any free blocks
    pub inline fn hasFree(self: *const Self) bool {
        return !self.free.empty() or !self.local_free.empty() or self.xthread_free.load(.acquire) != null;
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

    /// Collect thread-local free list into main free list
    pub inline fn collectLocalFree(self: *Self) void {
        // Move all from local_free to free
        while (self.local_free.pop()) |block| {
            self.free.push(block);
        }
    }

    /// Collect cross-thread free list (atomic)
    pub inline fn collectXthreadFree(self: *Self) usize {
        var head = self.xthread_free.swap(null, .acq_rel);
        var count: usize = 0;

        while (head) |block| {
            // Save next before clearing link (push asserts link.next == null)
            const next_link = block.link.next;
            const next: ?*Block = if (next_link) |link|
                @alignCast(@fieldParentPtr("link", link))
            else
                null;

            // Clear link before pushing (required by IntrusiveLifo.push assertion)
            block.link.next = null;
            self.free.push(block);
            head = next;
            count += 1;
        }

        if (count > 0) {
            self.used -|= @intCast(@min(count, self.used));
        }

        return count;
    }

    /// Collect all free lists
    pub inline fn freeCollect(self: *Self, force: bool) void {
        _ = force;
        self.collectLocalFree();
        _ = self.collectXthreadFree();
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
                // Success
                break;
            }
        }
    }

    /// Pop a free block - hot path
    pub inline fn popFreeBlock(self: *Self) ?*Block {
        // Hot path: pop from free list
        if (self.free.pop()) |block| {
            self.used += 1;
            return block;
        }

        // Cold path: collect local_free into free, then retry
        return self.popFreeBlockSlow();
    }

    inline fn popFreeBlockSlow(self: *Self) ?*Block {
        // Move local_free to free
        if (!self.local_free.empty()) {
            // Swap the lists - local_free becomes new free
            const temp = self.free;
            self.free = self.local_free;
            self.local_free = temp;

            if (self.free.pop()) |block| {
                self.used += 1;
                return block;
            }
        }

        // Try to collect xthread free
        if (self.collectXthreadFree() > 0) {
            if (self.free.pop()) |block| {
                self.used += 1;
                return block;
            }
        }
        return null;
    }

    /// Push freed block to local_free (write path separate from read path)
    pub inline fn pushFreeBlock(self: *Self, block: *Block) void {
        self.local_free.push(block);
        self.used -|= 1;
    }

    /// Reset page to initial state
    pub fn reset(self: *Self) void {
        self.used = 0;
        self.free = list.IntrusiveLifo(Block).init();
        self.local_free = list.IntrusiveLifo(Block).init();
        self.xthread_free.store(null, .release);
        self.reserved = 0;
        self.retire_expire = 0;
        self.flags.page_flags = .{};
    }

    /// Check if page can be retired (mostly empty and not in full queue)
    pub inline fn canRetire(self: *const Self) bool {
        return self.isMostlyEmpty() and !self.is_in_full();
    }

    /// Get number of free blocks
    pub fn freeCount(self: *const Self) usize {
        // Use the count() method which tracks the count
        return self.free.count() + self.local_free.count();
        // Note: xthread_free not counted as it requires atomic access
    }

    /// Get utilization ratio (0.0 to 1.0)
    pub fn utilization(self: *const Self) f32 {
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

/// Get small page size for a given block size
pub fn smallPageSizeFor(block_size: usize) usize {
    _ = block_size;
    return types.SMALL_PAGE_SIZE;
}

/// Get medium page size for a given block size
pub fn mediumPageSizeFor(block_size: usize) usize {
    _ = block_size;
    return types.MEDIUM_PAGE_SIZE;
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

/// Move page to end of queue (most recently accessed - for cache efficiency)
pub inline fn queueMoveToEnd(queue: *Page.Queue, page: *Page) void {
    if (queue.tail == page) return;

    // Remove from current position
    queue.remove(page);

    // Add to end (tail)
    queue.push(page);
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
    var buffer: [1024]u8 align(16) = undefined;

    page.init(64, &buffer, 1024);

    try testing.expectEqual(@as(usize, 64), page.block_size);
    try testing.expectEqual(@as(u16, 16), page.capacity); // 1024 / 64
    try testing.expectEqual(@as(u16, 0), page.reserved); // starts at 0, extended lazily
    try testing.expectEqual(@as(u16, 0), page.used);
}

test "Page: power of 2 block size shift" {
    var page: Page = .{};
    var buffer: [256]u8 align(16) = undefined;

    page.init(16, &buffer, 256);
    try testing.expectEqual(@as(u8, 4), page.block_size_shift); // log2(16) = 4

    page.init(64, &buffer, 256);
    try testing.expectEqual(@as(u8, 6), page.block_size_shift); // log2(64) = 6

    // Non-power-of-2
    page.init(48, &buffer, 256);
    try testing.expectEqual(@as(u8, 0), page.block_size_shift);
}

test "Page: extendFree" {
    var page: Page = .{};
    var buffer: [512]u8 align(16) = undefined;

    page.init(64, &buffer, 512);
    page.reserved = 0; // Reset reserved to test extension

    try testing.expect(page.free.empty());

    page.extendFree(4);

    try testing.expectEqual(@as(u16, 4), page.reserved);
    try testing.expect(!page.free.empty());

    // Count free blocks using count() method
    try testing.expectEqual(@as(u64, 4), page.free.count());
}

test "Page: popFreeBlock and pushFreeBlock" {
    var page: Page = .{};
    var buffer: [256]u8 align(16) = undefined;

    page.init(32, &buffer, 256);
    page.reserved = 0;
    page.extendFree(4);

    // Pop blocks
    const b1 = page.popFreeBlock();
    try testing.expect(b1 != null);
    try testing.expectEqual(@as(u16, 1), page.used);

    const b2 = page.popFreeBlock();
    try testing.expect(b2 != null);
    try testing.expectEqual(@as(u16, 2), page.used);

    // Push back
    page.pushFreeBlock(b1.?);
    try testing.expectEqual(@as(u16, 1), page.used);

    // Pop again
    const b3 = page.popFreeBlock();
    try testing.expect(b3 != null);
    try testing.expectEqual(@as(u16, 2), page.used);
}

test "Page: hasFree and allUsed" {
    var page: Page = .{};
    var buffer: [128]u8 align(16) = undefined;

    page.init(32, &buffer, 128);
    page.reserved = 0;

    try testing.expect(!page.hasFree()); // No free blocks yet
    page.extendFree(4);
    try testing.expect(page.hasFree());

    // Use all blocks
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

    var p1: Page = .{ .capacity = 10, .used = 10 }; // Full
    var p2: Page = .{ .capacity = 10, .used = 5 }; // Partially used
    p2.init(8, &buffer, 64);
    p2.reserved = 0;
    p2.extendFree(2);

    queue.push(&p1);
    queue.push(&p2);

    // p2 is pushed last so it's at tail, we iterate from tail via prev
    // p2 has free blocks so it should be found
    const found = queueFindFree(&queue);
    try testing.expectEqual(&p2, found);
}
