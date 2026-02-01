const std = @import("std");
const assert = @import("util.zig").assert;
const types = @import("types.zig");
const page = @import("page.zig");
const queue = @import("queue.zig");
const tld = @import("tld.zig");
const Atomic = std.atomic.Value;
const Mutex = @import("mutex.zig").Mutex;

pub const Heap = struct {
    _: void align(std.atomic.cache_line) = {},
    tld: ?*tld.TLD = null,
    thread_delayed_free: Atomic(?*page.Block) = .init(null), // atomic cross-thread delayed free list
    thread_id: std.Thread.Id = 0,
    arena_id: usize = 0,
    page_count: usize = 0,
    page_retired_min: usize = 0,
    page_retired_max: usize = 0,

    generic_count: i64 = 0,
    generic_collect_count: i64 = 0,
    // intrusive list of heaps per thread
    link: queue.IntrusiveLifo(Heap).Link = .{},
    no_reclaim: bool = false,
    tag: u8 = 0,

    pages_free_direct: [types.PAGES_DIRECT]?*page.Page = [_]?*page.Page{null} ** types.PAGES_DIRECT,

    // intrusive linked list for heap in thread
    prev: ?*Heap = null,
    next: ?*Heap = null,

    pages: [types.BIN_FULL + 1]page.Page.Queue = [_]page.Page.Queue{.{}} ** (types.BIN_FULL + 1),

    pub const Queue = queue.DoublyLinkedListType(Heap, .next, .prev);
    pub const List = queue.IntrusiveLifo(Heap);

    pub inline fn isBacking(self: *const Heap) bool {
        const t = self.tld orelse return false;
        return t.heap_backing == self;
    }

    pub fn isInitialized(self: *const Heap) bool {
        return self.tld != null;
    }

    // pub fn ptrCookie(p: ?*const anyopaque) usize {
    //     if (p == null) return 0;
    //     return @intFromPtr(p) ^ heap_main.cookie;
    // }
};

pub const heap_empty: Heap = .{};
pub threadlocal var heap_main: Heap = heap_empty;
pub threadlocal var heap_default: *const Heap = (&heap_empty);
pub threadlocal var heap_backing: *Heap = &heap_main;

pub inline fn init_main_heap() void {
    heap_main.thread_id = std.Thread.getCurrentId();
}

pub fn main_get_heap() *Heap {
    std.once(init_main_heap());
    return &heap_main;
}

pub fn is_heap_main_thread() bool {
    return heap_main.thread_id == std.Thread.getCurrentId() and heap_main.thread_id != 0;
}
