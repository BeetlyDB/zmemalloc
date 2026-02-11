const types = @import("types.zig");
const page = @import("page.zig");
const tld = @import("tld.zig");
const std = @import("std");

/// Thread-local heap with bin queues for different size classes
pub const Heap = struct {
    tld: ?*tld.TLD = null,
    thread_id: usize = 0, // TLD address for fast thread check

    /// Direct page lookup for small sizes (wsize 0-128)
    /// Enables O(1) allocation for sizes <= 1KB
    pages_free_direct: [types.PAGES_DIRECT]?*page.Page = [_]?*page.Page{null} ** types.PAGES_DIRECT,

    /// Bin queues for all size classes
    /// Each bin holds pages with blocks of a specific size range
    pages: [types.BIN_FULL + 1]page.Page.Queue = [_]page.Page.Queue{.{}} ** (types.BIN_FULL + 1),
};

pub const heap_empty: Heap = .{};
pub threadlocal var heap_main: Heap = heap_empty;
