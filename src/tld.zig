const std = @import("std");
const assert = @import("util.zig").assert;
const Atomic = std.atomic.Value;
const builtin = @import("builtin");
const page = @import("page.zig");
const types = @import("types.zig");
const queue = @import("queue.zig");
const Subproc = @import("subproc.zig").Subproc;
const Stats = @import("stats.zig").Stats;
const heap = @import("heap.zig");
const os = @import("os.zig");
const os_alloc = @import("os_allocator.zig");

// SegmentsTLD is defined in segment.zig to avoid circular imports
pub const SegmentsTLD = @import("segment.zig").SegmentsTLD;

// Thread local data
pub const TLD = struct {
    heartbeat: u64 = 0,
    recurse: bool = false,

    heap_backing: ?*heap.Heap = null,

    os_allocator: os_alloc.OsAllocator = .{ .config = os.mem_config_static },
    segments: SegmentsTLD = .{},
    stats: Stats = .{},
};
