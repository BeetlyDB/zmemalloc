const heap = @import("heap.zig");
const os = @import("os.zig");
const os_alloc = @import("os_allocator.zig");

pub const SegmentsTLD = @import("segment.zig").SegmentsTLD;

/// Thread-local data container
pub const TLD = struct {
    heap_backing: ?*heap.Heap = null,
    os_allocator: os_alloc.OsAllocator = .{ .config = os.mem_config_static },
    segments: SegmentsTLD = .{},
};
