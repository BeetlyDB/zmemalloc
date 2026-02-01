const std = @import("std");
const assert = @import("util.zig").assert;
const MemID = @import("mem.zig").MemID;
const Atomic = std.atomic.Value;
const builtin = @import("builtin");
const Mutex = @import("mutex.zig").Mutex;
const segment = @import("segment.zig");
const Segment = segment.Segment;
const SegmentAbandonedQueue = segment.SegmentAbandonedQueue;

// ------------------------------------------------------
// Sub processes do not reclaim or visit segments
// from other sub processes. These are essentially the
// static variables of a process.
// ------------------------------------------------------
pub const Subproc = struct {
    memid: MemID = .{},
    abandoned_count: Atomic(usize) = .init(0), // count of abandoned segments for this sub-process
    abandoned_os_list_count: Atomic(usize) = .init(0),
    // lock for the abandoned os segment list (outside of arena's) (this lock protects list operations)
    abandoned_os_lock: Mutex = .{},
    // ensure only one thread per subproc visits the abandoned os list
    abandoned_os_visit_lock: Mutex = .{},
    // list of abandoned OS segments
    abandoned_os_list: SegmentAbandonedQueue = .{},
};
