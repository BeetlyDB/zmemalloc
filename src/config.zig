const std = @import("std");
const assert = @import("util.zig").assert;
const builtin = @import("builtin");

pub const Config = struct {
    thread_safe: bool = !builtin.single_threaded,
    huge_allocations: bool = false,
};
