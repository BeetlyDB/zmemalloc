const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const Allocator = mem.Allocator;
const maxInt = std.math.maxInt;
const native_os = builtin.os.tag;
const windows = std.os.windows;
const ntdll = windows.ntdll;
const posix = std.posix;
const os = @import("os.zig");
const page_size_min = os.mem_config_static.page_size;
const utils = @import("util.zig");
const assert = utils.assert;

pub const OsAllocator = @This();

config: os.OsMemConfig,

pub fn allocator(self: *OsAllocator) Allocator {
    return .{
        .ptr = self,
        .vtable = &.{
            .alloc = alloc,
            .resize = resize,
            .remap = remap,
            .free = free,
        },
    };
}

pub inline fn map(self: *const OsAllocator, n: usize, alignment: mem.Alignment) ?[*]u8 {
    const page_size = self.config.page_size;
    if (n >= maxInt(usize) - page_size) return null;
    const alignment_bytes = alignment.toByteUnits();

    const aligned_len = utils.alignForward(usize, n, page_size);
    const max_drop_len = alignment_bytes - @min(alignment_bytes, page_size);
    const overalloc_len = if (max_drop_len <= aligned_len - n)
        aligned_len
    else
        utils.alignForward(usize, aligned_len + max_drop_len, page_size);
    const hint = @atomicLoad(@TypeOf(std.heap.next_mmap_addr_hint), &std.heap.next_mmap_addr_hint, .unordered);

    const slice = posix.mmap(
        hint,
        overalloc_len,
        posix.PROT.READ | posix.PROT.WRITE,
        if (self.config.has_overcommit) .{
            .TYPE = .PRIVATE,
            .ANONYMOUS = true,
            .NORESERVE = true,
        } else .{
            .TYPE = .PRIVATE,
            .ANONYMOUS = true,
        },
        -1,
        0,
    ) catch return null;
    const result_ptr = utils.alignPointer(slice.ptr, alignment_bytes) orelse return null;
    if (self.config.thp == .thp_mode_always) {
        os.maybe_enable_thp(result_ptr, aligned_len, alignment_bytes, self.config.thp);
    }
    // Unmap the extra bytes that were only requested in order to guarantee
    // that the range of memory we were provided had a proper alignment in it
    // somewhere. The extra bytes could be at the beginning, or end, or both.
    const drop_len = result_ptr - slice.ptr;
    if (drop_len != 0) posix.munmap(slice[0..drop_len]);
    const remaining_len = overalloc_len - drop_len;
    if (remaining_len > aligned_len) posix.munmap(@alignCast(result_ptr[aligned_len..remaining_len]));
    const new_hint: [*]align(page_size_min) u8 = @alignCast(result_ptr + aligned_len);
    _ = @cmpxchgStrong(@TypeOf(std.heap.next_mmap_addr_hint), &std.heap.next_mmap_addr_hint, hint, new_hint, .monotonic, .monotonic);
    return result_ptr;
}

fn alloc(context: *anyopaque, n: usize, alignment: mem.Alignment, ra: usize) ?[*]u8 {
    _ = ra;
    assert(n > 0);

    var self: *OsAllocator = @ptrCast(@alignCast(context));

    return self.map(n, alignment);
}

fn resize(context: *anyopaque, memory: []u8, alignment: mem.Alignment, new_len: usize, return_address: usize) bool {
    _ = alignment;
    _ = return_address;
    const self: *OsAllocator = @ptrCast(@alignCast(context));
    return self.realloc(memory, new_len, false) != null;
}

fn remap(context: *anyopaque, memory: []u8, alignment: mem.Alignment, new_len: usize, return_address: usize) ?[*]u8 {
    _ = alignment;
    _ = return_address;
    const self: *OsAllocator = @ptrCast(@alignCast(context));
    return self.realloc(memory, new_len, true);
}

fn free(context: *anyopaque, memory: []u8, alignment: mem.Alignment, return_address: usize) void {
    _ = alignment;
    _ = return_address;
    var self: *OsAllocator = @ptrCast(@alignCast(context));
    return self.unmap(@alignCast(memory));
}

pub fn unmap(self: *const OsAllocator, memory: []align(page_size_min) u8) void {
    const page_aligned_len = utils.alignForward(usize, memory.len, self.config.page_size);
    posix.munmap(memory.ptr[0..page_aligned_len]);
}

pub fn realloc(self: *const OsAllocator, uncasted_memory: []u8, new_len: usize, may_move: bool) ?[*]u8 {
    const memory: []align(page_size_min) u8 = @alignCast(uncasted_memory);
    const page_size = self.config.page_size;
    const new_size_aligned = utils.alignForward(usize, new_len, page_size);

    const page_aligned_len = utils.alignForward(usize, memory.len, page_size);
    if (new_size_aligned == page_aligned_len)
        return memory.ptr;

    if (posix.MREMAP != void) {
        // TODO: if the next_mmap_addr_hint is within the remapped range, update it
        const new_memory = posix.mremap(memory.ptr, memory.len, new_len, .{ .MAYMOVE = may_move }, null) catch return null;
        return new_memory.ptr;
    }

    if (new_size_aligned < page_aligned_len) {
        const ptr = memory.ptr + new_size_aligned;
        // TODO: if the next_mmap_addr_hint is within the unmapped range, update it
        posix.munmap(@alignCast(ptr[0 .. page_aligned_len - new_size_aligned]));
        return memory.ptr;
    }

    return null;
}
