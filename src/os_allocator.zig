//! # OS Memory Allocator
//!
//! Zig `std.mem.Allocator` interface wrapping OS memory primitives.
//! Used by the segment layer to obtain raw memory from the kernel.
//!
//! ## Memory Management
//!
//! - `map()`: Request memory from OS via mmap with alignment
//! - `unmap()`: Return memory to OS via munmap
//! - `realloc()`: Resize via mremap (Linux) or shrink via partial munmap
//!
//! ## Alignment Handling
//!
//! When alignment exceeds page size, we over-allocate and then
//! unmap the excess regions at the beginning and end:
//!
//! ```
//! Requested: [----aligned memory----]
//! Allocated: [excess][----aligned memory----][excess]
//!            ^unmap^                         ^unmap^
//! ```
//!
//! ## Address Hint Management
//!
//! Maintains `std.heap.next_mmap_addr_hint` for address locality.
//! Sequential allocations tend to be placed near each other,
//! improving cache behavior and reducing TLB pressure.
//!
//! ## Platform Support
//!
//! - Linux: Uses mremap for efficient resizing
//! - Other POSIX: Falls back to partial munmap for shrinking

const std = @import("std");
const builtin = @import("builtin");
const mem = std.mem;
const Allocator = mem.Allocator;
const maxInt = std.math.maxInt;
const native_os = builtin.os.tag;
const posix = std.posix;
const os = @import("os.zig");
const page_size_min = os.mem_config_static.page_size;
const utils = @import("util.zig");
const assert = utils.assert;

/// OS-backed memory allocator
///
/// Implements `std.mem.Allocator` using mmap/munmap/mremap.
/// Each thread-local TLD has its own OsAllocator instance.
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
    if (n >= maxInt(usize) - page_size) {
        @branchHint(.cold);
        return null;
    }
    const alignment_bytes = alignment.toByteUnits();

    const aligned_len = utils.alignForward(usize, n, page_size);

    const max_drop_len = alignment_bytes -| page_size;
    const overalloc_len = aligned_len + max_drop_len;
    const maybe_unaligned_hint = @atomicLoad(@TypeOf(std.heap.next_mmap_addr_hint), &std.heap.next_mmap_addr_hint, .monotonic);
    const hint: ?[*]align(page_size_min) u8 = @ptrFromInt(((@intFromPtr(maybe_unaligned_hint)) +% (alignment_bytes - 1)) & ~(alignment_bytes - 1));

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
    ) catch {
        @branchHint(.cold);
        return null;
    };
    const result_ptr = utils.alignPointer(slice.ptr, alignment_bytes) orelse {
        @branchHint(.cold);
        return null;
    };

    //does not enable for always hugepages,use only when madvise specified
    if (self.config.thp == .thp_mode_always) {
        os.maybe_enable_thp(result_ptr, aligned_len, alignment_bytes, self.config.thp);
    }
    // if (self.config.thp == .thp_mode_always) {
    //     @branchHint(.likely);
    //     os.maybe_enable_thp(result_ptr, aligned_len, alignment_bytes, self.config.thp);
    // }
    // Unmap the extra bytes that were only requested in order to guarantee
    // that the range of memory we were provided had a proper alignment in it
    // somewhere. The extra bytes could be at the beginning, or end, or both.
    const drop_len = result_ptr - slice.ptr;
    if (drop_len != 0) posix.munmap(slice[0..drop_len]);
    const remaining_len = overalloc_len - drop_len;
    if (remaining_len > aligned_len) posix.munmap(@alignCast(result_ptr[aligned_len..remaining_len]));
    const new_hint: [*]align(page_size_min) u8 = @alignCast(result_ptr + aligned_len);
    _ = @cmpxchgStrong(@TypeOf(std.heap.next_mmap_addr_hint), &std.heap.next_mmap_addr_hint, maybe_unaligned_hint, new_hint, .monotonic, .monotonic);
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

pub inline fn realloc(self: *const OsAllocator, uncasted_memory: []u8, new_len: usize, may_move: bool) ?[*]u8 {
    const memory: []align(page_size_min) u8 = @alignCast(uncasted_memory);
    const page_size = self.config.page_size;

    const old_page_len = utils.alignForward(usize, memory.len, page_size);
    const new_page_len = utils.alignForward(usize, new_len, page_size);

    if (new_page_len == old_page_len) {
        @branchHint(.likely);
        return memory.ptr;
    }

    if (posix.MREMAP != void) {
        const old_hint = @atomicLoad(@TypeOf(std.heap.next_mmap_addr_hint), &std.heap.next_mmap_addr_hint, .monotonic);

        const new_memory = posix.mremap(memory.ptr, memory.len, new_len, .{ .MAYMOVE = may_move }, null) catch return null;
        if (new_memory.ptr != memory.ptr) {
            const new_end = new_memory.ptr + new_page_len;
            if (old_hint) |hint_ptr| {
                const hint_addr = @intFromPtr(hint_ptr);
                const new_start_addr = @intFromPtr(new_memory.ptr);
                const new_end_addr = @intFromPtr(new_end);

                if (hint_addr >= new_start_addr and hint_addr < new_end_addr) {
                    const updated_hint: [*]align(page_size_min) u8 = @alignCast(new_end);
                    _ = @cmpxchgStrong(
                        @TypeOf(std.heap.next_mmap_addr_hint),
                        &std.heap.next_mmap_addr_hint,
                        old_hint,
                        updated_hint,
                        .monotonic,
                        .monotonic,
                    );
                }
            }
        }
        return new_memory.ptr;
    }

    if (new_page_len < old_page_len) {
        @branchHint(.likely);
        const shrink_start = memory.ptr + new_page_len;
        const shrink_len = old_page_len - new_page_len;

        const old_hint = @atomicLoad(@TypeOf(std.heap.next_mmap_addr_hint), &std.heap.next_mmap_addr_hint, .unordered);
        if (old_hint) |hint_ptr| {
            const hint_addr = @intFromPtr(hint_ptr);
            const shrink_start_addr = @intFromPtr(shrink_start);
            if (hint_addr >= shrink_start_addr) {
                const safe_hint: [*]align(page_size_min) u8 = @ptrCast(memory.ptr);
                _ = @cmpxchgStrong(
                    @TypeOf(std.heap.next_mmap_addr_hint),
                    &std.heap.next_mmap_addr_hint,
                    old_hint,
                    safe_hint,
                    .monotonic,
                    .monotonic,
                );
            }
        }

        posix.munmap(@alignCast(shrink_start[0..shrink_len])) catch {};
        return memory.ptr;
    }

    return null;
}
