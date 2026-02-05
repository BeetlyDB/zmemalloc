const std = @import("std");
const utils = @import("util.zig");
const assert = utils.assert;
const linux = std.os.linux;
const posix = std.posix;
const stats = @import("stats.zig");
const types = @import("types.zig");
const builtin = @import("builtin");
const Atomic = std.atomic.Value;
const testing = std.testing;
const config = @import("config");

pub const DEFAULT_PHYSICAL_MEMORY_IN_KIB = if (types.INTPTR_SIZE < 8) 4 * types.MiB else 8 * types.MiB;
pub const DEFAULT_VIRTUAL_ADDRESS = if (types.INTPTR_SIZE < 8) 32 else 48;

var reset_advice: Atomic(u32) = .{ .raw = if (@hasDecl(posix.MADV, "FREE")) posix.MADV.FREE else posix.MADV.DONTNEED };

pub const OsMemConfig = struct {
    const Self = @This();
    page_size: usize = config.Config.page_size,
    large_page_size: usize = std.heap.page_size_max,
    allocation_granularity: usize = std.heap.page_size_min,
    physical_memory_kib: usize = config.Config.physical_memory_kib,
    virtual_address_bits: usize = DEFAULT_VIRTUAL_ADDRESS,
    has_overcommit: bool = config.Config.has_overcommit,
    has_partial_free: bool = true,
    has_virtual_reserve: bool = true,
    thp: ThpMode = if (config.Config.thp) .thp_mode_always else .thp_mode_default,
    has_numa_available: bool = false,
    numa_nodes: usize = 1,

    pub const ThpMode = union(enum) {
        thp_mode_default, // Do not change hugepage settings. */
        thp_mode_always, // Always set MADV_HUGEPAGE. */
    };

    //WARN: numa does not have support right now

    // pub inline fn init(cfg: *Self) void { //TODO: make init in comptime and get in static
    //     cfg.page_size = std.heap.pageSize();
    //     cfg.allocation_granularity = cfg.page_size;
    //
    //     cfg.physical_memory_kib = phisical_memory() orelse DEFAULT_PHYSICAL_MEMORY_IN_KIB;
    //
    //     cfg.has_overcommit = unix_detect_overcommit();
    //     cfg.has_partial_free = true; //mmap can free in parts
    //     cfg.has_virtual_reserve = true; //always true for linux
    //     cfg.thp = unix_detect_thp();
    //     cfg.virtual_address_bits = DEFAULT_VIRTUAL_ADDRESS;
    // }
};

pub const mem_config_static: OsMemConfig = .{};

pub inline fn maybe_enable_thp(
    mem: [*]align(mem_config_static.page_size) u8,
    len: usize,
    alignment_bytes: usize,
    mode: OsMemConfig.ThpMode,
) void {
    if (len < 8 * 1024 * 1024) return;
    if (mode == .thp_mode_always and (builtin.mode == .ReleaseSafe or builtin.mode == .Debug)) {
        const huge = mem_config_static.large_page_size;

        if (len < huge) return;
        if (alignment_bytes < huge) return;
        if (@intFromPtr(mem) % huge != 0) return;
    }

    const p = switch (mode) {
        .thp_mode_always => posix.MADV.HUGEPAGE,
        .thp_mode_default => return,
    };

    posix.madvise(mem, len, p) catch {};
}

pub inline fn prim_numa_node() usize {
    var node: usize = 0;
    var cpu: usize = 0;
    const s = linux.syscall2(.getcpu, &cpu, &node);
    if (s != 0) return 0;
    return node;
}

pub inline fn prim_commit(mem: []align(mem_config_static.page_size) u8, is_zero: *bool) !void {
    if (mem.len == 0) return;

    is_zero.* = false;
    // commit: ensure we can access the area
    // note: we may think that *is_zero can be true since the memory
    // was either from mmap PROT_NONE, or from decommit MADV_DONTNEED, but
    // we sometimes call commit on a range with still partially committed
    // memory and `mprotect` does not zero the range.
    try posix.mprotect(mem, posix.PROT.READ | posix.PROT.WRITE);
}

pub inline fn prim_decommit(mem: []align(mem_config_static.page_size) u8, needs_recommit: *bool) !void {
    if (mem.len == 0) return;

    needs_recommit.* = false;
    // decommit: use MADV_DONTNEED as it decreases rss immediately (unlike MADV_FREE)
    try posix.madvise(mem.ptr, mem.len, posix.MADV.DONTNEED);
}

pub inline fn prim_protect(mem: []align(mem_config_static.page_size) u8, do_protect: bool) !void {
    if (mem.len == 0) return;
    const prot = if (do_protect) posix.PROT.NONE else (posix.PROT.READ | posix.PROT.WRITE);

    try posix.mprotect(mem, prot);
}

pub inline fn prim_reset(mem: []align(mem_config_static.page_size) u8) !void {
    if (mem.len == 0) return;

    const advice = reset_advice.load(.monotonic);

    while (true) {
        const res = linux.madvise(mem.ptr, mem.len, advice);
        if (res == 0) return;

        const e = posix.errno(res);
        switch (e) {
            .AGAIN => continue,

            .INVAL => {
                if (advice == posix.MADV.FREE) {
                    reset_advice.store(posix.MADV.DONTNEED, .release);

                    const r2 = linux.madvise(mem.ptr, mem.len, posix.MADV.DONTNEED);
                    if (r2 == 0) return;

                    const e2 = posix.errno(r2);
                    if (e2 == .AGAIN) continue;
                    return posix.unexpectedErrno(e2);
                }
                return error.InvalidSyscall;
            },

            else => return posix.unexpectedErrno(e),
        }
    }
}

pub inline fn os_use_large_page(size: usize, alignment: usize) bool {
    return (size % mem_config_static.large_page_size == 0) and (alignment % mem_config_static.large_page_size == 0);
}

const NumaInfo = struct {
    has_numa: bool,
    node_count: usize,
};

pub fn unix_detect_numa() NumaInfo {
    const dir_path = "/sys/devices/system/node/";

    var dir = std.fs.cwd().openDir(dir_path, .{ .iterate = true }) catch return .{ .has_numa = false, .node_count = 1 };
    defer dir.close();
    var iter = dir.iterate();
    var node_count: usize = 0;
    while (true) {
        const entry = iter.next() catch break;
        if (entry == null) break;
        if (std.mem.startsWith(u8, entry.?.name, "node")) {
            node_count += 1;
        }
        return .{ .has_numa = true, .node_count = node_count };
    }
    return .{ .has_numa = false, .node_count = 1 };
}

pub inline fn os_good_size(size: usize) usize {
    if (size == 0) return 0;

    const align_size = if (size < 512 * types.KiB)
        mem_config_static.page_size
    else if (size < 2 * types.MiB)
        64 * types.KiB
    else if (size < 8 * types.MiB)
        256 * types.KiB
    else if (size < 32 * types.MiB)
        1 * types.MiB
    else
        4 * types.MiB;

    if (size >= (types.SIZE_MAX - align_size)) return size;
    return utils.alignForward(usize, size, align_size);
}

pub fn os_commit_ex(mem: []u8, is_zero: *bool, stat_size: usize) bool {
    is_zero.* = false;
    stats.main_stats.addCount(.committed, @intCast(stat_size));
    stats.main_stats.incCounter(.commit_calls);

    var csize: usize = undefined;
    const start = page_align_area(false, mem.ptr, mem.len, &csize) orelse return true;
    if (csize == 0) return true;

    //commit
    var os_zero: bool = false;
    prim_commit(start[0..], &os_zero) catch false;
    return true;
}

pub fn os_decommit_ex(mem: []u8, needs_recommit: *bool, stat_size: usize) bool {
    needs_recommit.* = true;

    stats.main_stats.subCount(.committed, @intCast(stat_size));

    var csize: usize = undefined;
    const start = page_align_area(true, mem.ptr, mem.len, &csize) orelse return true;

    prim_decommit(start[0..], &needs_recommit) catch false;
    return true;
}

pub fn os_reset(mem: []u8) bool {
    var csize: usize = undefined;
    const start = page_align_area(true, mem.ptr, mem.len, &csize) orelse return true;

    stats.main_stats.addCount(._reset, @intCast(csize));
    stats.main_stats.incCounter(.reset_calls);

    prim_reset(start[0..]) catch false;
    return true;
}

pub inline fn page_align_area(
    conservative: bool,
    addr: [*]u8,
    size: usize,
    new_size: *usize,
) ?[*]u8 {
    new_size.* = 0;
    if (size == 0) return null;

    const page = mem_config_static.page_size;
    const start_ptr = @intFromPtr(addr);
    const end_ptr = start_ptr + size;

    const start = if (conservative)
        utils.alignForward(usize, start_ptr, page)
    else
        utils.alignBackward(usize, start_ptr, page);

    const end = if (conservative)
        utils.alignBackward(usize, end_ptr, page)
    else
        utils.alignForward(usize, end_ptr, page);

    const diff = if (end > start) end - start else 0;
    if (diff == 0) return null;

    new_size.* = diff;
    return @ptrFromInt(start);
}

fn test_mmapPages(len: usize) ![]align(mem_config_static.page_size) u8 {
    const prot = posix.PROT.READ | posix.PROT.WRITE;
    const ptr = try posix.mmap(
        null,
        len,
        prot,
        .{ .ANONYMOUS = true, .TYPE = .PRIVATE },
        -1,
        0,
    );
    return ptr[0..len];
}

fn test_munmapPages(mem: []align(mem_config_static.page_size) u8) void {
    posix.munmap(mem);
}

test "page_align_area conservative and non-conservative" {
    const page = mem_config_static.page_size;

    var backing: [page * 4]u8 = undefined;
    var new_size: usize = 0;

    const base = @intFromPtr(&backing);
    const addr: [*]u8 = @ptrFromInt(base + 1);

    // conservative: inside
    const p1 = page_align_area(true, addr, page * 2, &new_size);
    try testing.expect(p1 != null);
    try testing.expectEqual(page, new_size);

    // non-conservative: outside
    const p2 = page_align_area(false, addr, page * 2, &new_size);
    try testing.expect(p2 != null);
    try testing.expectEqual(page * 3, new_size);
}

test "page_align_area invariants" {
    const page = mem_config_static.page_size;

    var backing: [page * 4]u8 = undefined;
    var new_size: usize = 0;

    const base = @intFromPtr(&backing);
    const addr: [*]u8 = @ptrFromInt(base + 1);
    const size = page * 2;

    const p = page_align_area(false, addr, size, &new_size).?;
    const start = @intFromPtr(p);
    const end = start + new_size;

    // start page-aligned
    try testing.expect(start % page == 0);

    // end page-aligned
    try testing.expect(end % page == 0);

    try testing.expect(start <= base + 1);
    try testing.expect(end >= base + 1 + size);
}

test "prim_commit and prim_decommit basic" {
    const page = mem_config_static.page_size;
    const mem = try test_mmapPages(page * 2);
    defer test_munmapPages(mem);

    var is_zero = true;
    try prim_commit(mem, &is_zero);
    try testing.expect(!is_zero);

    var needs_recommit = true;
    try prim_decommit(mem, &needs_recommit);
    try testing.expect(!needs_recommit);
}

test "prim_reset works on valid memory" {
    const page = mem_config_static.page_size;
    const mem = try test_mmapPages(page * 4);
    defer test_munmapPages(mem);

    try prim_reset(mem);
}

test "os_good_size alignment logic" {
    const KiB = types.KiB;
    const MiB = types.MiB;

    try testing.expectEqual(0, os_good_size(0));
    try testing.expect(os_good_size(100 * KiB) % mem_config_static.page_size == 0);
    try testing.expect(os_good_size(1 * MiB) % (64 * KiB) == 0);
    try testing.expect(os_good_size(16 * MiB) % (1 * MiB) == 0);
}

test "os_use_large_page logic" {
    const lp = mem_config_static.large_page_size;

    try testing.expect(os_use_large_page(lp, lp));
    try testing.expect(!os_use_large_page(lp + 1, lp));
    try testing.expect(!os_use_large_page(lp, lp / 2));
}

test "prim_reset fallback from MADV_FREE to DONTNEED" {
    reset_advice.store(posix.MADV.FREE, .monotonic);

    const page = mem_config_static.page_size;

    const prot = posix.PROT.READ | posix.PROT.WRITE;

    const mem = try posix.mmap(
        null,
        page,
        prot,
        //panic with .HUGETLB its right
        .{
            .TYPE = .SHARED,
            .ANONYMOUS = true,
            .NORESERVE = true,
        },
        -1,
        0,
    );
    defer std.posix.munmap(mem);

    // aligned, len > 0 → syscall
    _ = prim_reset(mem) catch {};

    const new_advice = reset_advice.load(.monotonic);
    try testing.expectEqual(posix.MADV.DONTNEED, new_advice);
}

test "OsMemConfig init sets defaults correctly" {
    var cfg: OsMemConfig = undefined;
    OsMemConfig.init(&cfg);

    try testing.expect(cfg.page_size >= 4096);
    try testing.expect(cfg.allocation_granularity == cfg.page_size);
    std.debug.print("memory_in_kib: {}\n", .{cfg.physical_memory_kib});
    std.debug.print("default_in_kib: {}\n", .{DEFAULT_PHYSICAL_MEMORY_IN_KIB});
    try testing.expect(cfg.physical_memory_kib > DEFAULT_PHYSICAL_MEMORY_IN_KIB);
    try testing.expect(cfg.virtual_address_bits == DEFAULT_VIRTUAL_ADDRESS);
    try testing.expect(cfg.has_partial_free);
    try testing.expect(cfg.has_virtual_reserve);
}

pub inline fn pageSize() usize {
    return mem_config_static.page_size;
}

/// Commit memory (make it accessible)
pub fn commitEx(mem: []u8, is_zero: *bool) bool {
    is_zero.* = false;
    if (mem.len == 0) return true;

    var csize: usize = undefined;
    const start = page_align_area(false, mem.ptr, mem.len, &csize) orelse return true;
    if (csize == 0) return true;

    var os_zero: bool = false;
    prim_commit(start[0..csize], &os_zero) catch return false;
    return true;
}

/// Commit memory
pub fn commit(ptr: [*]u8, size: usize) bool {
    if (size == 0) return true;

    var csize: usize = undefined;
    const start = page_align_area(false, ptr, size, &csize) orelse return true;
    if (csize == 0) return true;

    var os_zero: bool = false;
    const aligned_start: [*]align(mem_config_static.page_size) u8 = @alignCast(start);
    prim_commit(aligned_start[0..csize], &os_zero) catch return false;

    stats.main_stats.addCount(.committed, @intCast(csize));
    return true;
}

/// Purge memory (advise kernel it's not needed, may be decommitted)
pub fn purge(mem: []u8) bool {
    if (mem.len == 0) return false;

    var csize: usize = undefined;
    const start = page_align_area(true, mem.ptr, mem.len, &csize) orelse return false;
    if (csize == 0) return false;

    var needs_recommit: bool = false;
    prim_decommit(start[0..csize], &needs_recommit) catch return false;
    return needs_recommit;
}

/// Protect memory (make it inaccessible)
pub fn protect(mem: []u8) !void {
    if (mem.len == 0) return;

    var csize: usize = undefined;
    const start = page_align_area(false, mem.ptr, mem.len, &csize) orelse return;
    if (csize == 0) return;

    try prim_protect(start[0..csize], true);
}

/// Unprotect memory (make it accessible again)
pub fn unprotect(mem: []u8) !void {
    if (mem.len == 0) return;

    var csize: usize = undefined;
    const start = page_align_area(false, mem.ptr, mem.len, &csize) orelse return;
    if (csize == 0) return;

    try prim_protect(start[0..csize], false);
}

/// Decommit memory (make it not backed by physical memory)
pub fn decommit(ptr: [*]u8, size: usize) bool {
    if (size == 0) return true;

    var csize: usize = undefined;
    const start = page_align_area(true, ptr, size, &csize) orelse return true;
    if (csize == 0) return true;

    var needs_recommit: bool = false;
    prim_decommit(start[0..csize], &needs_recommit) catch return false;

    stats.main_stats.subCount(.committed, @intCast(csize));
    return true;
}

/// Check if large pages (huge pages) are supported on this system
pub fn supportsLargePages() bool {
    // On Linux, check if huge pages are available
    return mem_config_static.large_page_size > mem_config_static.page_size;
}
