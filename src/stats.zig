const std = @import("std");
const assert = @import("util.zig").assert;
const types = @import("types.zig");

const BIN_HUGE = 73;

// count allocation over time
pub const StatCount = struct {
    total: i64 = 0,
    peak: i64 = 0,
    current: i64 = 0,

    pub inline fn statIncrease(stat: *StatCount, amount: i64) void {
        stat.total += amount;
        stat.current += amount;
        if (stat.current > stat.peak) stat.peak = stat.current;
    }

    pub inline fn statDecrease(stat: *StatCount, amount: i64) void {
        stat.current -= amount;
    }
};

// only grow
pub const StatCounter = struct {
    total: i64 = 0,
    pub inline fn statCounterIncrease(counter: *StatCounter, amount: i64) void {
        counter.total += amount;
    }
};

pub const ChunkBin = enum(u8) {
    small, // slice_count == 1
    other, // 1 <= slice_count <= MI_BCHUNK_BITS
    medium, // slice_count == 8
    large, // slice_count == MI_SIZE_BITS
    none, // free
};

pub const CHUNK_BIN_COUNT = @typeInfo(ChunkBin).@"enum".fields.len;

pub const Stats = struct {

    // base stats
    pages: StatCount = .{},
    reserved: StatCount = .{},
    committed: StatCount = .{},
    _reset: StatCount = .{},
    purged: StatCount = .{},
    page_committed: StatCount = .{},
    pages_abandoned: StatCount = .{},
    threads: StatCount = .{},

    malloc_normal: StatCount = .{},
    malloc_huge: StatCount = .{},
    malloc_requested: StatCount = .{},

    // calls
    mmap_calls: StatCounter = .{},
    commit_calls: StatCounter = .{},
    reset_calls: StatCounter = .{},
    purge_calls: StatCounter = .{},
    arena_count: StatCounter = .{},
    malloc_normal_count: StatCounter = .{},
    malloc_huge_count: StatCounter = .{},
    malloc_guarded_count: StatCounter = .{},

    // internal stats
    arena_rollback_count: StatCounter = .{},
    arena_purges: StatCounter = .{},
    pages_extended: StatCounter = .{},
    pages_retire: StatCounter = .{},
    page_searches: StatCounter = .{},

    segments: StatCount = .{},
    segments_abandoned: StatCount = .{},
    segments_cache: StatCount = .{},
    _segments_reserved: StatCount = .{},

    pages_reclaim_on_alloc: StatCounter = .{},
    pages_reclaim_on_free: StatCounter = .{},
    pages_reabandon_full: StatCounter = .{},
    pages_unabandon_busy_wait: StatCounter = .{},

    // stat on bins
    malloc_bins: [BIN_HUGE + 1]StatCount = [_]StatCount{.{}} ** (BIN_HUGE + 1),
    page_bins: [BIN_HUGE + 1]StatCount = [_]StatCount{.{}} ** (BIN_HUGE + 1),
    chunk_bins: [CHUNK_BIN_COUNT]StatCount = [_]StatCount{.{}} ** CHUNK_BIN_COUNT,

    pub fn reset(self: *Stats) void {
        self.* = std.mem.zeroes(Stats);
    }

    pub fn addCount(self: *Stats, comptime field: std.meta.FieldEnum(Stats), amount: i64) void {
        const f = &@field(self, @tagName(field));
        f.total += amount;
        f.current += amount;
        if (f.current > f.peak) f.peak = f.current;
    }

    pub fn subCount(self: *Stats, comptime field: std.meta.FieldEnum(Stats), amount: i64) void {
        const f = &@field(self, @tagName(field));
        f.current -= amount;
    }

    pub fn incCounter(self: *Stats, comptime field: std.meta.FieldEnum(Stats)) void {
        const f = &@field(self, @tagName(field));
        f.total += 1;
    }
};

pub threadlocal var main_stats: Stats = .{};
