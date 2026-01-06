const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const types = @import("types.zig");

pub const assert = switch (builtin.mode) {
    .Debug, .ReleaseSafe => std.debug.assert,

    .ReleaseSmall, .ReleaseFast => (struct {
        inline fn assert(ok: bool) void {
            if (!ok) unreachable;
        }
    }).assert,
};

pub inline fn clamp(comptime T: type, sz: T, min: T, max: T) T {
    if (sz < min) {
        return min;
    } else if (sz > max) {
        return max;
    } else sz;
}

// Align a byte size to a size in _machine words_
pub inline fn wsizeCount(size: usize) usize {
    assert(size <= types.SIZE_MAX - @sizeOf(usize));
    return (size + @sizeOf(usize) - 1) / @sizeOf(usize);
}

// Is memory zero initialized?
pub inline fn memIsZero(comptime T: type, buf: []const T) bool {
    if (buf.len <= 1024) {
        @branchHint(.likely);
        return allEqual(T, buf, 0);
    }
    comptime assert(@typeInfo(T) == .int or @typeInfo(T) == .float);
    if (std.simd.suggestVectorLength(T)) |vector_size| {
        @branchHint(.likely);
        const V = @Vector(vector_size, T);
        var i: usize = 0;
        const step = vector_size;

        while (i + step <= buf.len) : (i += step) {
            const chunk = buf[i .. i + step];
            const vec: V = @bitCast(chunk[0..vector_size].*);
            if (@reduce(.Or, vec) != 0) {
                return false;
            }
        }

        while (i < buf.len) : (i += 1) {
            if (buf[i] != 0) return false;
        }
        return true;
    } else {
        @branchHint(.unlikely);
        return allEqual(T, buf, 0);
    }
}

inline fn allEqual(comptime T: type, slice: []const T, scalar: T) bool {
    for (slice) |item| {
        if (item != scalar) return false;
    }
    return true;
}

test "memIsZero: empty buffer" {
    const buf: []const u8 = &[_]u8{};
    try testing.expect(memIsZero(u8, buf));
}

test "memIsZero: agrees with manual check" {
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const rnd = prng.random();

    var buf: [256]u8 = undefined;

    for (0..1000) |_| {
        for (&buf) |*b| {
            b.* = if (rnd.boolean()) 0 else rnd.int(u8);
        }

        var ref = true;
        for (buf) |b| {
            if (b != 0) {
                ref = false;
                break;
            }
        }

        try testing.expect(memIsZero(u8, &buf) == ref);
    }
}

fn bench(
    name: []const u8,
    iters: usize,
    buf: []u8,
    f: fn ([]const u8) bool,
) !u64 {
    var timer = try std.time.Timer.start();
    var sum: usize = 0;
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        buf[0] = @truncate(i);
        sum += @intFromBool(f(buf));
    }
    const ns = timer.read();
    std.mem.doNotOptimizeAway(sum);

    std.debug.print(
        "{s}: {d} ns total, {d} ns/iter\n",
        .{ name, ns, ns / iters },
    );
    return ns;
}

test "memIsZero:u32 agrees with manual check" {
    var prng = std.Random.DefaultPrng.init(0xdeadbeef);
    const rnd = prng.random();

    var buf: [256]u32 = undefined;

    for (0..1000) |_| {
        for (&buf) |*b| {
            b.* = if (rnd.boolean()) 0 else rnd.int(u32);
        }

        var ref = true;
        for (buf) |b| {
            if (b != 0) {
                ref = false;
                break;
            }
        }

        try testing.expect(memIsZero(u32, &buf) == ref);
    }
}

test "bench memIsZero vs allEqual" {
    // var buf = [_]u8{0} ** 4096;
    var buf = [_]u8{0} ** 1024;

    const slice = buf[0..];

    _ = memIsZero(u8, slice);
    _ = std.mem.allEqual(u8, slice, 0);

    const iters = 500;

    const t1 = try bench(
        "memIsZero SIMD",
        iters,
        slice,
        struct {
            fn f(b: []const u8) bool {
                return memIsZero(u8, b);
            }
        }.f,
    );

    const t2 = try bench(
        "std.mem.allEqual",
        iters,
        slice,
        struct {
            fn f(b: []const u8) bool {
                return std.mem.allEqual(u8, b, 0);
            }
        }.f,
    );

    try testing.expect(t1 > 0 and t2 > 0);
}

/// Round an address up to the next (or current) aligned address.
/// The alignment must be a power of 2 and greater than 0.
/// Asserts that rounding up the address does not cause integer overflow.
pub fn alignForward(comptime T: type, addr: T, alignment: T) T {
    assert(isValidAlignGeneric(T, alignment));
    return alignBackward(T, addr + (alignment - 1), alignment);
}

/// Round an address down to the previous (or current) aligned address.
/// The alignment must be a power of 2 and greater than 0.
pub fn alignBackward(comptime T: type, addr: T, alignment: T) T {
    assert(isValidAlignGeneric(T, alignment));
    // 000010000 // example alignment
    // 000001111 // subtract 1
    // 111110000 // binary not
    return addr & ~(alignment - 1);
}

pub fn isAlignedLog2(addr: usize, log2_alignment: u8) bool {
    return @ctz(addr) >= log2_alignment;
}

/// Given an address and an alignment, return true if the address is a multiple of the alignment
/// The alignment must be a power of 2 and greater than 0.
pub fn isAligned(addr: usize, alignment: usize) bool {
    return isAlignedGeneric(u64, addr, alignment);
}

pub fn isAlignedGeneric(comptime T: type, addr: T, alignment: T) bool {
    return alignBackward(T, addr, alignment) == addr;
}

/// Returns whether `alignment` is a valid alignment, meaning it is
/// a positive power of 2.
pub fn isValidAlignGeneric(comptime T: type, alignment: T) bool {
    return alignment > 0 and isPowerOfTwo(alignment);
}

pub inline fn isPowerOfTwo(int: anytype) bool {
    assert(int > 0);
    return (int & (int - 1)) == 0;
}
