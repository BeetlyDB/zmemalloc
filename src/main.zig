const std = @import("std");
const zmemalloc = @import("zmemalloc");

pub fn main() !void {
    const num_ptrs = 1000;
    const iterations = 1000;

    std.debug.print("\n=== Mixed Workload Benchmark ===\n", .{});
    std.debug.print("Pattern: alloc {}, free random half, alloc again, free all\n", .{num_ptrs});
    std.debug.print("Iterations: {}\n\n", .{iterations});

    var prng = std.Random.DefaultPrng.init(12345);
    const random = prng.random();

    // zmemalloc
    const zmem_time = blk: {
        var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
        for (0..iterations) |_| {
            var ptrs: [num_ptrs]?[*]u8 = undefined;

            // Allocate all
            for (&ptrs) |*p| {
                const size = 16 + random.uintLessThan(usize, 1024);
                p.* = zmemalloc.malloc(size);
            }

            // Free random half
            for (&ptrs) |*p| {
                if (random.boolean()) {
                    if (p.*) |ptr| zmemalloc.free_mem(ptr);
                    p.* = null;
                }
            }

            // Allocate again into freed slots
            for (&ptrs) |*p| {
                if (p.* == null) {
                    const size = 16 + random.uintLessThan(usize, 1024);
                    p.* = zmemalloc.malloc(size);
                }
            }

            // Free all
            for (ptrs) |p| {
                if (p) |ptr| zmemalloc.free_mem(ptr);
            }
        }
        break :blk timer.read();
    };

    // smp_allocator
    const smp_time = blk: {
        const smp = std.heap.smp_allocator;
        prng = std.Random.DefaultPrng.init(12345); // Reset PRNG
        var timer = std.time.Timer.start() catch break :blk @as(u64, 0);
        for (0..iterations) |_| {
            var ptrs: [num_ptrs]?[]u8 = undefined;
            @memset(&ptrs, null);

            // Allocate all
            for (&ptrs) |*p| {
                const size = 16 + random.uintLessThan(usize, 1024);
                p.* = smp.alloc(u8, size) catch null;
            }

            // Free random half
            for (&ptrs) |*p| {
                if (random.boolean()) {
                    if (p.*) |slice| smp.free(slice);
                    p.* = null;
                }
            }

            // Allocate again into freed slots
            for (&ptrs) |*p| {
                if (p.* == null) {
                    const size = 16 + random.uintLessThan(usize, 1024);
                    p.* = smp.alloc(u8, size) catch null;
                }
            }

            // Free all
            for (ptrs) |p| {
                if (p) |slice| smp.free(slice);
            }
        }
        break :blk timer.read();
    };

    const zmem_ms = @as(f64, @floatFromInt(zmem_time)) / 1_000_000.0;
    const smp_ms = @as(f64, @floatFromInt(smp_time)) / 1_000_000.0;
    const speedup = if (zmem_ms > 0) smp_ms / zmem_ms else 0.0;

    std.debug.print("{s:<15}: {d:>10.2} ms\n", .{ "zmemalloc", zmem_ms });
    std.debug.print("{s:<15}: {d:>10.2} ms\n", .{ "smp_allocator", smp_ms });
    std.debug.print("{s:<15}: {d:>10.2}x\n\n", .{ "Speedup", speedup });
}
