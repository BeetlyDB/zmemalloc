const std = @import("std");
const zmemalloc = @import("zmemalloc");
const mimalloc = @import("mimalloc_zig");

pub fn main() !void {
    const num_ptrs = 10000;
    const iterations = 1000;

    std.debug.print("\n=== Mixed Workload Benchmark ===\n", .{});
    std.debug.print("Pattern: alloc {}, free random half, alloc again, free all\n", .{num_ptrs});
    std.debug.print("Iterations: {}\n\n", .{iterations});

    // zmemalloc

    // smp_allocator
    const smp_time = blk: {
        const smp = std.heap.smp_allocator;
        var local_prng = std.Random.DefaultPrng.init(12345);
        const random = local_prng.random();
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

    const zmem_time = blk: {
        const smp = zmemalloc.allocator();
        var local_prng = std.Random.DefaultPrng.init(12345);
        const random = local_prng.random();

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

    // std.debug.print("Fast success: {}, No page: {}, Exhausted: {}\n", .{
    //     zmemalloc.dbg_fast_success,
    //     zmemalloc.dbg_fast_no_page,
    //     zmemalloc.dbg_fast_exhausted,
    // });
    // std.debug.print("Cold path: {}, Page readded: {}\n", .{
    //     zmemalloc.dbg_cold_path_calls,
    //     zmemalloc.dbg_page_readded,
    // });

    // // smp_allocator
    const mimalloc_time = blk: {
        const smp = mimalloc.mimalloc_allocator;
        var local_prng = std.Random.DefaultPrng.init(12345);
        const random = local_prng.random();

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
    //
    const c_time = blk: {
        const c = std.heap.c_allocator;
        var local_prng = std.Random.DefaultPrng.init(12345);
        const random = local_prng.random();

        var timer = std.time.Timer.start() catch break :blk @as(u64, 0);

        for (0..iterations) |_| {
            var ptrs: [num_ptrs]?[]u8 = undefined;
            @memset(&ptrs, null);

            // Allocate all
            for (&ptrs) |*p| {
                const size = 16 + random.uintLessThan(usize, 1024);
                p.* = c.alloc(u8, size) catch null;
            }

            // Free random half
            for (&ptrs) |*p| {
                if (random.boolean()) {
                    if (p.*) |slice| c.free(slice);
                    p.* = null;
                }
            }

            // Allocate again into freed slots
            for (&ptrs) |*p| {
                if (p.* == null) {
                    const size = 16 + random.uintLessThan(usize, 1024);
                    p.* = c.alloc(u8, size) catch null;
                }
            }

            // Free all
            for (ptrs) |p| {
                if (p) |slice| c.free(slice);
            }
        }

        break :blk timer.read();
    };
    //
    // // ------------------------------------------------------------
    // // Results
    // // ------------------------------------------------------------
    const zmem_ms = @as(f64, @floatFromInt(zmem_time)) / 1_000_000.0;
    const smp_ms = @as(f64, @floatFromInt(smp_time)) / 1_000_000.0;
    const mi_ms = @as(f64, @floatFromInt(mimalloc_time)) / 1_000_000.0;
    const c_ms = @as(f64, @floatFromInt(c_time)) / 1_000_000.0;

    const c_speedup = if (zmem_ms > 0) c_ms / zmem_ms else 0.0;

    const smp_speedup = if (zmem_ms > 0) smp_ms / zmem_ms else 0.0;
    const mi_speedup = if (zmem_ms > 0) mi_ms / zmem_ms else 0.0;

    std.debug.print("{s:<15}: {d:>10.2} ms\n", .{ "zmemalloc", zmem_ms });
    std.debug.print("{s:<15}: {d:>10.2} ms ({d:.2}x)\n", .{
        "smp_allocator", smp_ms, smp_speedup,
    });
    std.debug.print("{s:<15}: {d:>10.2} ms ({d:.2}x)\n\n", .{
        "mimalloc", mi_ms, mi_speedup,
    });

    std.debug.print("{s:<15}: {d:>10.2} ms ({d:.2}x)\n\n", .{
        "c_allocator", c_ms, c_speedup,
    });

    bench();
    multiThreadBench();
}

fn bench() void {
    const iterations = 100_000;
    const sizes = [_]usize{
        16,
        64,
        256,
        1024,
        4096,
        8192,
        16384,
        32768,
        65536,
        65536 * 2,
    };

    std.debug.print(
        "\n=== Allocator Benchmark ({} iterations per size) ===\n",
        .{iterations},
    );

    std.debug.print(
        "{s:>8} | {s:>6} | {s:>14} | {s:>14} | {s:>14}\n",
        .{ "Size", "zmemalloc", "smp", "mimalloc", "c_alloc" },
    );
    std.debug.print(
        "{s:-^8}-+-{s:-^6}-+-{s:-^14}-+-{s:-^14}-+-{s:-^14}\n",
        .{ "", "", "", "", "" },
    );

    for (sizes) |size| {
        // zmemalloc
        const zmem_time = blk: {
            const a = zmemalloc.allocator();
            var timer = std.time.Timer.start() catch break :blk 0;
            for (0..iterations) |_| {
                const ptr = a.alloc(u8, size) catch continue;
                a.free(ptr);
            }
            break :blk timer.read();
        };

        // smp_allocator
        const smp_time = blk: {
            const a = std.heap.smp_allocator;
            var timer = std.time.Timer.start() catch break :blk 0;
            for (0..iterations) |_| {
                const ptr = a.alloc(u8, size) catch continue;
                a.free(ptr);
            }
            break :blk timer.read();
        };

        // mimalloc
        const mi_time = blk: {
            const a = mimalloc.mimalloc_allocator;
            var timer = std.time.Timer.start() catch break :blk 0;
            for (0..iterations) |_| {
                const ptr = a.alloc(u8, size) catch continue;
                a.free(ptr);
            }
            break :blk timer.read();
        };

        // c_allocator
        const c_time = blk: {
            const a = std.heap.c_allocator;
            var timer = std.time.Timer.start() catch break :blk 0;
            for (0..iterations) |_| {
                const ptr = a.alloc(u8, size) catch continue;
                a.free(ptr);
            }
            break :blk timer.read();
        };

        const zmem_ns = zmem_time / iterations;
        const smp_ns = smp_time / iterations;
        const mi_ns = mi_time / iterations;
        const c_ns = c_time / iterations;

        const smp_speedup =
            @as(f64, @floatFromInt(smp_ns)) / @as(f64, @floatFromInt(zmem_ns));
        const mi_speedup =
            @as(f64, @floatFromInt(mi_ns)) / @as(f64, @floatFromInt(zmem_ns));
        const c_speedup =
            @as(f64, @floatFromInt(c_ns)) / @as(f64, @floatFromInt(zmem_ns));

        std.debug.print(
            "{d:>8} | {d:>6} ns | {d:>6} ns ({d:>4.2}x) | {d:>6} ns ({d:>4.2}x) | {d:>6} ns ({d:>4.2}x)\n",
            .{
                size,
                zmem_ns,
                smp_ns,
                smp_speedup,
                mi_ns,
                mi_speedup,
                c_ns,
                c_speedup,
            },
        );
    }

    std.debug.print("\n", .{});
}

fn multiThreadBench() void {
    const num_threads = 12;
    const num_ptrs = 10000;
    const iterations = 1000;
    std.debug.print("\n=== Multi-Threaded Mixed Workload Benchmark ===\n", .{});
    std.debug.print("Threads: {}\n", .{num_threads});
    std.debug.print("Per thread: alloc {}, free random half, alloc again, free all\n", .{num_ptrs});
    std.debug.print("Iterations per thread: {}\n\n", .{iterations});
    // zmemalloc
    const zmem_time = timedMultiThreadBench(zmemalloc.allocator(), num_threads, num_ptrs, iterations);
    // smp_allocator
    const smp_time = timedMultiThreadBench(std.heap.smp_allocator, num_threads, num_ptrs, iterations);
    // mimalloc
    const mimalloc_time = timedMultiThreadBench(mimalloc.mimalloc_allocator, num_threads, num_ptrs, iterations);
    // c_allocator
    const c_time = timedMultiThreadBench(std.heap.c_allocator, num_threads, num_ptrs, iterations);
    const zmem_ms = @as(f64, @floatFromInt(zmem_time)) / 1_000_000.0;
    const smp_ms = @as(f64, @floatFromInt(smp_time)) / 1_000_000.0;
    const mi_ms = @as(f64, @floatFromInt(mimalloc_time)) / 1_000_000.0;
    const c_ms = @as(f64, @floatFromInt(c_time)) / 1_000_000.0;
    const smp_speedup = if (zmem_ms > 0) smp_ms / zmem_ms else 0.0;
    const mi_speedup = if (zmem_ms > 0) mi_ms / zmem_ms else 0.0;
    const c_speedup = if (zmem_ms > 0) c_ms / zmem_ms else 0.0;
    std.debug.print("{s:<15}: {d:>10.2} ms\n", .{ "zmemalloc", zmem_ms });
    std.debug.print("{s:<15}: {d:>10.2} ms ({d:.2}x)\n", .{
        "smp_allocator", smp_ms, smp_speedup,
    });
    std.debug.print("{s:<15}: {d:>10.2} ms ({d:.2}x)\n", .{
        "mimalloc", mi_ms, mi_speedup,
    });
    std.debug.print("{s:<15}: {d:>10.2} ms ({d:.2}x)\n\n", .{
        "c_allocator", c_ms, c_speedup,
    });
}

fn timedMultiThreadBench(alloc: std.mem.Allocator, comptime num_threads: usize, comptime num_ptrs: usize, comptime iterations: usize) u64 {
    var threads: [num_threads]std.Thread = undefined;
    var timer = std.time.Timer.start() catch return 0;
    for (0..num_threads) |i| {
        threads[i] = std.Thread.spawn(.{}, threadWorkload, .{ alloc, num_ptrs, iterations }) catch |err| {
            std.debug.print("Failed to spawn thread {}: {}\n", .{ i, err });
            continue;
        };
    }
    for (threads) |th| {
        th.join();
    }
    return timer.read();
}

fn threadWorkload(alloc: std.mem.Allocator, comptime num_ptrs: usize, iterations: usize) void {
    var local_prng = std.Random.DefaultPrng.init(@as(u64, std.Thread.getCurrentId()));
    const random = local_prng.random();
    for (0..iterations) |_| {
        var ptrs: [num_ptrs]?[]u8 = undefined;
        @memset(&ptrs, null);
        // Allocate all
        for (&ptrs) |*p| {
            const size = 16 + random.uintLessThan(usize, 1024);
            p.* = alloc.alloc(u8, size) catch null;
        }
        // Free random half
        for (&ptrs) |*p| {
            if (random.boolean()) {
                if (p.*) |slice| alloc.free(slice);
                p.* = null;
            }
        }
        // Allocate again into freed slots
        for (&ptrs) |*p| {
            if (p.* == null) {
                const size = 16 + random.uintLessThan(usize, 1024);
                p.* = alloc.alloc(u8, size) catch null;
            }
        }
        // Free all
        for (ptrs) |p| {
            if (p) |slice| alloc.free(slice);
        }
    }
}
