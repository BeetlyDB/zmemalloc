//! # Adaptive Mutex Implementation
//!
//! High-performance mutex using Linux futex with adaptive spinning.
//! Optimized for low-contention scenarios common in memory allocators.
//!
//! ## Design
//!
//! Three states tracked in a single atomic u32:
//! - `UNLOCKED` (0b00): Mutex is free
//! - `LOCKED` (0b01): Mutex held, no waiters
//! - `CONTENDED` (0b11): Mutex held, has waiters
//!
//! ## Fast Path
//!
//! Uses `lock bts` on x86 for single-instruction atomic test-and-set.
//! If uncontended, lock/unlock is just two atomic operations.
//!
//! ## Slow Path
//!
//! On contention:
//! 1. Spin briefly (50 iterations) with exponential backoff
//! 2. If still contended, fall back to futex wait
//! 3. Mark as CONTENDED so unlock will wake waiters
//!
//! ## Usage
//!
//! ```zig
//! var mutex = Mutex{};
//! mutex.lock();
//! defer mutex.unlock();
//! // critical section
//! ```

const std = @import("std");
const builtin = @import("builtin");
const Futex = std.Thread.Futex;
const assert = @import("util.zig").assert;

/// Mutex with adaptive spinning and x86_64 bitset optimization
pub const Mutex = struct {
    comptime {
        if (builtin.os.tag != .linux) {
            @panic("this implementation only for linux");
        }
    }
    state: std.atomic.Value(u32) align(std.atomic.cache_line) = std.atomic.Value(u32).init(UNLOCKED),

    const unlocked: u32 = 0b00;
    const locked: u32 = 0b01;
    const contended: u32 = 0b11; // locked + has waiters
    //
    const UNLOCKED = unlocked;
    const LOCKED = locked;
    const CONTENDED = contended;

    inline fn wake(self: *Mutex) void {
        std.Thread.Futex.wake(&self.state, 1);
    }

    /// Try to acquire the lock without blocking
    pub inline fn trylock(self: *Mutex) bool {
        @branchHint(.likely);
        // On x86  `lock bts` - smaller instruction, better for inlining
        if (comptime builtin.target.cpu.arch.isX86()) {
            return self.state.bitSet(@ctz(LOCKED), .acquire) == 0;
        }

        return self.state.cmpxchgWeak(UNLOCKED, LOCKED, .acquire, .monotonic) == null;
    }

    pub inline fn lock(self: *Mutex) void {
        if (!self.trylock()) {
            self.lockSlow();
        }
    }

    pub inline fn unlock(self: *Mutex) void {
        // Release barrier ensures critical section happens before unlock
        switch (self.state.swap(UNLOCKED, .release)) {
            UNLOCKED => unreachable,
            LOCKED => {},
            CONTENDED => self.wake(),
            else => unreachable,
        }
    }

    fn lockSlow(self: *Mutex) void {
        @branchHint(.cold);
        var current_state = self.state.load(.monotonic);
        if (current_state == UNLOCKED) {
            if (self.trylock()) return;
            current_state = self.state.load(.monotonic);
        }

        if (current_state == LOCKED) {
            // Low contention
            var spin: u8 = 50;
            while (spin > 0) : (spin -= 1) {
                std.atomic.spinLoopHint();
                current_state = self.state.load(.monotonic);
                if (current_state == UNLOCKED) {
                    if (self.trylock()) return;
                } else if (current_state == CONTENDED) {
                    break;
                }
            }
        }

        if (current_state == CONTENDED) {
            self.wait(CONTENDED);
        }

        // Acquire with `contended` so next unlocker wakes another thread
        while (self.state.swap(CONTENDED, .acquire) != UNLOCKED) {
            self.wait(CONTENDED);
        }
    }
    inline fn wait(self: *Mutex, expect: u32) void {
        std.Thread.Futex.wait(&self.state, expect);
    }
};

test "Mutex: basic lock and unlock" {
    var mutex = Mutex{};
    try std.testing.expectEqual(mutex.state.raw, Mutex.UNLOCKED);
    mutex.lock();
    try std.testing.expect(mutex.state.raw != Mutex.UNLOCKED);
    mutex.unlock();
    try std.testing.expectEqual(mutex.state.raw, Mutex.UNLOCKED);
}

test "Mutex: trylock" {
    var mutex = Mutex{};
    try std.testing.expect(mutex.trylock()); // Should succeed
    try std.testing.expect(mutex.state.raw != Mutex.UNLOCKED);
    try std.testing.expect(!mutex.trylock()); // Should fail
    mutex.unlock();
    try std.testing.expectEqual(mutex.state.raw, Mutex.UNLOCKED);
}

test "Mutex: concurrent access" {
    const Counter = struct {
        value: i32 = 0,
        mutex: Mutex = .{},
        fn increment(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.value += 1;
        }
    };

    var counter = Counter{};
    const num_threads = 10;
    const increments_per_thread = 1000;
    var threads: [num_threads]std.Thread = undefined;

    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(c: *Counter) !void {
                for (0..increments_per_thread) |_| {
                    c.increment();
                }
            }
        }.run, .{&counter});
    }

    for (threads) |t| t.join();
    try std.testing.expectEqual(counter.value, @as(i32, num_threads * increments_per_thread));
}

test "Mutex: benchmark" {
    std.debug.print("\n", .{});

    // Test 1: High contention (many threads, short critical section)
    {
        const num_threads = 16;
        const ops_per_thread = 10000;

        var custom_mutex: Mutex = .{};
        var std_mutex: std.Thread.Mutex = .{};
        var custom_counter: i64 = 0;
        var std_counter: i64 = 0;

        // Custom mutex benchmark
        var threads: [num_threads]std.Thread = undefined;
        const start_custom = std.time.nanoTimestamp();
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, struct {
                fn run(m: *Mutex, c: *i64) void {
                    for (0..ops_per_thread) |_| {
                        m.lock();
                        c.* += 1;
                        m.unlock();
                    }
                }
            }.run, .{ &custom_mutex, &custom_counter });
        }
        for (threads) |t| t.join();
        const custom_ns = std.time.nanoTimestamp() - start_custom;

        // Std mutex benchmark
        const start_std = std.time.nanoTimestamp();
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, struct {
                fn run(m: *std.Thread.Mutex, c: *i64) void {
                    for (0..ops_per_thread) |_| {
                        m.lock();
                        c.* += 1;
                        m.unlock();
                    }
                }
            }.run, .{ &std_mutex, &std_counter });
        }
        for (threads) |t| t.join();
        const std_ns = std.time.nanoTimestamp() - start_std;

        std.debug.print("High contention ({} threads, {} ops):\n", .{ num_threads, num_threads * ops_per_thread });
        std.debug.print("  Custom: {} ns ({d:.0} ns/op)\n", .{ custom_ns, @as(f64, @floatFromInt(custom_ns)) / @as(f64, num_threads * ops_per_thread) });
        std.debug.print("  Std:    {} ns ({d:.0} ns/op)\n", .{ std_ns, @as(f64, @floatFromInt(std_ns)) / @as(f64, num_threads * ops_per_thread) });
    }

    // Test 2: Middle contention
    {
        const num_threads = 8;
        const ops_per_thread = 100000;

        var custom_mutex: Mutex = .{};
        var std_mutex: std.Thread.Mutex = .{};
        var custom_counter: i64 = 0;
        var std_counter: i64 = 0;

        var threads: [num_threads]std.Thread = undefined;
        const start_custom = std.time.nanoTimestamp();
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, struct {
                fn run(m: *Mutex, c: *i64) void {
                    for (0..ops_per_thread) |_| {
                        m.lock();
                        c.* += 1;
                        m.unlock();
                    }
                }
            }.run, .{ &custom_mutex, &custom_counter });
        }
        for (threads) |t| t.join();
        const custom_ns = std.time.nanoTimestamp() - start_custom;

        const start_std = std.time.nanoTimestamp();
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, struct {
                fn run(m: *std.Thread.Mutex, c: *i64) void {
                    for (0..ops_per_thread) |_| {
                        m.lock();
                        c.* += 1;
                        m.unlock();
                    }
                }
            }.run, .{ &std_mutex, &std_counter });
        }
        for (threads) |t| t.join();
        const std_ns = std.time.nanoTimestamp() - start_std;

        std.debug.print("Middle contention ({} threads, {} ops):\n", .{ num_threads, num_threads * ops_per_thread });
        std.debug.print("  Custom: {} ns ({d:.0} ns/op)\n", .{ custom_ns, @as(f64, @floatFromInt(custom_ns)) / @as(f64, num_threads * ops_per_thread) });
        std.debug.print("  Std:    {} ns ({d:.0} ns/op)\n", .{ std_ns, @as(f64, @floatFromInt(std_ns)) / @as(f64, num_threads * ops_per_thread) });
    }

    // Test 3: Low contention
    {
        const num_threads = 2;
        const ops_per_thread = 50000;

        var custom_mutex: Mutex = .{};
        var std_mutex: std.Thread.Mutex = .{};
        var custom_counter: i64 = 0;
        var std_counter: i64 = 0;

        var threads: [num_threads]std.Thread = undefined;
        const start_custom = std.time.nanoTimestamp();
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, struct {
                fn run(m: *Mutex, c: *i64) void {
                    for (0..ops_per_thread) |_| {
                        m.lock();
                        c.* += 1;
                        m.unlock();
                    }
                }
            }.run, .{ &custom_mutex, &custom_counter });
        }
        for (threads) |t| t.join();
        const custom_ns = std.time.nanoTimestamp() - start_custom;

        const start_std = std.time.nanoTimestamp();
        for (&threads) |*t| {
            t.* = try std.Thread.spawn(.{}, struct {
                fn run(m: *std.Thread.Mutex, c: *i64) void {
                    for (0..ops_per_thread) |_| {
                        m.lock();
                        c.* += 1;
                        m.unlock();
                    }
                }
            }.run, .{ &std_mutex, &std_counter });
        }
        for (threads) |t| t.join();
        const std_ns = std.time.nanoTimestamp() - start_std;

        std.debug.print("Low contention ({} threads, {} ops):\n", .{ num_threads, num_threads * ops_per_thread });
        std.debug.print("  Custom: {} ns ({d:.0} ns/op)\n", .{ custom_ns, @as(f64, @floatFromInt(custom_ns)) / @as(f64, num_threads * ops_per_thread) });
        std.debug.print("  Std:    {} ns ({d:.0} ns/op)\n", .{ std_ns, @as(f64, @floatFromInt(std_ns)) / @as(f64, num_threads * ops_per_thread) });
    }

    // Test 4: Uncontended
    {
        const ops = 100000;

        var custom_mutex: Mutex = .{};
        var std_mutex: std.Thread.Mutex = .{};
        var counter: i64 = 0;

        const start_custom = std.time.nanoTimestamp();
        for (0..ops) |_| {
            custom_mutex.lock();
            counter += 1;
            custom_mutex.unlock();
        }
        const custom_ns = std.time.nanoTimestamp() - start_custom;

        const start_std = std.time.nanoTimestamp();
        for (0..ops) |_| {
            std_mutex.lock();
            counter += 1;
            std_mutex.unlock();
        }
        const std_ns = std.time.nanoTimestamp() - start_std;

        std.debug.print("Uncontended (single thread, {} ops):\n", .{ops});
        std.debug.print("  Custom: {} ns ({d:.0} ns/op)\n", .{ custom_ns, @as(f64, @floatFromInt(custom_ns)) / @as(f64, ops) });
        std.debug.print("  Std:    {} ns ({d:.0} ns/op)\n", .{ std_ns, @as(f64, @floatFromInt(std_ns)) / @as(f64, ops) });
    }
}

test "Mutex: multiple waiters" {
    const Counter = struct {
        value: i32 = 0,
        mutex: Mutex = .{},

        fn increment(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            self.value += 1;
            std.Thread.sleep(1_000_000); // 1ms
        }
    };

    var counter = Counter{};
    const num_threads = 10;
    var threads: [num_threads]std.Thread = undefined;

    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, Counter.increment, .{&counter});
    }

    for (threads) |t| t.join();
    try std.testing.expectEqual(counter.value, num_threads);
}

test "Mutex: high contention with mixed trylock and lock" {
    const Counter = struct {
        value: i32 = 0,
        mutex: Mutex = .{},
        fn tryIncrement(self: *@This()) void {
            if (self.mutex.trylock()) {
                defer self.mutex.unlock();
                self.value += 1;
            } else {
                self.mutex.lock();
                defer self.mutex.unlock();
                self.value += 1;
            }
        }
    };
    var counter = Counter{};
    const num_threads = 16;
    const increments_per_thread = 500;
    var threads: [num_threads]std.Thread = undefined;
    for (&threads) |*t| {
        t.* = try std.Thread.spawn(.{}, struct {
            fn run(c: *Counter) !void {
                for (0..increments_per_thread) |_| {
                    c.tryIncrement();
                }
            }
        }.run, .{&counter});
    }
    for (threads) |t| t.join();
    try std.testing.expectEqual(counter.value, @as(i32, num_threads * increments_per_thread));
}
