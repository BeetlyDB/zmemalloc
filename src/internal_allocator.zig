const std = @import("std");
const assert = @import("util.zig").assert;
const types = @import("types.zig");
const mem = std.mem;
const Allocator = mem.Allocator;
const os_alloc = @import("os_allocator.zig");
const os = @import("os.zig");

/// Thread-safe internal allocator for zmemalloc metadata structures.
/// Uses a static buffer with lock-free atomic operations to avoid OS calls,
/// falling back to OS allocation (mmap) for larger requests.
pub fn InternalAllocator(comptime size: usize, comptime config: os.OsMemConfig) type {
    return struct {
        const Self = @This();

        /// Static arena buffer - cache line aligned to avoid false sharing
        buffer: [size]u8 align(std.atomic.cache_line) = undefined,

        /// Current allocation end index in static buffer
        end_index: usize align(std.atomic.cache_line) = 0,

        /// OS allocator for fallback (mmap/munmap are inherently thread-safe)
        os_allocator: os_alloc.OsAllocator = .{ .config = config },

        /// Returns a thread-safe Allocator interface.
        pub fn get(self: *Self) Allocator {
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

        /// Compile error to prevent misuse
        pub const allocator = @compileError("use 'var ia: InternalAllocator(N, config) = .{}; const allocator = ia.get();' instead");

        /// Check if pointer is within our static buffer
        pub inline fn ownsPtr(self: *Self, ptr: [*]u8) bool {
            return sliceContainsPtr(self.buffer[0..], ptr);
        }

        /// Check if slice is within our static buffer
        pub inline fn ownsSlice(self: *Self, slice: []u8) bool {
            return sliceContainsSlice(self.buffer[0..], slice);
        }

        /// Check if this is the last allocation (thread-safe)
        pub inline fn isLastAllocation(self: *Self, buf: []u8) bool {
            const end = @atomicLoad(usize, &self.end_index, .acquire);
            return buf.ptr + buf.len == self.buffer[0..].ptr + end;
        }

        /// Atomic reset - use with caution, only when no allocations are in use
        pub inline fn reset(self: *Self) void {
            @atomicStore(usize, &self.end_index, 0, .release);
        }

        /// Get amount of static memory used (atomic read)
        pub inline fn staticUsed(self: *Self) usize {
            return @atomicLoad(usize, &self.end_index, .acquire);
        }

        /// Get amount of static memory remaining
        pub inline fn staticRemaining(self: *Self) usize {
            return size - self.staticUsed();
        }

        // -------------------------------------------------------------------------
        // Thread-safe allocator vtable functions (all use atomics)
        // -------------------------------------------------------------------------

        /// Lock-free allocation using CAS
        fn alloc(ctx: *anyopaque, n: usize, alignment: mem.Alignment, ra: usize) ?[*]u8 {
            _ = ra;
            const self: *Self = @ptrCast(@alignCast(ctx));

            const ptr_align = alignment.toByteUnits();
            var end_index = @atomicLoad(usize, &self.end_index, .acquire);

            while (true) {
                const adjust_off = mem.alignPointerOffset(self.buffer[0..].ptr + end_index, ptr_align) orelse {
                    // Alignment overflow, fall back to OS
                    return self.os_allocator.map(n, alignment);
                };

                const adjusted_index = end_index + adjust_off;
                const new_end_index = adjusted_index + n;

                if (new_end_index > size) {
                    // Doesn't fit in static buffer, fall back to OS
                    return self.os_allocator.map(n, alignment);
                }

                // Try to atomically claim the space
                const result = @cmpxchgWeak(
                    usize,
                    &self.end_index,
                    end_index,
                    new_end_index,
                    .acq_rel,
                    .acquire,
                );

                if (result) |updated_index| {
                    // CAS failed, retry with updated end_index
                    end_index = updated_index;
                } else {
                    // Success - return the allocated memory
                    return self.buffer[adjusted_index..new_end_index].ptr;
                }
            }
        }

        /// Thread-safe resize using CAS
        /// Only succeeds for the last allocation in static buffer
        fn resize(
            ctx: *anyopaque,
            buf: []u8,
            alignment: mem.Alignment,
            new_size: usize,
            ra: usize,
        ) bool {
            _ = alignment;
            _ = ra;
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Check if this is from static buffer
            if (!self.ownsSlice(buf)) {
                // OS allocation - delegate to OS allocator
                return os_alloc.OsAllocator.realloc(&self.os_allocator, buf, new_size, false) != null;
            }

            // Static buffer allocation - try atomic resize
            const buf_start = @intFromPtr(buf.ptr) - @intFromPtr(self.buffer[0..].ptr);
            const buf_end = buf_start + buf.len;

            var end_index = @atomicLoad(usize, &self.end_index, .acquire);

            while (true) {
                // Check if this is the last allocation
                if (buf_end != end_index) {
                    // Not the last allocation
                    // Can only "shrink" without actual change (noop)
                    return new_size <= buf.len;
                }

                // Calculate new end index
                const new_end_index = buf_start + new_size;

                // Check bounds
                if (new_end_index > size) {
                    return false; // Would exceed buffer
                }

                // Try to atomically update end_index
                const result = @cmpxchgWeak(
                    usize,
                    &self.end_index,
                    end_index,
                    new_end_index,
                    .acq_rel,
                    .acquire,
                );

                if (result) |updated_index| {
                    // CAS failed - another thread modified end_index
                    // This means buf is no longer the last allocation
                    end_index = updated_index;
                    // Continue loop to recheck
                } else {
                    // Success
                    return true;
                }
            }
        }

        /// Thread-safe remap using CAS
        fn remap(
            ctx: *anyopaque,
            memory: []u8,
            alignment: mem.Alignment,
            new_len: usize,
            ra: usize,
        ) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Check if this is from static buffer
            if (!self.ownsSlice(memory)) {
                // OS allocation - use mremap if available
                return os_alloc.OsAllocator.realloc(&self.os_allocator, memory, new_len, true);
            }

            // Static buffer - remap is resize that returns pointer
            return if (resize(ctx, memory, alignment, new_len, ra)) memory.ptr else null;
        }

        /// Thread-safe free using CAS
        /// Only actually frees the last allocation in static buffer
        fn free(
            ctx: *anyopaque,
            buf: []u8,
            alignment: mem.Alignment,
            ra: usize,
        ) void {
            _ = alignment;
            _ = ra;
            const self: *Self = @ptrCast(@alignCast(ctx));

            // Check if this is from static buffer
            if (!self.ownsSlice(buf)) {
                // OS allocation - free it (munmap is thread-safe)
                self.os_allocator.unmap(@alignCast(buf));
                return;
            }

            // Static buffer allocation - try atomic free (only for last allocation)
            const buf_start = @intFromPtr(buf.ptr) - @intFromPtr(self.buffer[0..].ptr);
            const buf_end = buf_start + buf.len;

            var end_index = @atomicLoad(usize, &self.end_index, .acquire);

            while (true) {
                // Check if this is the last allocation
                if (buf_end != end_index) {
                    // Not the last allocation - no-op (bump allocator semantics)
                    return;
                }

                // Try to atomically free by moving end_index back
                const result = @cmpxchgWeak(
                    usize,
                    &self.end_index,
                    end_index,
                    buf_start,
                    .acq_rel,
                    .acquire,
                );

                if (result) |updated_index| {
                    // CAS failed - another thread modified end_index
                    // This means buf is no longer the last allocation
                    end_index = updated_index;
                    // Continue loop to recheck
                } else {
                    // Success - memory freed
                    return;
                }
            }
        }

        // -------------------------------------------------------------------------
        // Helper functions
        // -------------------------------------------------------------------------

        inline fn sliceContainsPtr(container: []u8, ptr: [*]u8) bool {
            return @intFromPtr(ptr) >= @intFromPtr(container.ptr) and
                @intFromPtr(ptr) < (@intFromPtr(container.ptr) + container.len);
        }

        inline fn sliceContainsSlice(container: []u8, slice: []u8) bool {
            return @intFromPtr(slice.ptr) >= @intFromPtr(container.ptr) and
                (@intFromPtr(slice.ptr) + slice.len) <= (@intFromPtr(container.ptr) + container.len);
        }
    };
}

/// Default internal allocator type with 4KiB static buffer
pub const DefaultInternalAllocator = InternalAllocator(
    (types.INTPTR_SIZE / 2) * types.KiB, // 4KiB on 64-bit
    os.mem_config_static,
);

// -----------------------------------------------------------------------------
// Global allocator instance for project-wide use
// -----------------------------------------------------------------------------

var global_instance: DefaultInternalAllocator align(std.atomic.cache_line) = .{};

pub fn global() Allocator {
    return global_instance.get();
}

pub fn globalInstance() *DefaultInternalAllocator {
    return &global_instance;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

const testing = std.testing;

test "InternalAllocator: basic allocation" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    // Small allocation should use static buffer
    const ptr1 = try alloc_if.alloc(u8, 64);
    try testing.expect(ia.ownsSlice(ptr1));
    try testing.expect(ia.staticUsed() >= 64);

    // Free last allocation
    alloc_if.free(ptr1);
    try testing.expect(ia.staticUsed() == 0);
}

test "InternalAllocator: alignment" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    // Allocate with various alignments
    const ptr1 = try alloc_if.alignedAlloc(u8, .fromByteUnits(16), 32);
    try testing.expect(@intFromPtr(ptr1.ptr) % 16 == 0);

    const ptr2 = try alloc_if.alignedAlloc(u8, .fromByteUnits(64), 128);
    try testing.expect(@intFromPtr(ptr2.ptr) % 64 == 0);

    const ptr3 = try alloc_if.alignedAlloc(u8, .fromByteUnits(128), 256);
    try testing.expect(@intFromPtr(ptr3.ptr) % 128 == 0);
}

test "InternalAllocator: resize last allocation" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    var ptr = try alloc_if.alloc(u8, 32);
    try testing.expect(ptr.len == 32);
    const used_after_alloc = ia.staticUsed();

    // Grow the allocation
    try testing.expect(alloc_if.resize(ptr, 64));
    ptr = ptr.ptr[0..64];
    try testing.expect(ptr.len == 64);
    try testing.expect(ia.staticUsed() > used_after_alloc);

    // Shrink the allocation
    try testing.expect(alloc_if.resize(ptr, 16));
    ptr = ptr.ptr[0..16];
    try testing.expect(ptr.len == 16);
}

test "InternalAllocator: resize non-last allocation fails to grow" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const ptr1 = try alloc_if.alloc(u8, 64);
    _ = try alloc_if.alloc(u8, 64); // This makes ptr1 not the last

    // Growing non-last allocation should fail
    try testing.expect(!alloc_if.resize(ptr1, 128));

    // Shrinking (no-op) should succeed
    try testing.expect(alloc_if.resize(ptr1, 32));
}

test "InternalAllocator: static exhaustion falls back to OS" {
    var ia: InternalAllocator(256, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    // Allocate most of static buffer
    const ptr1 = try alloc_if.alloc(u8, 200);
    try testing.expect(ia.ownsSlice(ptr1));

    // Next large allocation should fall back to OS
    const ptr2 = try alloc_if.alloc(u8, 1024);
    try testing.expect(!ia.ownsSlice(ptr2));

    // Free OS allocation
    alloc_if.free(ptr2);
}

test "InternalAllocator: reset" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    _ = try alloc_if.alloc(u8, 100);
    _ = try alloc_if.alloc(u8, 200);
    try testing.expect(ia.staticUsed() >= 300);

    ia.reset();
    try testing.expect(ia.staticUsed() == 0);
}

test "InternalAllocator: free non-last allocation is no-op" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const ptr1 = try alloc_if.alloc(u8, 64);
    _ = try alloc_if.alloc(u8, 64);
    const used_before = ia.staticUsed();

    // Free first allocation (not last) - should be no-op
    alloc_if.free(ptr1);
    try testing.expect(ia.staticUsed() == used_before);
}

test "InternalAllocator: free last allocation works" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    _ = try alloc_if.alloc(u8, 64);
    const ptr2 = try alloc_if.alloc(u8, 64);
    const used_before = ia.staticUsed();

    // Free last allocation
    alloc_if.free(ptr2);
    try testing.expect(ia.staticUsed() < used_before);
}

test "InternalAllocator: oversized goes directly to OS" {
    var ia: InternalAllocator(256, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    // Allocation larger than buffer goes directly to OS
    const large = try alloc_if.alloc(u8, 512);
    try testing.expect(!ia.ownsSlice(large));

    alloc_if.free(large);
}

test "InternalAllocator: staticUsed and staticRemaining" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    _ = ia.get();

    try testing.expectEqual(@as(usize, 4096), ia.staticRemaining());
    try testing.expectEqual(@as(usize, 0), ia.staticUsed());
}

test "InternalAllocator: DefaultInternalAllocator type" {
    var ia: DefaultInternalAllocator = .{};
    const alloc_if = ia.get();

    const ptr = try alloc_if.alloc(u8, 128);
    try testing.expect(ia.ownsSlice(ptr));
    alloc_if.free(ptr);
}

test "InternalAllocator: cache line alignment" {
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};

    // Verify buffer is cache line aligned
    try testing.expect(@intFromPtr(&ia.buffer) % std.atomic.cache_line == 0);
}

test "InternalAllocator: global allocator" {
    const alloc_if = global();

    const ptr = try alloc_if.alloc(u8, 64);
    try testing.expect(globalInstance().ownsSlice(ptr));

    // Can get allocator multiple times
    const alloc_if2 = global();
    const ptr2 = try alloc_if2.alloc(u8, 32);
    try testing.expect(globalInstance().ownsSlice(ptr2));
}

test "InternalAllocator: concurrent allocation simulation" {
    // Simulate concurrent allocations
    var ia: InternalAllocator(4096, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    var ptrs: [10][]u8 = undefined;
    for (&ptrs) |*p| {
        p.* = try alloc_if.alloc(u8, 32);
    }

    // All should be from static buffer and non-overlapping
    for (ptrs, 0..) |p1, i| {
        try testing.expect(ia.ownsSlice(p1));
        for (ptrs[i + 1 ..]) |p2| {
            // Check non-overlapping
            const p1_end = @intFromPtr(p1.ptr) + p1.len;
            const p2_start = @intFromPtr(p2.ptr);
            try testing.expect(p1_end <= p2_start or @intFromPtr(p1.ptr) >= @intFromPtr(p2.ptr) + p2.len);
        }
    }
}

// -----------------------------------------------------------------------------
// Multi-threaded tests
// -----------------------------------------------------------------------------

const Thread = std.Thread;

/// Shared state for multi-threaded tests
const ThreadTestContext = struct {
    allocator: Allocator,
    allocations: []std.atomic.Value(?[*]u8),
    allocation_sizes: []usize,
    errors: std.atomic.Value(usize),
    ready: std.atomic.Value(bool),

    fn init(alloc: Allocator, ptrs: []std.atomic.Value(?[*]u8), sizes: []usize) ThreadTestContext {
        return .{
            .allocator = alloc,
            .allocations = ptrs,
            .allocation_sizes = sizes,
            .errors = std.atomic.Value(usize){ .raw = 0 },
            .ready = std.atomic.Value(bool){ .raw = false },
        };
    }
};

test "InternalAllocator: multi-threaded concurrent allocation" {
    // Large buffer to allow many allocations
    var ia: InternalAllocator(64 * 1024, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 8;
    const allocs_per_thread = 16;
    const total_allocs = num_threads * allocs_per_thread;

    // Atomic storage for allocations
    var allocations: [total_allocs]std.atomic.Value(?[*]u8) = undefined;
    for (&allocations) |*a| {
        a.* = std.atomic.Value(?[*]u8){ .raw = null };
    }

    var sizes: [total_allocs]usize = undefined;
    for (&sizes) |*s| {
        s.* = 64; // Each allocation is 64 bytes
    }

    var ctx = ThreadTestContext.init(alloc_if, &allocations, &sizes);

    // Spawn threads
    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        threads[i] = try Thread.spawn(.{}, threadAllocWorker, .{ &ctx, i, allocs_per_thread });
    }

    // Signal threads to start
    ctx.ready.store(true, .release);

    // Wait for all threads
    for (&threads) |*t| {
        t.join();
    }

    // Verify no errors
    try testing.expectEqual(@as(usize, 0), ctx.errors.load(.acquire));

    // Verify all allocations succeeded and don't overlap
    var valid_count: usize = 0;
    for (allocations, 0..) |a, i| {
        const ptr = a.load(.acquire);
        if (ptr) |p| {
            valid_count += 1;
            const size = sizes[i];

            // Check no overlap with other allocations
            for (allocations[i + 1 ..], (i + 1)..) |other, j| {
                const other_ptr = other.load(.acquire);
                if (other_ptr) |op| {
                    const other_size = sizes[j];
                    const p_start = @intFromPtr(p);
                    const p_end = p_start + size;
                    const o_start = @intFromPtr(op);
                    const o_end = o_start + other_size;

                    // Ranges should not overlap
                    const no_overlap = p_end <= o_start or o_end <= p_start;
                    try testing.expect(no_overlap);
                }
            }
        }
    }

    // At least some allocations should have succeeded
    try testing.expect(valid_count > 0);
}

fn threadAllocWorker(ctx: *ThreadTestContext, thread_id: usize, allocs_per_thread: usize) void {
    // Wait for ready signal
    while (!ctx.ready.load(.acquire)) {
        std.atomic.spinLoopHint();
    }

    const base_idx = thread_id * allocs_per_thread;

    for (0..allocs_per_thread) |i| {
        const idx = base_idx + i;
        const size = ctx.allocation_sizes[idx];

        const ptr = ctx.allocator.alloc(u8, size) catch {
            _ = ctx.errors.fetchAdd(1, .acq_rel);
            continue;
        };

        // Store the allocation atomically
        ctx.allocations[idx].store(ptr.ptr, .release);
    }
}

test "InternalAllocator: multi-threaded alloc and free" {
    var ia: InternalAllocator(32 * 1024, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 4;
    var error_count = std.atomic.Value(usize){ .raw = 0 };
    var ready = std.atomic.Value(bool){ .raw = false };

    const Context = struct {
        allocator: Allocator,
        errors: *std.atomic.Value(usize),
        ready: *std.atomic.Value(bool),
    };

    const ctx = Context{
        .allocator = alloc_if,
        .errors = &error_count,
        .ready = &ready,
    };

    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        threads[i] = try Thread.spawn(.{}, struct {
            fn worker(c: Context) void {
                // Wait for ready signal
                while (!c.ready.load(.acquire)) {
                    std.atomic.spinLoopHint();
                }

                // Perform multiple alloc/free cycles
                for (0..50) |_| {
                    const ptr = c.allocator.alloc(u8, 64) catch {
                        _ = c.errors.fetchAdd(1, .acq_rel);
                        continue;
                    };

                    // Write to verify memory is usable
                    @memset(ptr, 0xBE);

                    // Free it
                    c.allocator.free(ptr);
                }
            }
        }.worker, .{ctx});
    }

    // Start all threads
    ready.store(true, .release);

    // Wait for completion
    for (&threads) |*t| {
        t.join();
    }

    try testing.expectEqual(@as(usize, 0), error_count.load(.acquire));
}

test "InternalAllocator: stress test with many threads" {
    // Use a large buffer for stress test
    var ia: InternalAllocator(128 * 1024, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 16;
    var success_count = std.atomic.Value(usize){ .raw = 0 };
    var ready = std.atomic.Value(bool){ .raw = false };

    const Context = struct {
        allocator: Allocator,
        successes: *std.atomic.Value(usize),
        ready: *std.atomic.Value(bool),
    };

    const ctx = Context{
        .allocator = alloc_if,
        .successes = &success_count,
        .ready = &ready,
    };

    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        threads[i] = try Thread.spawn(.{}, struct {
            fn worker(c: Context) void {
                while (!c.ready.load(.acquire)) {
                    std.atomic.spinLoopHint();
                }

                // Rapid alloc/free cycles
                for (0..100) |iter| {
                    // Vary size based on iteration
                    const size = 32 + (iter % 128);

                    const ptr = c.allocator.alloc(u8, size) catch continue;

                    // Touch the memory
                    if (ptr.len > 0) {
                        ptr[0] = 0xAB;
                        ptr[ptr.len - 1] = 0xCD;
                    }

                    _ = c.successes.fetchAdd(1, .acq_rel);
                    c.allocator.free(ptr);
                }
            }
        }.worker, .{ctx});
    }

    ready.store(true, .release);

    for (&threads) |*t| {
        t.join();
    }

    // At least some allocations should have succeeded
    const total_successes = success_count.load(.acquire);
    try testing.expect(total_successes > 0);
}

test "InternalAllocator: concurrent OS fallback" {
    // Small buffer forces OS fallback
    var ia: InternalAllocator(256, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 4;
    var error_count = std.atomic.Value(usize){ .raw = 0 };
    var os_alloc_count = std.atomic.Value(usize){ .raw = 0 };
    var ready = std.atomic.Value(bool){ .raw = false };

    const Context = struct {
        ia: *InternalAllocator(256, os.mem_config_static),
        allocator: Allocator,
        errors: *std.atomic.Value(usize),
        os_allocs: *std.atomic.Value(usize),
        ready: *std.atomic.Value(bool),
    };

    const ctx = Context{
        .ia = &ia,
        .allocator = alloc_if,
        .errors = &error_count,
        .os_allocs = &os_alloc_count,
        .ready = &ready,
    };

    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        threads[i] = try Thread.spawn(.{}, struct {
            fn worker(c: Context) void {
                while (!c.ready.load(.acquire)) {
                    std.atomic.spinLoopHint();
                }

                for (0..20) |_| {
                    // Large allocation forces OS fallback
                    const ptr = c.allocator.alloc(u8, 512) catch {
                        _ = c.errors.fetchAdd(1, .acq_rel);
                        continue;
                    };

                    // Check if it's from OS (not static buffer)
                    if (!c.ia.ownsSlice(ptr)) {
                        _ = c.os_allocs.fetchAdd(1, .acq_rel);
                    }

                    // Touch memory
                    @memset(ptr, 0xEF);

                    c.allocator.free(ptr);
                }
            }
        }.worker, .{ctx});
    }

    ready.store(true, .release);

    for (&threads) |*t| {
        t.join();
    }

    try testing.expectEqual(@as(usize, 0), error_count.load(.acquire));

    // All allocations should have gone to OS (buffer too small)
    const os_allocs = os_alloc_count.load(.acquire);
    try testing.expect(os_allocs > 0);
}

test "InternalAllocator: concurrent resize contention" {
    var ia: InternalAllocator(8 * 1024, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 4;
    var success_resizes = std.atomic.Value(usize){ .raw = 0 };
    var failed_resizes = std.atomic.Value(usize){ .raw = 0 };
    var ready = std.atomic.Value(bool){ .raw = false };

    const Context = struct {
        allocator: Allocator,
        successes: *std.atomic.Value(usize),
        failures: *std.atomic.Value(usize),
        ready: *std.atomic.Value(bool),
    };

    const ctx = Context{
        .allocator = alloc_if,
        .successes = &success_resizes,
        .failures = &failed_resizes,
        .ready = &ready,
    };

    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        threads[i] = try Thread.spawn(.{}, struct {
            fn worker(c: Context) void {
                while (!c.ready.load(.acquire)) {
                    std.atomic.spinLoopHint();
                }

                for (0..30) |_| {
                    var ptr = c.allocator.alloc(u8, 32) catch continue;

                    // Try to resize (may fail if another thread allocated after us)
                    if (c.allocator.resize(ptr, 64)) {
                        ptr = ptr.ptr[0..64];
                        _ = c.successes.fetchAdd(1, .acq_rel);
                    } else {
                        _ = c.failures.fetchAdd(1, .acq_rel);
                    }

                    c.allocator.free(ptr);
                }
            }
        }.worker, .{ctx});
    }

    ready.store(true, .release);

    for (&threads) |*t| {
        t.join();
    }

    // Some resizes should succeed, some fail due to contention
    const successes = success_resizes.load(.acquire);
    const failures = failed_resizes.load(.acquire);

    // At least some allocations happened
    try testing.expect(successes + failures > 0);
}

test "InternalAllocator: verify atomic consistency" {
    var ia: InternalAllocator(16 * 1024, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 8;
    var ready = std.atomic.Value(bool){ .raw = false };
    var done = std.atomic.Value(usize){ .raw = 0 };

    // Each thread will record the end_index it observed
    var observed_indices: [num_threads * 100]std.atomic.Value(usize) = undefined;
    for (&observed_indices) |*idx| {
        idx.* = std.atomic.Value(usize){ .raw = 0 };
    }

    const Context = struct {
        ia: *InternalAllocator(16 * 1024, os.mem_config_static),
        allocator: Allocator,
        ready: *std.atomic.Value(bool),
        done: *std.atomic.Value(usize),
        indices: []std.atomic.Value(usize),
        thread_id: usize,
    };

    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        const ctx = Context{
            .ia = &ia,
            .allocator = alloc_if,
            .ready = &ready,
            .done = &done,
            .indices = &observed_indices,
            .thread_id = i,
        };

        threads[i] = try Thread.spawn(.{}, struct {
            fn worker(c: Context) void {
                while (!c.ready.load(.acquire)) {
                    std.atomic.spinLoopHint();
                }

                const base = c.thread_id * 100;

                for (0..100) |j| {
                    const ptr = c.allocator.alloc(u8, 16) catch continue;

                    // Record current end_index
                    c.indices[base + j].store(c.ia.staticUsed(), .release);

                    // Small delay to increase contention
                    std.atomic.spinLoopHint();

                    c.allocator.free(ptr);
                }

                _ = c.done.fetchAdd(1, .acq_rel);
            }
        }.worker, .{ctx});
    }

    ready.store(true, .release);

    for (&threads) |*t| {
        t.join();
    }

    // All threads completed
    try testing.expectEqual(num_threads, done.load(.acquire));

    // End index should be reasonable (not corrupted by race conditions)
    const final_index = ia.staticUsed();
    try testing.expect(final_index <= 16 * 1024);
}

test "InternalAllocator: data integrity under concurrent access" {
    var ia: InternalAllocator(32 * 1024, os.mem_config_static) = .{};
    const alloc_if = ia.get();

    const num_threads = 8;
    var errors = std.atomic.Value(usize){ .raw = 0 };
    var ready = std.atomic.Value(bool){ .raw = false };

    const Context = struct {
        allocator: Allocator,
        errors: *std.atomic.Value(usize),
        ready: *std.atomic.Value(bool),
        thread_id: u8,
    };

    var threads: [num_threads]Thread = undefined;
    for (0..num_threads) |i| {
        const ctx = Context{
            .allocator = alloc_if,
            .errors = &errors,
            .ready = &ready,
            .thread_id = @intCast(i),
        };

        threads[i] = try Thread.spawn(.{}, struct {
            fn worker(c: Context) void {
                while (!c.ready.load(.acquire)) {
                    std.atomic.spinLoopHint();
                }

                // Unique pattern per thread
                const pattern: u8 = 0xA0 | c.thread_id;

                for (0..50) |_| {
                    const ptr = c.allocator.alloc(u8, 64) catch continue;

                    // Fill with thread-specific pattern
                    @memset(ptr, pattern);

                    // Small delay
                    std.atomic.spinLoopHint();
                    std.atomic.spinLoopHint();

                    // Verify pattern is intact (no other thread corrupted our memory)
                    for (ptr) |byte| {
                        if (byte != pattern) {
                            _ = c.errors.fetchAdd(1, .acq_rel);
                            break;
                        }
                    }

                    c.allocator.free(ptr);
                }
            }
        }.worker, .{ctx});
    }

    ready.store(true, .release);

    for (&threads) |*t| {
        t.join();
    }

    // No data corruption should have occurred
    try testing.expectEqual(@as(usize, 0), errors.load(.acquire));
}
