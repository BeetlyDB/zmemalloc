const Mutex = @import("mutex.zig").Mutex;
const std = @import("std");
const Semaphore = @import("semaphore.zig");

pub const DefaultRwLock = struct {
    state: usize = 0,
    mutex: Mutex = .{},
    semaphore: Semaphore = .{},

    const IS_WRITING: usize = 1;
    const WRITER: usize = 1 << 1;
    const READER: usize = 1 << (1 + @bitSizeOf(Count));
    const WRITER_MASK: usize = std.math.maxInt(Count) << @ctz(WRITER);
    const READER_MASK: usize = std.math.maxInt(Count) << @ctz(READER);
    const Count = std.meta.Int(.unsigned, @divFloor(@bitSizeOf(usize) - 1, 2));

    pub fn tryLock(rwl: *DefaultRwLock) bool {
        if (rwl.mutex.trylock()) {
            const state = @atomicLoad(usize, &rwl.state, .seq_cst);
            if (state & READER_MASK == 0) {
                _ = @atomicRmw(usize, &rwl.state, .Or, IS_WRITING, .seq_cst);
                return true;
            }

            rwl.mutex.unlock();
        }

        return false;
    }

    pub fn lock(rwl: *DefaultRwLock) void {
        _ = @atomicRmw(usize, &rwl.state, .Add, WRITER, .seq_cst);
        rwl.mutex.lock();

        const state = @atomicRmw(usize, &rwl.state, .Add, IS_WRITING -% WRITER, .seq_cst);
        if (state & READER_MASK != 0)
            rwl.semaphore.wait();
    }

    pub fn unlock(rwl: *DefaultRwLock) void {
        _ = @atomicRmw(usize, &rwl.state, .And, ~IS_WRITING, .seq_cst);
        rwl.mutex.unlock();
    }

    pub fn tryLockShared(rwl: *DefaultRwLock) bool {
        const state = @atomicLoad(usize, &rwl.state, .seq_cst);
        if (state & (IS_WRITING | WRITER_MASK) == 0) {
            _ = @cmpxchgStrong(
                usize,
                &rwl.state,
                state,
                state + READER,
                .seq_cst,
                .seq_cst,
            ) orelse return true;
        }

        if (rwl.mutex.trylock()) {
            _ = @atomicRmw(usize, &rwl.state, .Add, READER, .seq_cst);
            rwl.mutex.unlock();
            return true;
        }

        return false;
    }

    pub fn lockShared(rwl: *DefaultRwLock) void {
        var state = @atomicLoad(usize, &rwl.state, .seq_cst);
        while (state & (IS_WRITING | WRITER_MASK) == 0) {
            state = @cmpxchgWeak(
                usize,
                &rwl.state,
                state,
                state + READER,
                .seq_cst,
                .seq_cst,
            ) orelse return;
        }

        rwl.mutex.lock();
        _ = @atomicRmw(usize, &rwl.state, .Add, READER, .seq_cst);
        rwl.mutex.unlock();
    }

    pub fn unlockShared(rwl: *DefaultRwLock) void {
        const state = @atomicRmw(usize, &rwl.state, .Sub, READER, .seq_cst);

        if ((state & READER_MASK == READER) and (state & IS_WRITING != 0))
            rwl.semaphore.post();
    }
};

test "DefaultRwLock - internal state" {
    var rwl = DefaultRwLock{};
    const testing = std.testing;

    // The following failed prior to the fix for Issue #13163,
    // where the WRITER flag was subtracted by the lock method.

    rwl.lock();
    rwl.unlock();
    try testing.expectEqual(rwl, DefaultRwLock{});
}
