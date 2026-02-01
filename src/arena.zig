const std = @import("std");
const MemID = @import("mem.zig").MemID;
const Mutex = @import("mutex.zig").Mutex;
const BitPool = @import("bit_pool.zig");
const types = @import("types.zig");
const assert = @import("util.zig").assert;
const utils = @import("util.zig");
const internal_alloc = @import("internal_allocator.zig");
const os_alloc = @import("os_allocator.zig");
const os = @import("os.zig");

const Atomic = std.atomic.Value;
const Allocator = std.mem.Allocator;

pub const MAX_ARENAS: usize = 256;
pub const ARENA_BLOCK_SIZE: usize = types.SEGMENT_SIZE; // 32 MiB
pub const ARENA_MIN_OBJ_SIZE: usize = ARENA_BLOCK_SIZE / 2; // 16 MiB minimum for arena alloc

/// Default purge delay in milliseconds (10 seconds)
pub const PURGE_DELAY_MS: i64 = 10 * 1000;

pub const Arenas = struct {
    const Self = @This();

    arenas: [MAX_ARENAS]Atomic(?*Arena) align(std.atomic.cache_line) =
        [_]Atomic(?*Arena){Atomic(?*Arena).init(null)} ** MAX_ARENAS,
    count: Atomic(usize) align(std.atomic.cache_line) = Atomic(usize).init(0),
    purge_expire: Atomic(i64) align(std.atomic.cache_line) = Atomic(i64).init(0),

    /// Add arena to collection, returns false if full
    pub fn add(self: *Self, arena: *Arena) bool {
        const idx = self.count.fetchAdd(1, .acq_rel);
        if (idx >= MAX_ARENAS) {
            @branchHint(.cold);
            _ = self.count.fetchSub(1, .acq_rel);
            return false;
        }
        arena.id = @intCast(idx + 1);
        self.arenas[idx].store(arena, .release);
        return true;
    }

    /// Get arena by ID (1-based)
    pub fn get(self: *Self, arena_id: u32) ?*Arena {
        if (arena_id == 0) return null;
        const idx = arena_id - 1;
        if (idx >= MAX_ARENAS) {
            @branchHint(.cold);
            return null;
        }
        return self.arenas[idx].load(.acquire);
    }

    /// Get arena by index (0-based)
    pub fn getByIndex(self: *Self, idx: usize) ?*Arena {
        if (idx >= self.getCount()) return null;
        return self.arenas[idx].load(.acquire);
    }

    /// Get current arena count
    pub inline fn getCount(self: *const Self) usize {
        return self.count.load(.acquire);
    }

    /// Try to allocate from any suitable arena
    pub fn tryAlloc(
        self: *Self,
        size: usize,
        _: usize, // alignment - arena blocks are always block-aligned
        commit: bool,
        allow_large: bool,
        req_arena_id: ?u32,
        mem_id: *MemID,
    ) ?[*]u8 {
        const needed_blocks = blockCountForSize(size);
        if (needed_blocks == 0) return null;

        // If specific arena requested, try only that one
        if (req_arena_id) |arena_id| {
            if (self.get(arena_id)) |arena| {
                return arena.tryAlloc(needed_blocks, commit, allow_large, mem_id);
            }
            return null;
        }

        // Try all arenas
        const arena_count = self.getCount();
        for (0..arena_count) |i| {
            if (self.arenas[i].load(.acquire)) |arena| {
                // Skip exclusive arenas unless specifically requested
                if (arena.exclusive) continue;

                // Skip large page arenas if not allowed
                if (arena.is_large and !allow_large) continue;

                if (arena.tryAlloc(needed_blocks, commit, allow_large, mem_id)) |ptr| {
                    return ptr;
                }
            }
        }

        return null;
    }

    /// Try purge across all arenas
    pub fn tryPurge(self: *Self, force: bool) void {
        const now = std.time.milliTimestamp();
        const expire = self.purge_expire.load(.acquire);

        if (!force and (expire == 0 or now < expire)) {
            return;
        }

        const arena_count = self.getCount();
        for (0..arena_count) |i| {
            if (self.arenas[i].load(.acquire)) |arena| {
                arena.tryPurge(force);
            }
        }

        // Reset global expire
        self.purge_expire.store(0, .release);
    }

    /// Update global purge expiration
    pub fn updatePurgeExpire(self: *Self, expire: i64) void {
        var current = self.purge_expire.load(.acquire);
        while (current == 0 or (expire != 0 and expire < current)) {
            const result = self.purge_expire.cmpxchgWeak(
                current,
                expire,
                .acq_rel,
                .acquire,
            );
            if (result) |new_current| {
                current = new_current;
            } else {
                break;
            }
        }
    }
};

// =============================================================================
// Arena
// =============================================================================

pub const Arena = struct {
    const Self = @This();

    // Identity
    id: u32 = 0,
    mem_id: MemID,

    // Memory region
    start: Atomic([*]u8),
    block_count: usize,
    field_count: usize,
    meta_size: usize,

    // Configuration
    numa_node: i32 = -1,
    exclusive: bool = false,
    is_large: bool = false,

    // Synchronization
    abandoned_visit_lock: Mutex = .{},

    // Search optimization
    search_idx: Atomic(usize) = Atomic(usize).init(0),

    // Purge scheduling
    purge_expire: Atomic(i64) = Atomic(i64).init(0),

    // Block tracking bitmaps
    blocks_inuse: BitPool.DynamicBitmap,
    blocks_dirty: ?BitPool.DynamicBitmap = null,
    blocks_committed: ?BitPool.DynamicBitmap = null,
    blocks_purge: ?BitPool.DynamicBitmap = null,
    blocks_abandoned: ?BitPool.DynamicBitmap = null,

    /// Create a new arena
    pub fn create(
        allocator: Allocator,
        start: [*]u8,
        size: usize,
        committed: bool,
        is_large: bool,
        is_zero: bool,
        exclusive: bool,
        numa_node: i32,
    ) !*Self {
        const block_count = size / ARENA_BLOCK_SIZE;
        if (block_count == 0) return error.ArenaTooSmall;

        const field_count = (block_count + BitPool.FIELD_BITS - 1) / BitPool.FIELD_BITS;

        // Allocate arena struct
        const arena = try allocator.create(Self);
        errdefer allocator.destroy(arena);

        // Initialize all fields first with defaults
        arena.* = .{
            .id = 0,
            .mem_id = MemID.create_os(
                start[0..size],
                committed,
                is_zero,
                is_large,
            ),
            .start = Atomic([*]u8).init(start),
            .block_count = block_count,
            .field_count = field_count,
            .meta_size = @sizeOf(Self),
            .numa_node = numa_node,
            .exclusive = exclusive,
            .is_large = is_large,
            .blocks_inuse = undefined,
        };

        // Initialize bitmaps
        arena.blocks_inuse = try BitPool.DynamicBitmap.init(allocator, block_count);
        errdefer arena.blocks_inuse.deinit();

        // Dirty bitmap (for tracking zero-initialized memory)
        if (is_zero) {
            arena.blocks_dirty = try BitPool.DynamicBitmap.init(allocator, block_count);
        }

        // Committed bitmap (for lazy commit)
        if (!committed and !is_large) {
            arena.blocks_committed = try BitPool.DynamicBitmap.init(allocator, block_count);
        }

        // Purge bitmap (for deferred decommit)
        if (!is_large) {
            arena.blocks_purge = try BitPool.DynamicBitmap.init(allocator, block_count);
        }

        // Abandoned bitmap (for multi-threaded segment tracking)
        arena.blocks_abandoned = try BitPool.DynamicBitmap.init(allocator, block_count);

        return arena;
    }

    /// Destroy arena and free resources
    pub fn destroy(self: *Self, allocator: Allocator) void {
        self.blocks_inuse.deinit();
        if (self.blocks_dirty) |*b| b.deinit();
        if (self.blocks_committed) |*b| b.deinit();
        if (self.blocks_purge) |*b| b.deinit();
        if (self.blocks_abandoned) |*b| b.deinit();
        allocator.destroy(self);
    }

    /// Get arena ID (0 if not registered)
    pub inline fn getArenaId(self: *const Self) u32 {
        return self.id;
    }

    /// Get total arena size in bytes
    pub inline fn sizeInBytes(self: *const Self) usize {
        return self.block_count * ARENA_BLOCK_SIZE;
    }

    /// Get arena index (0-based)
    pub inline fn index(self: *const Self) usize {
        return if (self.id > 0) @as(usize, self.id - 1) else MAX_ARENAS;
    }

    /// Get pointer to start of specific block
    pub inline fn blockStart(self: *const Self, block_idx: usize) [*]u8 {
        assert(block_idx < self.block_count);
        return self.start.load(.monotonic) + block_idx * ARENA_BLOCK_SIZE;
    }

    /// Try to allocate blocks from this arena
    pub fn tryAlloc(
        self: *Self,
        needed_blocks: usize,
        commit: bool,
        allow_large: bool,
        mem_id: *MemID,
    ) ?[*]u8 {
        if (needed_blocks == 0) {
            @branchHint(.cold);
            return null;
        }

        // Check large page requirement
        if (self.is_large and !allow_large) {
            return null;
        }

        // Try to claim blocks
        var block_idx: usize = 0;
        if (!self.tryClaim(needed_blocks, &block_idx)) {
            return null;
        }

        // Get pointer to allocated memory
        const ptr = self.blockStart(block_idx);

        // Clear purge scheduling for these blocks
        if (self.blocks_purge) |*purge| {
            _ = purge.unclaimAcrossUnsafe(block_idx, needed_blocks);
        }

        // Track zero-initialized state
        var initially_zero = self.mem_id.flags.initially_zero;
        if (self.blocks_dirty) |*dirty| {
            // If dirty bits were already set, memory may not be zero
            initially_zero = !dirty.claimAcross(block_idx, needed_blocks);
        }

        // Track commit state
        var initially_committed = self.is_large or (self.blocks_committed == null);
        if (!initially_committed and commit) {
            if (self.blocks_committed) |*committed| {
                const stats = committed.isClaimedAcrossStats(block_idx, needed_blocks);
                initially_committed = stats.all_claimed;

                if (!stats.all_claimed) {
                    // Need to commit memory
                    const commit_size = needed_blocks * ARENA_BLOCK_SIZE;
                    if (os.commit(ptr, commit_size)) {
                        // Mark as committed
                        _ = committed.claimAcross(block_idx, needed_blocks);
                        initially_committed = true;
                    } else {
                        // Commit failed - release blocks and return null
                        self.blocks_inuse.unclaim(block_idx, needed_blocks);
                        return null;
                    }
                }
            }
        }

        // Build memory ID
        mem_id.* = MemID.create_arena(
            block_idx,
            self.id,
            self.exclusive,
            initially_committed,
            initially_zero,
            self.is_large,
        );

        return ptr;
    }

    /// Free previously allocated blocks
    pub fn free(
        self: *Self,
        block_idx: usize,
        block_count_to_free: usize,
        was_committed: bool,
    ) void {
        if (block_count_to_free == 0) return;
        assert(block_idx + block_count_to_free <= self.block_count);

        // Update committed tracking if partially committed
        if (self.blocks_committed) |*committed| {
            if (!was_committed) {
                // Mark as uncommitted
                committed.unclaim(block_idx, block_count_to_free);
            }
        }

        // Schedule for purge (deferred decommit)
        self.schedulePurge(block_idx, block_count_to_free);

        // Release blocks (make available for reuse)
        self.blocks_inuse.unclaim(block_idx, block_count_to_free);
    }

    /// Try to claim blocks atomically
    fn tryClaim(self: *Self, needed: usize, out_idx: *usize) bool {
        if (needed == 0) {
            @branchHint(.cold);
            return false;
        }

        const start_idx = self.search_idx.load(.monotonic);
        if (self.blocks_inuse.tryFindFromAndClaim(start_idx, needed)) |found_idx| {
            @branchHint(.likely);
            // Update search hint for next allocation
            self.search_idx.store(found_idx + needed, .monotonic);
            out_idx.* = found_idx;
            return true;
        }

        return false;
    }

    /// Schedule blocks for purge (deferred decommit)
    inline fn schedulePurge(self: *Self, block_idx: usize, count: usize) void {
        if (self.is_large) return; // Can't decommit large pages

        if (self.blocks_purge) |*purge| {
            // Mark blocks for purge
            _ = purge.claimAcross(block_idx, count);

            // Set expiration time
            const now = std.time.milliTimestamp();
            const expire = now + PURGE_DELAY_MS;

            var current_expire = self.purge_expire.load(.acquire);
            while (current_expire == 0 or expire < current_expire) {
                const result = self.purge_expire.cmpxchgWeak(
                    current_expire,
                    expire,
                    .acq_rel,
                    .acquire,
                );
                if (result) |new_current| {
                    current_expire = new_current;
                } else {
                    break;
                }
            }
        }
    }

    /// Try to execute pending purge operations
    pub fn tryPurge(self: *Self, force: bool) void {
        if (self.is_large) return;

        const now = std.time.milliTimestamp();
        const expire = self.purge_expire.load(.acquire);

        if (!force and (expire == 0 or now < expire)) {
            return;
        }

        if (self.blocks_purge) |*purge| {
            // Find ranges to purge
            var block_idx: usize = 0;
            while (block_idx < self.block_count) {
                // Skip blocks not scheduled for purge
                if (!purge.isAnyClaimed(block_idx, 1)) {
                    block_idx += 1;
                    continue;
                }

                // Find contiguous range
                var range_end = block_idx + 1;
                while (range_end < self.block_count and purge.isAnyClaimed(range_end, 1)) {
                    range_end += 1;
                }

                const range_count = range_end - block_idx;

                // Try to claim these blocks temporarily (for thread safety)
                if (self.blocks_inuse.tryClaim(block_idx, range_count)) {
                    // Perform purge/decommit
                    const ptr = self.blockStart(block_idx);
                    const purge_size = range_count * ARENA_BLOCK_SIZE;

                    if (os.decommit(ptr, purge_size)) {
                        // Update committed bitmap
                        if (self.blocks_committed) |*committed| {
                            committed.unclaim(block_idx, range_count);
                        }
                    }

                    // Clear purge bits
                    purge.unclaim(block_idx, range_count);

                    // Release blocks back
                    self.blocks_inuse.unclaim(block_idx, range_count);
                }

                block_idx = range_end;
            }

            // Reset purge expire
            self.purge_expire.store(0, .release);
        }
    }

    /// Mark blocks as abandoned (for segment abandonment)
    pub fn markAbandoned(self: *Self, block_idx: usize, count: usize) void {
        if (self.blocks_abandoned) |*abandoned| {
            _ = abandoned.claimAcross(block_idx, count);
        }
    }

    /// Clear abandoned mark
    pub fn clearAbandoned(self: *Self, block_idx: usize, count: usize) void {
        if (self.blocks_abandoned) |*abandoned| {
            abandoned.unclaim(block_idx, count);
        }
    }

    /// Check if any blocks are abandoned
    pub fn hasAbandoned(self: *const Self) bool {
        if (self.blocks_abandoned) |*abandoned| {
            return !abandoned.isEmpty();
        }
        return false;
    }

    /// Get statistics
    pub fn getStats(self: *const Self) ArenaStats {
        return .{
            .total_blocks = self.block_count,
            .used_blocks = self.blocks_inuse.countClaimed(),
            .committed_blocks = if (self.blocks_committed) |*c| c.countClaimed() else self.block_count,
            .dirty_blocks = if (self.blocks_dirty) |*d| d.countClaimed() else 0,
            .purge_pending = if (self.blocks_purge) |*p| p.countClaimed() else 0,
            .abandoned_blocks = if (self.blocks_abandoned) |*a| a.countClaimed() else 0,
        };
    }
};

pub const ArenaStats = struct {
    total_blocks: usize,
    used_blocks: usize,
    committed_blocks: usize,
    dirty_blocks: usize,
    purge_pending: usize,
    abandoned_blocks: usize,

    pub fn totalBytes(self: ArenaStats) usize {
        return self.total_blocks * ARENA_BLOCK_SIZE;
    }

    pub fn usedBytes(self: ArenaStats) usize {
        return self.used_blocks * ARENA_BLOCK_SIZE;
    }

    pub fn committedBytes(self: ArenaStats) usize {
        return self.committed_blocks * ARENA_BLOCK_SIZE;
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Calculate number of blocks needed for given size
pub inline fn blockCountForSize(size: usize) usize {
    return utils.divCeil(size, ARENA_BLOCK_SIZE);
}

/// Calculate size in bytes for given block count
pub inline fn sizeForBlockCount(blocks: usize) usize {
    return blocks * ARENA_BLOCK_SIZE;
}

/// Check if size is suitable for arena allocation
pub inline fn isSuitableForArena(size: usize) bool {
    return size >= ARENA_MIN_OBJ_SIZE;
}

var global_arenas: Arenas = .{};

pub fn globalArenas() *Arenas {
    return &global_arenas;
}

/// Reserve a new arena with given size
pub fn reserve(
    size: usize,
    commit: bool,
    allow_large: bool,
    exclusive: bool,
    numa_node: i32,
) !*Arena {
    const arena_size = utils.alignUp(size, ARENA_BLOCK_SIZE);
    if (arena_size == 0) return error.InvalidSize;

    // Check arena limit
    if (global_arenas.getCount() >= MAX_ARENAS) {
        return error.TooManyArenas;
    }

    // Allocate arena memory from OS
    var os_allocator = os_alloc.OsAllocator{ .config = os.mem_config_static };
    const alignment = std.mem.Alignment.fromByteUnits(ARENA_BLOCK_SIZE);
    const mem_ptr = os_allocator.map(arena_size, alignment) orelse {
        return error.OutOfMemory;
    };

    errdefer os_allocator.unmap(@alignCast(mem_ptr[0..arena_size]));

    // Create arena structure
    const allocator = internal_alloc.global();
    const arena = Arena.create(
        allocator,
        mem_ptr,
        arena_size,
        commit,
        allow_large and os.supportsLargePages(),
        true, // initially zero from OS
        exclusive,
        numa_node,
    ) catch |err| {
        return err;
    };

    // Register arena
    if (!global_arenas.add(arena)) {
        arena.destroy(allocator);
        return error.TooManyArenas;
    }

    return arena;
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "Arena: block count calculation" {
    try testing.expectEqual(@as(usize, 1), blockCountForSize(1));
    try testing.expectEqual(@as(usize, 1), blockCountForSize(ARENA_BLOCK_SIZE));
    try testing.expectEqual(@as(usize, 2), blockCountForSize(ARENA_BLOCK_SIZE + 1));
    try testing.expectEqual(@as(usize, 2), blockCountForSize(ARENA_BLOCK_SIZE * 2));
}

test "Arena: size suitability" {
    try testing.expect(!isSuitableForArena(0));
    try testing.expect(!isSuitableForArena(ARENA_MIN_OBJ_SIZE - 1));
    try testing.expect(isSuitableForArena(ARENA_MIN_OBJ_SIZE));
    try testing.expect(isSuitableForArena(ARENA_MIN_OBJ_SIZE + 1));
}

test "Arenas: add and get" {
    var arenas = Arenas{};

    // Create mock arena (just for testing collection)
    var mock_inuse = try BitPool.DynamicBitmap.init(testing.allocator, 64);
    defer mock_inuse.deinit();

    var arena = Arena{
        .mem_id = MemID.none(),
        .start = Atomic([*]u8).init(@ptrFromInt(0x1000)),
        .block_count = 64,
        .field_count = 1,
        .meta_size = @sizeOf(Arena),
        .blocks_inuse = mock_inuse,
    };

    try testing.expect(arenas.add(&arena));
    try testing.expectEqual(@as(u32, 1), arena.id);
    try testing.expectEqual(@as(usize, 1), arenas.getCount());

    const retrieved = arenas.get(1);
    try testing.expect(retrieved != null);
    try testing.expectEqual(&arena, retrieved.?);

    // Invalid IDs
    try testing.expect(arenas.get(0) == null);
    try testing.expect(arenas.get(2) == null);
}

test "Arena: tryClaim basic" {
    var inuse = try BitPool.DynamicBitmap.init(testing.allocator, 64);
    defer inuse.deinit();

    var arena = Arena{
        .mem_id = MemID.none(),
        .start = Atomic([*]u8).init(@ptrFromInt(0x10000)),
        .block_count = 64,
        .field_count = 1,
        .meta_size = @sizeOf(Arena),
        .blocks_inuse = inuse,
    };

    var idx: usize = 0;

    // Claim 4 blocks
    try testing.expect(arena.tryClaim(4, &idx));
    try testing.expectEqual(@as(usize, 0), idx);

    // Claim another 4 blocks
    try testing.expect(arena.tryClaim(4, &idx));
    try testing.expectEqual(@as(usize, 4), idx);

    // Release first blocks
    arena.blocks_inuse.unclaim(0, 4);

    // Should find the freed space
    try testing.expect(arena.tryClaim(4, &idx));
    try testing.expectEqual(@as(usize, 8), idx); // Search continues from last position
}

test "Arena: blockStart" {
    var inuse = try BitPool.DynamicBitmap.init(testing.allocator, 4);
    defer inuse.deinit();

    const base: [*]u8 = @ptrFromInt(0x10000000);
    var arena = Arena{
        .mem_id = MemID.none(),
        .start = Atomic([*]u8).init(base),
        .block_count = 4,
        .field_count = 1,
        .meta_size = @sizeOf(Arena),
        .blocks_inuse = inuse,
    };

    try testing.expectEqual(base, arena.blockStart(0));
    try testing.expectEqual(base + ARENA_BLOCK_SIZE, arena.blockStart(1));
    try testing.expectEqual(base + 2 * ARENA_BLOCK_SIZE, arena.blockStart(2));
}

test "ArenaStats: byte calculations" {
    const stats = ArenaStats{
        .total_blocks = 10,
        .used_blocks = 5,
        .committed_blocks = 8,
        .dirty_blocks = 3,
        .purge_pending = 2,
        .abandoned_blocks = 0,
    };

    try testing.expectEqual(@as(usize, 10 * ARENA_BLOCK_SIZE), stats.totalBytes());
    try testing.expectEqual(@as(usize, 5 * ARENA_BLOCK_SIZE), stats.usedBytes());
    try testing.expectEqual(@as(usize, 8 * ARENA_BLOCK_SIZE), stats.committedBytes());
}
