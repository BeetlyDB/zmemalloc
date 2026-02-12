//! # Atomic Bitmap Pool
//!
//! Lock-free bitmap data structures for concurrent bit manipulation.
//! Used throughout the allocator for tracking allocation state.
//!
//! ## Bitmap Types
//!
//! | Type          | Bits | Storage | Use Case                        |
//! |---------------|------|---------|----------------------------------|
//! | SmallBitmap   | 64   | Stack   | Single-word tracking             |
//! | MediumBitmap  | 256  | Stack   | Medium-sized pools               |
//! | LargeBitmap   | 1024 | Stack   | Larger fixed pools               |
//! | SliceBitmap   | 512  | Stack   | Segment slice tracking           |
//! | DynamicBitmap | N    | Heap    | Runtime-sized (arenas)           |
//!
//! ## Operations
//!
//! - `tryClaim(idx, count)`: Atomically set bits if all are clear
//! - `unclaim(idx, count)`: Atomically clear bits
//! - `isClaimed(idx, count)`: Check if all bits are set
//! - `tryFindAndClaim(count)`: Find and claim first available range
//!
//! ## Atomicity
//!
//! All operations use atomic compare-and-swap for thread safety.
//! Multi-field operations roll back on failure to maintain consistency.
//!
//! ## Memory Layout
//!
//! ```
//! Field 0:  [b0 b1 b2 ... b63]
//! Field 1:  [b64 b65 ... b127]
//! ...
//! ```

const std = @import("std");
const assert = @import("util.zig").assert;
const Atomic = std.atomic.Value;

/// A range of indices within a bitset.
const Range = struct {
    /// The index of the first bit of interest.
    start: usize,
    /// The index immediately after the last bit of interest.
    end: usize,
};

pub const ClaimedStats = struct {
    all_claimed: bool,
    already_claimed_count: usize,
};

/// Number of bits per field
pub const FIELD_BITS: usize = @bitSizeOf(usize);

/// Maximum bits for stack-allocated bitmap (8KB worth of bits)
pub const MAX_STACK_BITMAP_BITS: usize = 1024 * 8;

/// Atomic bitmap for concurrent bit manipulation.
/// Storage type is selected via optional comptime parameter:
/// - is_stack = true (default): stack-allocated fixed array, size known at comptime
/// - is_stack = false: heap-allocated slice, size determined at runtime in init()
///
/// Example usage:
///   // Stack allocated (comptime size)
///   const SmallBitmap = AtomicBitmap(512, true);  // or just AtomicBitmap(512)
///   var small = SmallBitmap{};
///
///   // Heap allocated (runtime size)
///   const DynamicBitmap = AtomicBitmap(0, false);
///   var dynamic = try DynamicBitmap.init(allocator, runtime_size);
///   defer dynamic.deinit();
pub fn AtomicBitmap(comptime num_bits: usize, comptime is_stack: bool) type {
    // For stack allocation, compute field count at comptime
    // For heap allocation, field count is determined at runtime
    const comptime_field_count = if (is_stack) (num_bits + FIELD_BITS - 1) / FIELD_BITS else 0;

    // Storage type: fixed array for stack, slice for heap
    const FieldsType = if (is_stack)
        [comptime_field_count]Atomic(usize)
    else
        []Atomic(usize);

    // Allocator type: void for stack (not needed), real allocator for heap
    const AllocatorType = if (is_stack) void else std.mem.Allocator;

    return struct {
        const Self = @This();

        /// Comptime bit count (only meaningful for stack-allocated)
        pub const comptime_bit_count: usize = num_bits;
        /// Comptime field count (only meaningful for stack-allocated)
        pub const comptime_fields_count: usize = comptime_field_count;
        /// Whether this bitmap is stack-allocated
        pub const is_stack_allocated: bool = is_stack;

        fields: FieldsType = if (is_stack)
            [_]Atomic(usize){Atomic(usize).init(0)} ** comptime_field_count
        else
            undefined,
        allocator: AllocatorType = if (is_stack) {} else undefined,
        /// Runtime bit count (for heap-allocated, this is set in init; for stack, equals comptime_bit_count)
        bit_count: usize = if (is_stack) num_bits else 0,

        /// Initialize bitmap
        /// - Stack version: no parameters needed
        /// - Heap version: requires allocator and runtime size
        pub const init = if (is_stack) initStack else initHeap;

        fn initStack() Self {
            return .{};
        }

        fn initHeap(allocator: std.mem.Allocator, size: usize) !Self {
            const field_count_runtime = (size + FIELD_BITS - 1) / FIELD_BITS;
            const fields = try allocator.alloc(Atomic(usize), field_count_runtime);
            for (fields) |*field| {
                field.* = Atomic(usize).init(0);
            }
            return .{
                .fields = fields,
                .allocator = allocator,
                .bit_count = size,
            };
        }

        /// Free bitmap memory (only does something for heap-allocated)
        pub inline fn deinit(self: *Self) void {
            if (!is_stack) {
                self.allocator.free(self.fields);
                self.fields = &.{};
                self.bit_count = 0;
            }
        }

        /// No Thread safe!
        pub fn unclaimAcrossUnsafe(
            self: *Self,
            global_start_bit: usize,
            count: usize,
        ) bool {
            if (count == 0) return true;

            var was_all_claimed = true;
            var pos = global_start_bit;
            var remain = count;

            while (remain > 0) {
                const fi = pos / FIELD_BITS;
                const bo = pos % FIELD_BITS;
                const take = @min(remain, FIELD_BITS - bo);

                if (fi >= self.fields.len) {
                    was_all_claimed = false;
                    break;
                }

                const mask = makeMask(bo, take);

                const prev = self.fields[fi].raw;
                self.fields[fi].raw &= ~mask;

                if ((prev & mask) != mask) {
                    was_all_claimed = false;
                }

                pos += take;
                remain -= take;
            }

            return was_all_claimed;
        }

        pub fn tryFindFromAndClaimAcross(
            self: *Self,
            search_start_field: *Atomic(usize),
            needed: usize,
        ) ?usize {
            if (needed == 0) return 0;
            if (needed > self.bit_count) return null;

            const max_fields = self.fields.len;
            if (max_fields == 0) return null;

            var field_idx = search_start_field.load(.monotonic);

            var wrapped = false;
            while (true) {
                if (field_idx >= max_fields) {
                    if (wrapped) break;
                    field_idx = 0;
                    wrapped = true;
                    continue;
                }

                const field_val = self.fields[field_idx].load(.acquire);
                const free = ~field_val;

                if (free != 0) {
                    var bit = @ctz(free);
                    while (bit < FIELD_BITS) {
                        const global_bit = field_idx * FIELD_BITS + bit;

                        if (global_bit + needed > self.bit_count) break;
                        // try to claim
                        if (self.tryClaim(global_bit, needed)) {
                            search_start_field.store(field_idx, .monotonic);
                            return global_bit;
                        }

                        const next_bit = @ctz(~(free >> @intCast(bit + 1))) + bit + 1;
                        if (next_bit >= FIELD_BITS) break;
                        bit = next_bit;
                    }
                }

                field_idx += 1;
                if (field_idx >= max_fields and !wrapped) {
                    field_idx = 0;
                    wrapped = true;
                }
            }

            return null;
        }

        /// Check, if all bits in dp is set +count how much already is set
        pub fn isClaimedAcrossStats(
            self: *const Self,
            global_start: usize,
            count: usize,
        ) ClaimedStats {
            if (count == 0) return .{ .all_claimed = true, .already_claimed_count = 0 };

            var all = true;
            var already = @as(usize, 0);

            var pos = global_start;
            var remain = count;

            while (remain > 0) {
                const fi = pos / FIELD_BITS;
                const bo = pos % FIELD_BITS;
                const take = @min(remain, FIELD_BITS - bo);

                if (fi >= self.fields.len) {
                    all = false;
                    break;
                }

                const mask = makeMask(bo, take);

                const val = self.fields[fi].load(.acquire);
                const claimed_here = @popCount(val & mask);

                already += claimed_here;
                if (claimed_here != take) all = false;

                pos += take;
                remain -= take;
            }

            return .{ .all_claimed = all, .already_claimed_count = already };
        }

        /// Initialize all bits to zero (stack-only convenience)
        pub fn initEmpty() Self {
            if (is_stack) {
                return .{};
            } else {
                @compileError("Heap-allocated AtomicBitmap requires init(allocator, size)");
            }
        }

        /// Initialize all bits to one (stack-only convenience)
        pub fn initFull() Self {
            if (is_stack) {
                var self = Self{};
                for (0..comptime_field_count) |i| {
                    self.fields[i].store(~@as(usize, 0), .release);
                }
                return self;
            } else {
                @compileError("Heap-allocated AtomicBitmap requires init(allocator, size)");
            }
        }

        /// Get number of fields
        pub inline fn fieldCount(self: *const Self) usize {
            return self.fields.len;
        }

        /// Get total number of bits
        pub inline fn bitCount(self: *const Self) usize {
            return self.bit_count;
        }

        /// Try to atomically claim a sequence of bits (set them to 1)
        /// Returns true if successful, false if any bit was already set
        pub fn tryClaim(self: *Self, bit_idx: usize, count: usize) bool {
            if (count == 0) return true;
            if (bit_idx + count > self.bit_count) return false;

            const fld_idx = bit_idx / FIELD_BITS;
            const bit_offset = bit_idx % FIELD_BITS;

            // Single field case (most common)
            if (bit_offset + count <= FIELD_BITS) {
                @branchHint(.likely);
                return self.tryClaimField(fld_idx, bit_offset, count);
            }

            // Multi-field case - need to claim across fields
            return self.tryClaimAcross(bit_idx, count);
        }

        /// Try to claim bits within a single field atomically
        inline fn tryClaimField(self: *Self, fld_idx: usize, bit_offset: usize, count: usize) bool {
            if (fld_idx >= self.fields.len) return false;

            const mask = makeMask(bit_offset, count);
            const field = &self.fields[fld_idx];

            // Atomic compare-and-swap loop
            var current = field.load(.acquire);
            while (true) {
                // Check if any bits are already set
                if ((current & mask) != 0) {
                    return false; // Some bits already claimed
                }

                // Try to set the bits
                const result = field.cmpxchgWeak(
                    current,
                    current | mask,
                    .acq_rel,
                    .acquire,
                );

                if (result) |new_current| {
                    current = new_current; // CAS failed, retry
                } else {
                    return true; // Success
                }
            }
        }

        /// Try to claim bits across multiple fields
        fn tryClaimAcross(self: *Self, bit_idx: usize, count: usize) bool {
            var idx = bit_idx;
            var remaining = count;
            var claimed_bits: usize = 0;

            // Try to claim each field
            while (remaining > 0) {
                const fld_idx = idx / FIELD_BITS;
                const bit_offset = idx % FIELD_BITS;
                const bits_in_field = @min(remaining, FIELD_BITS - bit_offset);

                if (!self.tryClaimField(fld_idx, bit_offset, bits_in_field)) {
                    // Failed - rollback previously claimed bits
                    if (claimed_bits > 0) {
                        self.unclaim(bit_idx, claimed_bits);
                    }
                    return false;
                }

                claimed_bits += bits_in_field;
                idx += bits_in_field;
                remaining -= bits_in_field;
            }

            return true;
        }

        pub fn claimAcross(self: *Self, global_start_bit: usize, count: usize) bool {
            if (count == 0) return false;

            var any_was_set = false;
            var pos = global_start_bit;
            var remain = count;

            while (remain > 0) {
                const fi = pos / FIELD_BITS;
                const bo = pos % FIELD_BITS;
                const take = @min(remain, FIELD_BITS - bo);

                if (fi >= self.fields.len) break;

                const mask = makeMask(bo, take);

                const prev = self.fields[fi].fetchOr(mask, .acq_rel);
                if ((prev & mask) != 0) {
                    any_was_set = true;
                }

                pos += take;
                remain -= take;
            }

            return any_was_set;
        }

        pub fn claimAcrossUnsafe(self: *Self, global_start_bit: usize, count: usize) bool {
            if (count == 0) return false;

            var any_was_set = false;
            var pos = global_start_bit;
            var remain = count;

            while (remain > 0) {
                const fi = pos / FIELD_BITS;
                const bo = pos % FIELD_BITS;
                const take = @min(remain, FIELD_BITS - bo);

                if (fi >= self.fields.len) break;

                const mask = makeMask(bo, take);

                const prev = self.fields[fi].raw;
                self.fields[fi].raw &= ~mask;
                if ((prev & mask) != 0) {
                    any_was_set = true;
                }

                pos += take;
                remain -= take;
            }

            return any_was_set;
        }

        /// Atomically unclaim (clear) a sequence of bits
        pub inline fn unclaim(
            self: *Self,
            bit_idx: usize,
            count: usize,
        ) void {
            if (count == 0) return;

            var idx = bit_idx;
            var remaining = count;

            while (remaining > 0) {
                const fld_idx = idx / FIELD_BITS;
                const bit_offset = idx % FIELD_BITS;
                const bits_in_field = @min(remaining, FIELD_BITS - bit_offset);

                if (fld_idx < self.fields.len) {
                    const mask = makeMask(bit_offset, bits_in_field);
                    _ = self.fields[fld_idx].fetchAnd(~mask, .acq_rel);
                }

                idx += bits_in_field;
                remaining -= bits_in_field;
            }
        }

        /// Atomically set bits (unconditionally) and return whether any were previously set
        pub fn claim(self: *Self, bit_idx: usize, count: usize) bool {
            if (count == 0) return false;

            var any_set = false;
            var idx = bit_idx;
            var remaining = count;

            while (remaining > 0) {
                const fld_idx = idx / FIELD_BITS;
                const bit_offset = idx % FIELD_BITS;
                const bits_in_field = @min(remaining, FIELD_BITS - bit_offset);

                if (fld_idx < self.fields.len) {
                    const mask = makeMask(bit_offset, bits_in_field);
                    const prev = self.fields[fld_idx].fetchOr(mask, .acq_rel);
                    if ((prev & mask) != 0) {
                        any_set = true;
                    }
                }

                idx += bits_in_field;
                remaining -= bits_in_field;
            }

            return any_set;
        }

        /// Check if all bits in sequence are claimed (set to 1)
        pub fn isClaimed(self: *const Self, bit_idx: usize, count: usize) bool {
            if (count == 0) return true;

            var idx = bit_idx;
            var remaining = count;

            while (remaining > 0) {
                const fld_idx = idx / FIELD_BITS;
                const bit_offset = idx % FIELD_BITS;
                const bits_in_field = @min(remaining, FIELD_BITS - bit_offset);

                if (fld_idx >= self.fields.len) return false;

                const mask = makeMask(bit_offset, bits_in_field);
                const value = self.fields[fld_idx].load(.acquire);
                if ((value & mask) != mask) {
                    return false;
                }

                idx += bits_in_field;
                remaining -= bits_in_field;
            }

            return true;
        }

        /// Check if any bits in sequence are claimed
        pub fn isAnyClaimed(self: *const Self, bit_idx: usize, count: usize) bool {
            if (count == 0) return false;

            var idx = bit_idx;
            var remaining = count;

            while (remaining > 0) {
                const fld_idx = idx / FIELD_BITS;
                const bit_offset = idx % FIELD_BITS;
                const bits_in_field = @min(remaining, FIELD_BITS - bit_offset);

                if (fld_idx < self.fields.len) {
                    const mask = makeMask(bit_offset, bits_in_field);
                    const value = self.fields[fld_idx].load(.acquire);
                    if ((value & mask) != 0) {
                        return true;
                    }
                }

                idx += bits_in_field;
                remaining -= bits_in_field;
            }

            return false;
        }

        /// Count total number of claimed (set) bits
        pub fn countClaimed(self: *const Self) usize {
            var total: usize = 0;
            for (0..self.fields.len) |i| {
                total += @popCount(self.fields[i].load(.acquire));
            }
            // Mask out bits beyond bit_count in the last field
            const remainder = self.bit_count % FIELD_BITS;
            if (remainder != 0 and self.fields.len > 0) {
                const last_field = self.fields[self.fields.len - 1].load(.acquire);
                const extra_bits = @popCount(last_field >> @intCast(remainder));
                total -= extra_bits;
            }
            return total;
        }

        /// Count total number of unclaimed (clear) bits
        pub fn countUnclaimed(self: *const Self) usize {
            return self.bit_count - self.countClaimed();
        }

        /// Find first sequence of `count` unclaimed bits and claim them
        /// Returns the bit index if found, null otherwise
        pub fn tryFindAndClaim(self: *Self, count: usize) ?usize {
            if (count == 0) return 0;
            if (count > self.bit_count) return null;

            for (0..self.fields.len) |fld_idx| {
                const field_value = self.fields[fld_idx].load(.acquire);
                const inverted = ~field_value;

                if (inverted == 0) continue; // Field is full

                // Find first zero bit
                const first_zero = @ctz(inverted);
                if (first_zero >= FIELD_BITS) continue;

                const bit_idx = fld_idx * FIELD_BITS + first_zero;
                if (bit_idx + count > self.bit_count) return null; // Not enough bits left

                // Try to claim starting from this position
                if (self.tryClaim(bit_idx, count)) {
                    return bit_idx;
                }
            }

            return null;
        }

        /// Find first sequence of `count` unclaimed bits starting from `start_idx`
        /// Wraps around to search the entire bitmap
        pub fn tryFindFromAndClaim(self: *Self, start_idx: usize, count: usize) ?usize {
            if (count == 0) return start_idx;
            if (count > self.bit_count) return null;
            if (self.bit_count == 0) return null;

            var search_idx = start_idx % self.bit_count;
            var searched: usize = 0;

            while (searched < self.bit_count) {
                const fld_idx = search_idx / FIELD_BITS;
                if (fld_idx >= self.fields.len) {
                    search_idx = 0;
                    continue;
                }

                const field_value = self.fields[fld_idx].load(.acquire);
                const bit_offset = search_idx % FIELD_BITS;

                // Mask out bits before our search position
                const search_mask = if (bit_offset == 0)
                    ~@as(usize, 0)
                else
                    ~@as(usize, 0) << @intCast(bit_offset);

                const inverted = (~field_value) & search_mask;

                if (inverted != 0) {
                    const first_zero = @ctz(inverted);
                    const bit_idx = fld_idx * FIELD_BITS + first_zero;

                    if (bit_idx + count <= self.bit_count and self.tryClaim(bit_idx, count)) {
                        return bit_idx;
                    }
                }

                // Move to next field
                search_idx = (fld_idx + 1) * FIELD_BITS;
                if (search_idx >= self.bit_count) {
                    search_idx = 0;
                }
                searched += FIELD_BITS - bit_offset;
            }

            return null;
        }

        /// Clear all bits (reset to empty)
        pub fn clear(self: *Self) void {
            for (0..self.fields.len) |i| {
                self.fields[i].store(0, .release);
            }
        }

        /// Set all bits (mark everything as claimed)
        pub fn setAll(self: *Self) void {
            for (0..self.fields.len) |i| {
                self.fields[i].store(~@as(usize, 0), .release);
            }
        }

        /// Check if bitmap is completely empty
        pub fn isEmpty(self: *const Self) bool {
            for (0..self.fields.len) |i| {
                if (self.fields[i].load(.acquire) != 0) return false;
            }
            return true;
        }

        /// Check if bitmap is completely full
        pub fn isFull(self: *const Self) bool {
            if (self.fields.len == 0) return true;
            // Check all full fields
            const full_fields = self.bit_count / FIELD_BITS;
            for (0..full_fields) |i| {
                if (self.fields[i].load(.acquire) != ~@as(usize, 0)) return false;
            }
            // Check remaining bits in last partial field
            const remainder = self.bit_count % FIELD_BITS;
            if (remainder != 0) {
                const mask = (@as(usize, 1) << @intCast(remainder)) - 1;
                if ((self.fields[full_fields].load(.acquire) & mask) != mask) return false;
            }
            return true;
        }

        /// Create a bitmask for `count` bits starting at `offset`
        inline fn makeMask(offset: usize, count: usize) usize {
            if (count >= FIELD_BITS) return ~@as(usize, 0) << @intCast(offset);
            const mask = (@as(usize, 1) << @intCast(count)) - 1;
            return mask << @intCast(offset);
        }
    };
}

/// SmallBitmap: Single-word atomic bitmap (64 bits on 64-bit systems)
/// Good for tracking small fixed-size pools
pub const SmallBitmap = AtomicBitmap(FIELD_BITS, true);

/// MediumBitmap: 256 bits for medium-sized pools
pub const MediumBitmap = AtomicBitmap(256, true);

/// LargeBitmap: 1024 bits for larger pools
pub const LargeBitmap = AtomicBitmap(1024, true);

/// SliceBitmap: 512 bits - matches segment slice count in mimalloc
pub const SliceBitmap = AtomicBitmap(512, true);

/// DynamicBitmap: heap-allocated bitmap with runtime size (for arena)
pub const DynamicBitmap = AtomicBitmap(0, false);

/// Iterator options for BitSet iteration
pub const IteratorOptions = struct {
    kind: Kind = .set,
    direction: Direction = .forward,

    pub const Kind = enum { set, unset };
    pub const Direction = enum { forward, reverse };
};

/// A bit set with static size, which is backed by a single integer.
/// This set is good for sets with a small size, but may generate
/// inefficient code for larger sets, especially in debug mode.
pub fn BitSet(comptime size: u16) type {
    return packed struct {
        const Self = @This();

        // TODO: Make this a comptime field once those are fixed
        /// The number of items in this bit set
        pub const bit_length: usize = size;

        /// The integer type used to represent a mask in this bit set
        pub const MaskInt = std.meta.Int(.unsigned, size);

        /// The integer type used to shift a mask in this bit set
        pub const ShiftInt = std.math.Log2Int(MaskInt);

        /// The bit mask, as a single integer
        mask: MaskInt,

        /// Creates a bit set with no elements present.
        pub fn initEmpty() Self {
            return .{ .mask = 0 };
        }

        /// Creates a bit set with all elements present.
        pub fn initFull() Self {
            return .{ .mask = ~@as(MaskInt, 0) };
        }

        /// Returns the number of bits in this bit set
        pub inline fn capacity(self: Self) usize {
            _ = self;
            return bit_length;
        }

        /// Find first unset (free) bit, return its index
        /// Returns null if all bits are set (pool is full)
        ///
        /// Invariants:
        ///   - Pre: self.bits is valid u64
        ///   - Post: returned index is < 64, or null if all bits set
        ///   - Does not modify state
        pub fn first_unset(bit_set: Self) ?usize {
            const result = @ctz(~bit_set.mask);
            return if (result < bit_set.capacity()) result else null;
        }

        /// Returns true if the bit at the specified index
        /// is present in the set, false otherwise.
        pub fn isSet(self: Self, index: usize) bool {
            assert(index < bit_length);
            return (self.mask & maskBit(index)) != 0;
        }

        /// Returns the total number of set bits in this bit set.
        pub fn count(self: Self) usize {
            return @popCount(self.mask);
        }

        /// Changes the value of the specified bit of the bit
        /// set to match the passed boolean.
        pub fn setValue(self: *Self, index: usize, value: bool) void {
            assert(index < bit_length);
            if (MaskInt == u0) return;
            const bit = maskBit(index);
            const new_bit = bit & std.math.boolMask(MaskInt, value);
            self.mask = (self.mask & ~bit) | new_bit;
        }

        /// Adds a specific bit to the bit set
        pub fn set(self: *Self, index: usize) void {
            assert(index < bit_length);
            self.mask |= maskBit(index);
        }

        /// Changes the value of all bits in the specified range to
        /// match the passed boolean.
        pub fn setRangeValue(self: *Self, range: Range, value: bool) void {
            assert(range.end <= bit_length);
            assert(range.start <= range.end);
            if (range.start == range.end) return;
            if (MaskInt == u0) return;

            const start_bit = @as(ShiftInt, @intCast(range.start));

            var mask = std.math.boolMask(MaskInt, true) << start_bit;
            if (range.end != bit_length) {
                const end_bit = @as(ShiftInt, @intCast(range.end));
                mask &= std.math.boolMask(MaskInt, true) >> @as(ShiftInt, @truncate(@as(usize, @bitSizeOf(MaskInt)) - @as(usize, end_bit)));
            }
            self.mask &= ~mask;

            mask = std.math.boolMask(MaskInt, value) << start_bit;
            if (range.end != bit_length) {
                const end_bit = @as(ShiftInt, @intCast(range.end));
                mask &= std.math.boolMask(MaskInt, value) >> @as(ShiftInt, @truncate(@as(usize, @bitSizeOf(MaskInt)) - @as(usize, end_bit)));
            }
            self.mask |= mask;
        }

        /// Removes a specific bit from the bit set
        pub fn unset(self: *Self, index: usize) void {
            assert(index < bit_length);
            // Workaround for #7953
            if (MaskInt == u0) return;
            self.mask &= ~maskBit(index);
        }

        /// Flips a specific bit in the bit set
        pub fn toggle(self: *Self, index: usize) void {
            assert(index < bit_length);
            self.mask ^= maskBit(index);
        }

        /// Flips all bits in this bit set which are present
        /// in the toggles bit set.
        pub fn toggleSet(self: *Self, toggles: Self) void {
            self.mask ^= toggles.mask;
        }

        /// Flips every bit in the bit set.
        pub fn toggleAll(self: *Self) void {
            self.mask = ~self.mask;
        }

        /// Performs a union of two bit sets, and stores the
        /// result in the first one.  Bits in the result are
        /// set if the corresponding bits were set in either input.
        pub fn setUnion(self: *Self, other: Self) void {
            self.mask |= other.mask;
        }

        /// Performs an intersection of two bit sets, and stores
        /// the result in the first one.  Bits in the result are
        /// set if the corresponding bits were set in both inputs.
        pub fn setIntersection(self: *Self, other: Self) void {
            self.mask &= other.mask;
        }

        /// Finds the index of the first set bit.
        /// If no bits are set, returns null.
        pub fn findFirstSet(self: Self) ?usize {
            const mask = self.mask;
            if (mask == 0) return null;
            return @ctz(mask);
        }

        /// Finds the index of the last set bit.
        /// If no bits are set, returns null.
        pub fn findLastSet(self: Self) ?usize {
            const mask = self.mask;
            if (mask == 0) return null;
            return bit_length - @clz(mask) - 1;
        }

        /// Finds the index of the first set bit, and unsets it.
        /// If no bits are set, returns null.
        pub fn toggleFirstSet(self: *Self) ?usize {
            const mask = self.mask;
            if (mask == 0) return null;
            const index = @ctz(mask);
            self.mask = mask & (mask - 1);
            return index;
        }

        /// Returns true iff every corresponding bit in both
        /// bit sets are the same.
        pub fn eql(self: Self, other: Self) bool {
            return bit_length == 0 or self.mask == other.mask;
        }

        /// Returns true iff the first bit set is the subset
        /// of the second one.
        pub fn subsetOf(self: Self, other: Self) bool {
            return self.intersectWith(other).eql(self);
        }

        /// Returns true iff the first bit set is the superset
        /// of the second one.
        pub fn supersetOf(self: Self, other: Self) bool {
            return other.subsetOf(self);
        }

        /// Returns the complement bit sets. Bits in the result
        /// are set if the corresponding bits were not set.
        pub fn complement(self: Self) Self {
            var result = self;
            result.toggleAll();
            return result;
        }

        /// Returns the union of two bit sets. Bits in the
        /// result are set if the corresponding bits were set
        /// in either input.
        pub fn unionWith(self: Self, other: Self) Self {
            var result = self;
            result.setUnion(other);
            return result;
        }

        /// Returns the intersection of two bit sets. Bits in
        /// the result are set if the corresponding bits were
        /// set in both inputs.
        pub fn intersectWith(self: Self, other: Self) Self {
            var result = self;
            result.setIntersection(other);
            return result;
        }

        /// Returns the xor of two bit sets. Bits in the
        /// result are set if the corresponding bits were
        /// not the same in both inputs.
        pub fn xorWith(self: Self, other: Self) Self {
            var result = self;
            result.toggleSet(other);
            return result;
        }

        /// Returns the difference of two bit sets. Bits in
        /// the result are set if set in the first but not
        /// set in the second set.
        pub fn differenceWith(self: Self, other: Self) Self {
            var result = self;
            result.setIntersection(other.complement());
            return result;
        }

        /// Iterates through the items in the set, according to the options.
        /// The default options (.{}) will iterate indices of set bits in
        /// ascending order.  Modifications to the underlying bit set may
        /// or may not be observed by the iterator.
        pub fn iterator(self: *const Self, comptime options: IteratorOptions) Iterator(options) {
            return .{
                .bits_remain = switch (options.kind) {
                    .set => self.mask,
                    .unset => ~self.mask,
                },
            };
        }

        pub fn Iterator(comptime options: IteratorOptions) type {
            return SingleWordIterator(options.direction);
        }

        fn SingleWordIterator(comptime direction: IteratorOptions.Direction) type {
            return struct {
                const IterSelf = @This();
                // all bits which have not yet been iterated over
                bits_remain: MaskInt,

                /// Returns the index of the next unvisited set bit
                /// in the bit set, in ascending order.
                pub fn next(self: *IterSelf) ?usize {
                    if (self.bits_remain == 0) return null;

                    switch (direction) {
                        .forward => {
                            const next_index = @ctz(self.bits_remain);
                            self.bits_remain &= self.bits_remain - 1;
                            return next_index;
                        },
                        .reverse => {
                            const leading_zeroes = @clz(self.bits_remain);
                            const top_bit = (@bitSizeOf(MaskInt) - 1) - leading_zeroes;
                            self.bits_remain &= (@as(MaskInt, 1) << @as(ShiftInt, @intCast(top_bit))) - 1;
                            return top_bit;
                        },
                    }
                }
            };
        }

        inline fn maskBit(index: usize) MaskInt {
            if (MaskInt == u0) return 0;
            return @as(MaskInt, 1) << @as(ShiftInt, @intCast(index));
        }
        inline fn boolMaskBit(index: usize, value: bool) MaskInt {
            if (MaskInt == u0) return 0;
            return @as(MaskInt, @intFromBool(value)) << @as(ShiftInt, @intCast(index));
        }
    };
}

// =============================================================================
//  Tests
// =============================================================================

test "AtomicBitmap: basic claim/unclaim" {
    const testing = std.testing;
    const Bitmap128 = AtomicBitmap(128, true);

    var bitmap = Bitmap128{};

    // Claim single bit
    try testing.expect(bitmap.tryClaim(0, 1));
    try testing.expect(bitmap.isClaimed(0, 1));
    try testing.expect(!bitmap.tryClaim(0, 1)); // Already claimed

    // Unclaim
    bitmap.unclaim(0, 1);
    try testing.expect(!bitmap.isClaimed(0, 1));

    // Can claim again
    try testing.expect(bitmap.tryClaim(0, 1));
}

test "AtomicBitmap: claim sequence" {
    const testing = std.testing;
    const Bitmap128 = AtomicBitmap(128, true);

    var bitmap = Bitmap128{};

    // Claim 4 bits
    try testing.expect(bitmap.tryClaim(0, 4));
    try testing.expect(bitmap.isClaimed(0, 4));

    // Can't claim overlapping
    try testing.expect(!bitmap.tryClaim(2, 4));

    // Can claim non-overlapping
    try testing.expect(bitmap.tryClaim(4, 4));
    try testing.expect(bitmap.isClaimed(4, 4));
}

test "AtomicBitmap: cross-field claim" {
    const testing = std.testing;
    const Bitmap128 = AtomicBitmap(128, true);

    var bitmap = Bitmap128{};

    // Claim bits spanning two fields (at boundary)
    const boundary = FIELD_BITS - 2;
    try testing.expect(bitmap.tryClaim(boundary, 4)); // 2 bits in field 0, 2 in field 1

    try testing.expect(bitmap.isClaimed(boundary, 4));
    try testing.expect(bitmap.isAnyClaimed(boundary, 1));
    try testing.expect(bitmap.isAnyClaimed(FIELD_BITS, 1));
}

test "AtomicBitmap: tryFindAndClaim" {
    const testing = std.testing;
    const Bitmap128 = AtomicBitmap(128, true);

    var bitmap = Bitmap128{};

    // Find and claim 4 bits
    const idx1 = bitmap.tryFindAndClaim(4);
    try testing.expect(idx1 != null);
    try testing.expectEqual(@as(usize, 0), idx1.?);

    // Find and claim another 4 bits
    const idx2 = bitmap.tryFindAndClaim(4);
    try testing.expect(idx2 != null);
    try testing.expectEqual(@as(usize, 4), idx2.?);

    // Free first allocation
    bitmap.unclaim(0, 4);

    // Should find the freed space
    const idx3 = bitmap.tryFindAndClaim(4);
    try testing.expect(idx3 != null);
    try testing.expectEqual(@as(usize, 0), idx3.?);
}

test "AtomicBitmap: isAnyClaimed" {
    const testing = std.testing;
    const Bitmap64 = AtomicBitmap(64, true);

    var bitmap = Bitmap64{};

    try testing.expect(!bitmap.isAnyClaimed(0, 8));

    _ = bitmap.tryClaim(3, 1);

    try testing.expect(bitmap.isAnyClaimed(0, 8));
    try testing.expect(bitmap.isAnyClaimed(3, 1));
    try testing.expect(!bitmap.isAnyClaimed(0, 3));
    try testing.expect(!bitmap.isAnyClaimed(4, 4));
}

test "AtomicBitmap: claim returns previous state" {
    const testing = std.testing;
    const Bitmap64 = AtomicBitmap(64, true);

    var bitmap = Bitmap64{};

    // First claim - nothing was set
    const was_set1 = bitmap.claim(0, 4);
    try testing.expect(!was_set1);

    // Second claim on same bits - should report they were set
    const was_set2 = bitmap.claim(0, 4);
    try testing.expect(was_set2);
}

test "AtomicBitmap: countClaimed and countUnclaimed" {
    const testing = std.testing;
    const Bitmap128 = AtomicBitmap(128, true);

    var bitmap = Bitmap128{};

    try testing.expectEqual(@as(usize, 0), bitmap.countClaimed());
    try testing.expectEqual(@as(usize, 128), bitmap.countUnclaimed());

    _ = bitmap.tryClaim(0, 10);
    try testing.expectEqual(@as(usize, 10), bitmap.countClaimed());
    try testing.expectEqual(@as(usize, 118), bitmap.countUnclaimed());

    _ = bitmap.tryClaim(64, 20); // Cross field boundary
    try testing.expectEqual(@as(usize, 30), bitmap.countClaimed());
}

test "AtomicBitmap: isEmpty and isFull" {
    const testing = std.testing;
    const Bitmap64 = AtomicBitmap(64, true);

    var bitmap = Bitmap64{};

    try testing.expect(bitmap.isEmpty());
    try testing.expect(!bitmap.isFull());

    _ = bitmap.tryClaim(0, 64);
    try testing.expect(!bitmap.isEmpty());
    try testing.expect(bitmap.isFull());

    bitmap.clear();
    try testing.expect(bitmap.isEmpty());
}

test "AtomicBitmap: initFull and clear" {
    const testing = std.testing;
    const Bitmap64 = AtomicBitmap(64, true);

    var bitmap = Bitmap64.initFull();
    try testing.expect(bitmap.isFull());

    bitmap.clear();
    try testing.expect(bitmap.isEmpty());
}

test "AtomicBitmap: stack allocated sizes" {
    const testing = std.testing;

    // Verify comptime sizes are correct
    try testing.expectEqual(@as(usize, 64), SmallBitmap.comptime_bit_count);
    try testing.expectEqual(@as(usize, 1), SmallBitmap.comptime_fields_count);
    try testing.expect(SmallBitmap.is_stack_allocated);

    try testing.expectEqual(@as(usize, 256), MediumBitmap.comptime_bit_count);
    try testing.expectEqual(@as(usize, 256 / FIELD_BITS), MediumBitmap.comptime_fields_count);
    try testing.expect(MediumBitmap.is_stack_allocated);

    try testing.expectEqual(@as(usize, 512), SliceBitmap.comptime_bit_count);
    try testing.expectEqual(@as(usize, 512 / FIELD_BITS), SliceBitmap.comptime_fields_count);
    try testing.expect(SliceBitmap.is_stack_allocated);

    // Verify they can be stack allocated
    var small = SmallBitmap{};
    var medium = MediumBitmap{};
    var large = LargeBitmap{};

    _ = small.tryClaim(0, 1);
    _ = medium.tryClaim(0, 1);
    _ = large.tryClaim(0, 1);

    try testing.expect(small.isClaimed(0, 1));
    try testing.expect(medium.isClaimed(0, 1));
    try testing.expect(large.isClaimed(0, 1));
}

test "AtomicBitmap: heap allocated (dynamic size)" {
    const testing = std.testing;

    // DynamicBitmap is heap-allocated
    try testing.expect(!DynamicBitmap.is_stack_allocated);

    // Initialize with allocator and runtime size
    const runtime_size: usize = 1000;
    var bitmap = try DynamicBitmap.init(testing.allocator, runtime_size);
    defer bitmap.deinit();

    // Verify runtime size
    try testing.expectEqual(runtime_size, bitmap.bitCount());

    // Basic operations should work the same
    try testing.expect(bitmap.isEmpty());
    try testing.expect(!bitmap.isFull());

    // Claim some bits
    try testing.expect(bitmap.tryClaim(0, 10));
    try testing.expect(bitmap.isClaimed(0, 10));
    try testing.expect(!bitmap.tryClaim(5, 5)); // Overlapping

    // Claim near the end
    try testing.expect(bitmap.tryClaim(990, 10));
    try testing.expect(bitmap.isClaimed(990, 10));

    // Can't claim past the end
    try testing.expect(!bitmap.tryClaim(995, 10));

    // Count
    try testing.expectEqual(@as(usize, 20), bitmap.countClaimed());

    // Clear
    bitmap.clear();
    try testing.expect(bitmap.isEmpty());
}

test "AtomicBitmap: arena-sized bitmap" {
    const testing = std.testing;

    // Simulate arena with 16384 blocks (like a 1GB arena with 64KB blocks)
    // Using DynamicBitmap with runtime size
    var bitmap = try DynamicBitmap.init(testing.allocator, 16384);
    defer bitmap.deinit();

    try testing.expectEqual(@as(usize, 16384), bitmap.bitCount());

    // Find and claim blocks
    const idx1 = bitmap.tryFindAndClaim(64);
    try testing.expect(idx1 != null);
    try testing.expectEqual(@as(usize, 0), idx1.?);

    const idx2 = bitmap.tryFindAndClaim(64);
    try testing.expect(idx2 != null);
    try testing.expectEqual(@as(usize, 64), idx2.?);

    // Free first block and find again
    bitmap.unclaim(0, 64);
    const idx3 = bitmap.tryFindAndClaim(64);
    try testing.expectEqual(@as(usize, 0), idx3.?);

    // Test wrap-around search
    bitmap.clear();
    _ = bitmap.tryClaim(0, 100); // Claim first 100

    const idx4 = bitmap.tryFindFromAndClaim(50, 10);
    try testing.expect(idx4 != null);
    try testing.expectEqual(@as(usize, 100), idx4.?); // Should find after claimed region
}
