const std = @import("std");
const assert = @import("util.zig").assert;
const types = @import("types.zig");
const testing = std.testing;

/// A bit set with static size, which is backed by an array of usize.
/// This set is good for sets with a larger size, but may use
/// more bytes than necessary if your set is small.
pub fn ArrayBitSet(comptime MaskIntType: type, comptime size: usize) type {
    const mask_info: std.builtin.Type = @typeInfo(MaskIntType);

    // Make sure the mask int is indeed an int
    if (mask_info != .int) @compileError("ArrayBitSet can only operate on integer masks, but was passed " ++ @typeName(MaskIntType));

    // It must also be unsigned.
    if (mask_info.int.signedness != .unsigned) @compileError("ArrayBitSet requires an unsigned integer mask type, but was passed " ++ @typeName(MaskIntType));

    // And it must not be empty.
    if (MaskIntType == u0)
        @compileError("ArrayBitSet requires a sized integer for its mask int.  u0 does not work.");

    const byte_size = std.mem.byte_size_in_bits;

    // We use shift and truncate to decompose indices into mask indices and bit indices.
    // This operation requires that the mask has an exact power of two number of bits.
    if (!std.math.isPowerOfTwo(@bitSizeOf(MaskIntType))) {
        var desired_bits = std.math.ceilPowerOfTwoAssert(usize, @bitSizeOf(MaskIntType));
        if (desired_bits < byte_size) desired_bits = byte_size;
        const FixedMaskType = std.meta.Int(.unsigned, desired_bits);
        @compileError("ArrayBitSet was passed integer type " ++ @typeName(MaskIntType) ++
            ", which is not a power of two.  Please round this up to a power of two integer size (i.e. " ++ @typeName(FixedMaskType) ++ ").");
    }

    // Make sure the integer has no padding bits.
    // Those would be wasteful here and are probably a mistake by the user.
    // This case may be hit with small powers of two, like u4.
    if (@bitSizeOf(MaskIntType) != @sizeOf(MaskIntType) * byte_size) {
        var desired_bits = @sizeOf(MaskIntType) * byte_size;
        desired_bits = std.math.ceilPowerOfTwoAssert(usize, desired_bits);
        const FixedMaskType = std.meta.Int(.unsigned, desired_bits);
        @compileError("ArrayBitSet was passed integer type " ++ @typeName(MaskIntType) ++
            ", which contains padding bits.  Please round this up to an unpadded integer size (i.e. " ++ @typeName(FixedMaskType) ++ ").");
    }

    return extern struct {
        const Self = @This();

        // TODO: Make this a comptime field once those are fixed
        /// The number of items in this bit set
        pub const bit_length: usize = size;

        /// The integer type used to represent a mask in this bit set
        pub const MaskInt = MaskIntType;

        /// The integer type used to shift a mask in this bit set
        pub const ShiftInt = std.math.Log2Int(MaskInt);

        // bits in one mask
        const mask_len = @bitSizeOf(MaskInt);
        // total number of masks
        pub const num_masks = (size + mask_len - 1) / mask_len;
        comptime {
            assert(num_masks > 0);
        }
        // padding bits in the last mask (may be 0)
        const last_pad_bits = mask_len * num_masks - size;
        // Mask of valid bits in the last mask.
        // All functions will ensure that the invalid
        // bits in the last mask are zero.
        pub const last_item_mask = ~@as(MaskInt, 0) >> last_pad_bits;

        /// The bit masks, ordered with lower indices first.
        /// Padding bits at the end are undefined.
        masks: [num_masks]MaskInt,

        /// Creates a bit set with no elements present.
        pub fn initEmpty() Self {
            return .{ .masks = [_]MaskInt{0} ** num_masks };
        }

        /// Creates a bit set with all elements present.
        pub fn initFull() Self {
            return .{ .masks = [_]MaskInt{~@as(MaskInt, 0)} ** (num_masks - 1) ++ [_]MaskInt{last_item_mask} };
        }

        /// Returns the number of bits in this bit set
        pub inline fn capacity(self: *const Self) usize {
            _ = self;
            return bit_length;
        }

        /// Returns true if the bit at the specified index
        /// is present in the set, false otherwise.
        pub inline fn isSet(self: Self, index: usize) bool {
            assert(index < bit_length);
            return (self.masks[maskIndex(index)] & maskBit(index)) != 0;
        }

        /// Returns the total number of set bits in this bit set.
        pub inline fn count(self: *const Self) usize {
            var total: usize = 0;
            inline for (self.masks) |mask| {
                total += @popCount(mask);
            }
            return total;
        }

        /// Changes the value of the specified bit of the bit
        /// set to match the passed boolean.
        pub fn setValue(self: *Self, index: usize, value: bool) void {
            assert(index < bit_length);
            const bit = maskBit(index);
            const mask_index = maskIndex(index);
            const new_bit = bit & std.math.boolMask(MaskInt, value);
            self.masks[mask_index] = (self.masks[mask_index] & ~bit) | new_bit;
        }

        /// Adds a specific bit to the bit set
        pub fn set(self: *Self, index: usize) void {
            assert(index < bit_length);
            self.masks[maskIndex(index)] |= maskBit(index);
        }

        /// Changes the value of all bits in the specified range to
        /// match the passed boolean.
        pub fn setRangeValue(self: *Self, range: Range, value: bool) void {
            assert(range.end <= bit_length);
            assert(range.start <= range.end);
            if (range.start == range.end) return;
            const start_mask_index = maskIndex(range.start);
            const start_bit = @as(ShiftInt, @truncate(range.start));

            const end_mask_index = maskIndex(range.end);
            const end_bit = @as(ShiftInt, @truncate(range.end));

            if (start_mask_index == end_mask_index) {
                var mask1 = std.math.boolMask(MaskInt, true) << start_bit;
                var mask2 = std.math.boolMask(MaskInt, true) >> (mask_len - 1) - (end_bit - 1);
                self.masks[start_mask_index] &= ~(mask1 & mask2);

                mask1 = std.math.boolMask(MaskInt, value) << start_bit;
                mask2 = std.math.boolMask(MaskInt, value) >> (mask_len - 1) - (end_bit - 1);
                self.masks[start_mask_index] |= mask1 & mask2;
            } else {
                var bulk_mask_index: usize = undefined;
                if (start_bit > 0) {
                    self.masks[start_mask_index] =
                        (self.masks[start_mask_index] & ~(std.math.boolMask(MaskInt, true) << start_bit)) |
                        (std.math.boolMask(MaskInt, value) << start_bit);
                    bulk_mask_index = start_mask_index + 1;
                } else {
                    bulk_mask_index = start_mask_index;
                }

                while (bulk_mask_index < end_mask_index) : (bulk_mask_index += 1) {
                    self.masks[bulk_mask_index] = std.math.boolMask(MaskInt, value);
                }

                if (end_bit > 0) {
                    self.masks[end_mask_index] =
                        (self.masks[end_mask_index] & (std.math.boolMask(MaskInt, true) << end_bit)) |
                        (std.math.boolMask(MaskInt, value) >> ((@bitSizeOf(MaskInt) - 1) - (end_bit - 1)));
                }
            }
        }

        /// Removes a specific bit from the bit set
        pub fn unset(self: *Self, index: usize) void {
            assert(index < bit_length);
            self.masks[maskIndex(index)] &= ~maskBit(index);
        }

        /// Flips a specific bit in the bit set
        pub fn toggle(self: *Self, index: usize) void {
            assert(index < bit_length);
            self.masks[maskIndex(index)] ^= maskBit(index);
        }

        /// Flips all bits in this bit set which are present
        /// in the toggles bit set.
        pub fn toggleSet(self: *Self, toggles: *const Self) void {
            inline for (&self.masks, 0..) |*mask, i| {
                mask.* ^= toggles.masks[i];
            }
        }

        /// Flips every bit in the bit set.
        pub fn toggleAll(self: *Self) void {
            inline for (&self.masks) |*mask| {
                mask.* = ~mask.*;
            }

            // Zero the padding bits
            self.masks[num_masks - 1] &= last_item_mask;
        }

        /// Performs a union of two bit sets, and stores the
        /// result in the first one.  Bits in the result are
        /// set if the corresponding bits were set in either input.
        pub inline fn setUnion(self: *Self, other: *const Self) void {
            inline for (&self.masks, 0..) |*mask, i| {
                mask.* |= other.masks[i];
            }
        }

        /// Performs an intersection of two bit sets, and stores
        /// the result in the first one.  Bits in the result are
        /// set if the corresponding bits were set in both inputs.
        pub inline fn setIntersection(self: *Self, other: *const Self) void {
            inline for (&self.masks, 0..) |*mask, i| {
                mask.* &= other.masks[i];
            }
        }

        /// Finds the index of the first set bit.
        /// If no bits are set, returns null.
        pub fn findFirstSet(self: *const Self) ?usize {
            var offset: usize = 0;
            const mask = for (self.masks) |mask| {
                if (mask != 0) break mask;
                offset += @bitSizeOf(MaskInt);
            } else return null;
            return offset + @ctz(mask);
        }

        /// Finds the index of the last set bit.
        /// If no bits are set, returns null.
        pub fn findLastSet(self: *const Self) ?usize {
            if (bit_length == 0) return null;
            const bs = @bitSizeOf(MaskInt);
            var len = bit_length / bs;
            if (bit_length % bs != 0) len += 1;
            var offset: usize = len * bs;
            var idx: usize = len - 1;
            inline while (self.masks[idx] == 0) : (idx -= 1) {
                offset -= bs;
                if (idx == 0) return null;
            }
            offset -= @clz(self.masks[idx]);
            offset -= 1;
            return offset;
        }

        /// Finds the index of the first set bit, and unsets it.
        /// If no bits are set, returns null.
        pub fn toggleFirstSet(self: *Self) ?usize {
            var offset: usize = 0;
            const mask = inline for (&self.masks) |*mask| {
                if (mask.* != 0) break mask;
                offset += @bitSizeOf(MaskInt);
            } else return null;
            const index = @ctz(mask.*);
            mask.* &= (mask.* - 1);
            return offset + index;
        }

        /// Returns true iff every corresponding bit in both
        /// bit sets are the same.
        pub fn eql(self: *const Self, other: *const Self) bool {
            inline for (0..num_masks) |i| {
                if (self.masks[i] != other.masks[i]) return false;
            }
            return true;
        }
        /// Returns true iff the first bit set is the subset
        /// of the second one.
        pub fn subsetOf(self: *const Self, other: *const Self) bool {
            return self.intersectWith(other).eql(self);
        }

        /// Returns true iff the first bit set is the superset
        /// of the second one.
        pub fn supersetOf(self: *const Self, other: *const Self) bool {
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
        pub fn unionWith(self: Self, other: *const Self) Self {
            var result = self;
            result.setUnion(other);
            return result;
        }

        /// Returns the intersection of two bit sets. Bits in
        /// the result are set if the corresponding bits were
        /// set in both inputs.
        pub fn intersectWith(self: Self, other: *const Self) Self {
            var result = self;
            result.setIntersection(other);
            return result;
        }

        /// Returns the xor of two bit sets. Bits in the
        /// result are set if the corresponding bits were
        /// not the same in both inputs.
        pub fn xorWith(self: Self, other: *const Self) Self {
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
            return Iterator(options).init(&self.masks, last_item_mask);
        }

        pub fn Iterator(comptime options: IteratorOptions) type {
            return BitSetIterator(MaskInt, options);
        }

        inline fn maskBit(index: usize) MaskInt {
            return @as(MaskInt, 1) << @as(ShiftInt, @truncate(index));
        }
        inline fn maskIndex(index: usize) usize {
            return index >> @bitSizeOf(ShiftInt);
        }
        inline fn boolMaskBit(index: usize, value: bool) MaskInt {
            return @as(MaskInt, @intFromBool(value)) << @as(ShiftInt, @intCast(index));
        }
    };
}

fn BitSetIterator(comptime MaskInt: type, comptime options: IteratorOptions) type {
    const ShiftInt = std.math.Log2Int(MaskInt);
    const kind = options.kind;
    const direction = options.direction;
    return struct {
        const Self = @This();

        // all bits which have not yet been iterated over
        bits_remain: MaskInt,
        // all words which have not yet been iterated over
        words_remain: []const MaskInt,
        // the offset of the current word
        bit_offset: usize,
        // the mask of the last word
        last_word_mask: MaskInt,

        fn init(masks: []const MaskInt, last_word_mask: MaskInt) Self {
            if (masks.len == 0) {
                return Self{
                    .bits_remain = 0,
                    .words_remain = &[_]MaskInt{},
                    .last_word_mask = last_word_mask,
                    .bit_offset = 0,
                };
            } else {
                var result = Self{
                    .bits_remain = 0,
                    .words_remain = masks,
                    .last_word_mask = last_word_mask,
                    .bit_offset = if (direction == .forward) 0 else (masks.len - 1) * @bitSizeOf(MaskInt),
                };
                result.nextWord(true);
                return result;
            }
        }

        /// Returns the index of the next unvisited set bit
        /// in the bit set, in ascending order.
        pub fn next(self: *Self) ?usize {
            while (self.bits_remain == 0) {
                if (self.words_remain.len == 0) return null;
                self.nextWord(false);
                switch (direction) {
                    .forward => self.bit_offset += @bitSizeOf(MaskInt),
                    .reverse => self.bit_offset -= @bitSizeOf(MaskInt),
                }
            }

            switch (direction) {
                .forward => {
                    const next_index = @ctz(self.bits_remain) + self.bit_offset;
                    self.bits_remain &= self.bits_remain - 1;
                    return next_index;
                },
                .reverse => {
                    const leading_zeroes = @clz(self.bits_remain);
                    const top_bit = (@bitSizeOf(MaskInt) - 1) - leading_zeroes;
                    const no_top_bit_mask = (@as(MaskInt, 1) << @as(ShiftInt, @intCast(top_bit))) - 1;
                    self.bits_remain &= no_top_bit_mask;
                    return top_bit + self.bit_offset;
                },
            }
        }

        // Load the next word.  Don't call this if there
        // isn't a next word.  If the next word is the
        // last word, mask off the padding bits so we
        // don't visit them.
        inline fn nextWord(self: *Self, comptime is_first_word: bool) void {
            var word = switch (direction) {
                .forward => self.words_remain[0],
                .reverse => self.words_remain[self.words_remain.len - 1],
            };
            switch (kind) {
                .set => {},
                .unset => {
                    word = ~word;
                    if ((direction == .reverse and is_first_word) or
                        (direction == .forward and self.words_remain.len == 1))
                    {
                        word &= self.last_word_mask;
                    }
                },
            }
            switch (direction) {
                .forward => self.words_remain = self.words_remain[1..],
                .reverse => self.words_remain.len -= 1,
            }
            self.bits_remain = word;
        }
    };
}

/// A range of indices within a bitset.
pub const Range = struct {
    /// The index of the first bit of interest.
    start: usize,
    /// The index immediately after the last bit of interest.
    end: usize,
};

pub const IteratorOptions = struct {
    /// determines which bits should be visited
    kind: Type = .set,
    /// determines the order in which bit indices should be visited
    direction: Direction = .forward,

    pub const Type = enum {
        /// visit indexes of set bits
        set,
        /// visit indexes of unset bits
        unset,
    };

    pub const Direction = enum {
        /// visit indices in ascending order
        forward,
        /// visit indices in descending order.
        /// Note that this may be slightly more expensive than forward iteration.
        reverse,
    };
};

pub const CommitMask = struct {
    mask: ArrayBitSet(usize, types.COMMIT_MASK_BITS), //512 bits
    const mask_len = @bitSizeOf(usize);

    const MaskType = ArrayBitSet(usize, types.COMMIT_MASK_BITS);

    const num_masks = MaskType.num_masks;

    const Self = @This();

    pub fn initEmpty() Self {
        return .{ .mask = .initEmpty() };
    }

    pub fn initFull() Self {
        return .{ .mask = .initFull() };
    }

    pub fn isEmpty(self: *const Self) bool {
        return self.mask.count() == 0;
    }

    pub inline fn isFull(self: *const Self) bool {
        return self.mask.count() == types.COMMIT_MASK_BITS;
    }

    pub inline fn count(self: *const Self) usize {
        return self.mask.count();
    }

    // all bits from othe sets inself?
    pub fn allSet(self: *const Self, other: *const Self) bool {
        const a = self.mask.masks;
        const b = other.mask.masks;
        inline for (0..num_masks) |i| {
            if ((a[i] & b[i]) != b[i]) return false;
        }
        return true;
    }

    // is there any sets bits?
    pub fn anySet(self: *const Self, other: *const Self) bool {
        const a = self.mask.masks;
        const b = other.mask.masks;
        inline for (0..num_masks) |i| {
            if ((a[i] & b[i]) != 0) return true;
        }
        return false;
    }

    pub inline fn eql(self: *const Self, other: *const Self) bool {
        return self.mask.eql(&other.mask);
    }

    //  OR: self |= other
    pub inline fn set(self: *Self, other: *const Self) void {
        self.mask.setUnion(&other.mask);
    }

    //  AND-NOT: self &= ~other
    pub fn clear(self: *Self, other: *const Self) void {
        var dest = &self.mask.masks;
        const src = other.mask.masks;
        inline for (0..num_masks) |i| {
            dest[i] &= ~src[i];
        }
    }

    //  self & other → new commit mask
    pub fn intersect(self: Self, other: *const Self) Self {
        return .{ .mask = self.mask.intersectWith(&other.mask) };
    }

    pub fn initRange(start: usize, len: usize) Self {
        assert(start + len <= types.COMMIT_MASK_BITS);
        var cm = Self.initEmpty();

        if (len > 0) {
            cm.mask.setRangeValue(.{ .start = start, .end = start + len }, true);
        }
        return cm;
    }

    // Commited size
    pub fn committedSize(self: *const Self, total: usize) usize {
        assert(total % types.COMMIT_MASK_BITS == 0);
        const unit = total / types.COMMIT_MASK_BITS;
        return self.mask.count() * unit;
    }

    // search next run-а sets bits
    pub fn nextRun(self: Self, idx: *usize) usize {
        var i = idx.*;
        while (i < types.COMMIT_MASK_BITS and !self.mask.isSet(i)) : (i += 1) {}
        if (i >= types.COMMIT_MASK_BITS) {
            idx.* = types.COMMIT_MASK_BITS;
            return 0;
        }
        const start = i;
        while (i < types.COMMIT_MASK_BITS and self.mask.isSet(i)) : (i += 1) {}
        const c = i - start;
        idx.* = i;
        return c;
    }

    // iterator on nextrun,foreach
    pub const Iterator = struct {
        cm: *const Self,
        idx: usize = 0,

        pub fn next(it: *Iterator) ?struct { start: usize, len: usize } {
            const len = it.cm.nextRun(&it.idx);
            if (len == 0) return null;
            return .{ .start = it.idx - len, .len = len };
        }
    };

    pub fn iterator(self: *const Self) Iterator {
        return .{ .cm = self };
    }
};

test "CommitMask: initEmpty / initFull / isEmpty / isFull" {
    const empty = CommitMask.initEmpty();
    const full = CommitMask.initFull();

    try testing.expect(empty.isEmpty());
    try testing.expect(!empty.isFull());

    try testing.expect(!full.isEmpty());
    try testing.expect(full.isFull());

    try testing.expect(full.mask.count() == types.COMMIT_MASK_BITS);
    try testing.expect(empty.mask.count() == 0);
}

test "CommitMask: initRange" {
    var cm = CommitMask.initRange(100, 50);
    try testing.expect(cm.mask.count() == 50);
    try testing.expect(cm.mask.isSet(100));
    try testing.expect(cm.mask.isSet(149));
    try testing.expect(!cm.mask.isSet(99));
    try testing.expect(!cm.mask.isSet(150));

    // full diap
    cm = CommitMask.initRange(0, types.COMMIT_MASK_BITS);
    try testing.expect(cm.isFull());

    // zero diap
    cm = CommitMask.initRange(500, 0);
    try testing.expect(cm.isEmpty());
}

test "CommitMask: allSet / anySet" {
    const BITS = types.COMMIT_MASK_BITS; // 512
    const half = BITS / 2; // 256
    const quarter = BITS / 4; // 128

    var a = CommitMask.initRange(0, half); // 0..255
    var b = CommitMask.initRange(quarter, half); // 128..383

    try testing.expect(a.allSet(&a));
    try testing.expect(!a.allSet(&b)); // b has  256..383, which not set in a
    try testing.expect(!b.allSet(&a));

    try testing.expect(a.anySet(&b)); // intersection 128..255
    try testing.expect(b.anySet(&a));

    // c — outside
    const c_start = 400;
    if (c_start + 50 <= BITS) {
        var c = CommitMask.initRange(c_start, 50); // 400..449
        try testing.expect(!a.anySet(&c));
        try testing.expect(!b.anySet(&c));
    }
}

test "CommitMask: set / clear / intersect" {
    const BITS = types.COMMIT_MASK_BITS; // 512
    const part1 = BITS / 4; // 128
    const part2 = BITS / 2; // 256

    var a = CommitMask.initRange(0, part1); // bits 0..127  → 128 bits
    var b = CommitMask.initRange(100, part2); // bits 100..355 → 256 bits

    // after a |= b:
    // a has 0..355 → 356 bits
    a.set(&b);
    try testing.expect(a.count() == 356);

    // intersection: only 100..355 → 256 bits
    const inter = a.intersect(&b);
    try testing.expect(inter.count() == 256);

    // after a &= ~b:
    // delete a from b (100..355)
    // remained only 0..99 → 100 bits
    a.clear(&b);
    try testing.expect(a.count() == 100);
}

test "CommitMask: eql" {
    const BITS = types.COMMIT_MASK_BITS;
    var a = CommitMask.initRange(0, BITS);
    const b = CommitMask.initRange(0, BITS);
    try testing.expect(a.eql(&b) == true);
}
