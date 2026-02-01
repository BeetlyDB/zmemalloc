const std = @import("std");
const assert = @import("util.zig").assert;

pub fn BoundedArrayType(comptime T: type, comptime buffer_capacity: usize) type {
    return struct {
        buffer: [buffer_capacity]T = undefined,
        count_u32: u32 = 0,

        const BoundedArray = @This();

        pub inline fn from_slice(items: []const T) error{Overflow}!BoundedArray {
            if (items.len <= buffer_capacity) {
                var result: BoundedArray = .{};
                result.push_slice(items);
                return result;
            } else {
                return error.Overflow;
            }
        }

        pub inline fn count(array: *const BoundedArray) usize {
            return array.count_u32;
        }

        /// Returns count of elements in this BoundedArray in the specified integer types,
        /// checking at compile time that it indeed can represent the length.
        pub inline fn count_as(array: *const BoundedArray, comptime Int: type) Int {
            comptime assert(buffer_capacity <= std.math.maxInt(Int));
            return @intCast(array.count_u32);
        }

        pub inline fn full(array: BoundedArray) bool {
            return array.count_u32 == buffer_capacity;
        }

        pub inline fn empty(array: BoundedArray) bool {
            return array.count_u32 == 0;
        }

        pub inline fn get(array: *const BoundedArray, index: usize) T {
            assert(index < array.count_u32);
            return array.buffer[index];
        }

        pub inline fn slice(array: *BoundedArray) []T {
            return array.buffer[0..array.count_u32];
        }

        pub inline fn const_slice(array: *const BoundedArray) []const T {
            return array.buffer[0..array.count_u32];
        }

        pub inline fn unused_capacity_slice(array: *BoundedArray) []T {
            return array.buffer[array.count_u32..];
        }

        pub fn insert_at(array: *BoundedArray, index: usize, item: T) void {
            assert(!array.full());
            assert(index <= array.count_u32);
            @memmove(
                array.buffer[index + 1 .. array.count_u32 + 1],
                array.buffer[index..array.count_u32],
            );
            array.buffer[index] = item;
            array.count_u32 += 1;
        }

        pub fn push(array: *BoundedArray, item: T) void {
            assert(!array.full());
            array.buffer[array.count_u32] = item;
            array.count_u32 += 1;
        }

        pub fn push_slice(array: *BoundedArray, items: []const T) void {
            assert(array.count_u32 + items.len <= array.capacity());
            @memmove(
                array.buffer[array.count_u32..],
                items,
            );
            array.count_u32 += @intCast(items.len);
        }

        pub inline fn swap_remove(array: *BoundedArray, index: usize) T {
            assert(array.count_u32 > 0);
            assert(index < array.count_u32);
            const result = array.buffer[index];
            array.count_u32 -= 1;
            array.buffer[index] = array.buffer[array.count_u32];
            return result;
        }

        pub inline fn ordered_remove(array: *BoundedArray, index: usize) T {
            assert(array.count_u32 > 0);
            assert(index < array.count_u32);
            const result = array.buffer[index];
            @memmove(
                array.buffer[index .. array.count_u32 - 1],
                array.buffer[index + 1 .. array.count_u32],
            );
            array.count_u32 -= 1;
            return result;
        }

        pub fn resize(array: *BoundedArray, count_new: usize) error{Overflow}!void {
            if (count_new <= buffer_capacity) {
                array.count_u32 = @intCast(count_new);
            } else {
                return error.Overflow;
            }
        }

        pub inline fn truncate(array: *BoundedArray, count_new: usize) void {
            assert(count_new <= array.count_u32);
            array.count_u32 = @intCast(count_new); // can't overflow due to check above.
        }

        pub inline fn clear(array: *BoundedArray) void {
            array.count_u32 = 0;
        }

        pub inline fn pop(array: *BoundedArray) ?T {
            if (array.count_u32 == 0) return null;
            array.count_u32 -= 1;
            return array.buffer[array.count_u32];
        }

        pub inline fn capacity(_: *BoundedArray) usize {
            return buffer_capacity;
        }
    };
}

test "BoundedArray.insert_at" {
    const items_max = 32;
    const BoundedArrayU64 = BoundedArrayType(u64, items_max);

    // Test lists of every size (less than the capacity).
    for (0..items_max) |len| {
        var list_base = BoundedArrayU64{};
        for (0..len) |i| {
            list_base.push(i);
        }

        // Test an insert at every possible position (including an append).
        for (0..list_base.count() + 1) |i| {
            var list = list_base;

            list.insert_at(i, 12345);

            // Verify the result:

            try std.testing.expectEqual(list.count(), list_base.count() + 1);
            try std.testing.expectEqual(list.get(i), 12345);

            for (0..i) |j| {
                try std.testing.expectEqual(list.get(j), j);
            }

            for (i + 1..list.count()) |j| {
                try std.testing.expectEqual(list.get(j), j - 1);
            }
        }
    }
}
