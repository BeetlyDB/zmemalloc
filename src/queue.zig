const std = @import("std");
const assert = @import("util.zig").assert;
const builtin = @import("builtin");

/// An intrusive queue implementation. The type T must have a field
/// "next" and prev of type `?*T` which is an enum with a value matching the passed in
/// value
/// FIFO  A -> B -> C
/// pop:   A
/// or first - last
pub fn Intrusive(
    comptime Node: type,
    comptime next_field_enum: std.meta.FieldEnum(Node),
    comptime prev_field_enum: std.meta.FieldEnum(Node),
) type {
    comptime {
        assert(@typeInfo(Node) == .@"struct");
        assert(prev_field_enum != next_field_enum);
    }

    const prev_field = @tagName(prev_field_enum);
    const next_field = @tagName(next_field_enum);
    assert(@FieldType(Node, prev_field) == ?*Node);
    assert(@FieldType(Node, next_field) == ?*Node);

    return struct {
        const Self = @This();

        /// Head is the front of the queue and tail is the back of the queue.
        head: ?*Node = null,
        tail: ?*Node = null,
        count: u64 = 0,

        /// Enqueue a new element to the back of the queue.
        pub inline fn push(self: *Self, v: *Node) void {
            assert(@field(v, next_field) == null);
            assert(@field(v, prev_field) == null);
            if (self.tail) |tail| {
                // If we have elements in the queue, then we add a new tail.
                @field(tail, next_field) = v;
                @field(v, prev_field) = tail;
                self.tail = v;
            } else {
                // No elements in the queue we setup the initial state.
                self.head = v;
                self.tail = v;
            }
            self.count += 1;
        }

        /// Dequeue the next element from the queue.
        pub inline fn pop(self: *Self) ?*Node {
            assert(self.count > 0);
            // The next element is in "head".
            const next = self.head orelse return null;

            // If the head and tail are equal this is the last element
            // so we also set tail to null so we can now be empty.
            if (self.head == self.tail) self.tail = null;

            // Head is whatever is next (if we're the last element,
            // this will be null);
            self.head = @field(next, next_field);
            if (self.head) |h| @field(h, prev_field) = null;

            // We set the "next" field to null so that this element
            // can be inserted again.
            @field(next, next_field) = null;
            @field(next, prev_field) = null;
            self.count -= 1;
            return next;
        }

        /// Returns true if the queue is empty.
        pub inline fn empty(self: *const Self) bool {
            return self.head == null;
        }

        /// Removes the item from the queue. Asserts that Queue contains the item
        pub inline fn remove(self: *Self, item: *Node) void {
            assert(self.hasItem(item));
            assert(self.count > 0);
            const prev = @field(item, prev_field);
            const next = @field(item, next_field);

            if (prev) |p| {
                @field(p, next_field) = next;
            } else {
                self.head = next;
            }

            if (next) |n| {
                @field(n, prev_field) = prev;
            } else {
                self.tail = prev;
            }

            @field(item, next_field) = null;
            @field(item, prev_field) = null;
        }

        pub inline fn hasItem(self: *const Self, item: *Node) bool {
            var maybe_node = self.head;
            while (maybe_node) |node| {
                if (node == item) return true;
                maybe_node = @field(node, next_field);
            } else return false;
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }
    };
}

const N = struct {
    next: ?*N = null,
    prev: ?*N = null,
    value: usize,
};

test "intrusive queue push/pop basic" {
    const Queue = Intrusive(N, .next, .prev);

    var q = Queue{};

    var a = N{ .value = 1 };
    var b = N{ .value = 2 };
    var c = N{ .value = 3 };

    try std.testing.expect(q.empty());
    try std.testing.expectEqual(@as(usize, 0), q.len());

    q.push(&a);
    try std.testing.expect(!q.empty());
    try std.testing.expectEqual(@as(usize, 1), q.len());

    q.push(&b);
    q.push(&c);
    try std.testing.expectEqual(@as(usize, 3), q.len());

    const p1 = q.pop().?;
    try std.testing.expectEqual(@as(usize, 1), p1.value);
    try std.testing.expectEqual(@as(usize, 2), q.len());

    const p2 = q.pop().?;
    try std.testing.expectEqual(@as(usize, 2), p2.value);

    const p3 = q.pop().?;
    try std.testing.expectEqual(@as(usize, 3), p3.value);

    try std.testing.expect(q.empty());
    try std.testing.expectEqual(@as(usize, 0), q.len());
}

/// An intrusive doubly-linked list. lifo
pub fn DoublyLinkedListType(
    comptime Node: type,
    comptime field_back_enum: std.meta.FieldEnum(Node),
    comptime field_next_enum: std.meta.FieldEnum(Node),
) type {
    assert(@typeInfo(Node) == .@"struct");
    assert(field_back_enum != field_next_enum);

    const field_back = @tagName(field_back_enum);
    const field_next = @tagName(field_next_enum);
    assert(@FieldType(Node, field_back) == ?*Node);
    assert(@FieldType(Node, field_next) == ?*Node);

    return struct {
        const DoublyLinkedList = @This();

        tail: ?*Node = null,
        count: u32 = 0,

        pub inline fn contains(list: *const DoublyLinkedList, target: *const Node) bool {
            var count: u32 = 0;

            var iterator = list.tail;
            while (iterator) |node| {
                if (node == target) return true;
                iterator = @field(node, field_back);
                count += 1;
            }

            assert(count == list.count);
            return false;
        }

        pub const hasItem = contains;

        pub inline fn getTail(list: *const DoublyLinkedList) ?*Node {
            if (list.tail) |tail| {
                assert(list.count > 0);
                assert(@field(tail, field_next) == null);
                return tail;
            } else {
                return null;
            }
        }

        pub inline fn empty(list: *const DoublyLinkedList) bool {
            assert((list.count == 0) == (list.tail == null));
            return list.count == 0;
        }

        pub inline fn push(list: *DoublyLinkedList, node: *Node) void {
            assert(@field(node, field_back) == null);
            assert(@field(node, field_next) == null);

            @prefetch(node, .{ .cache = .data, .locality = 3, .rw = .read });

            if (list.tail) |tail| {
                assert(list.count > 0);
                assert(@field(tail, field_next) == null);

                @field(node, field_back) = tail;
                @field(tail, field_next) = node;
            } else {
                assert(list.count == 0);
            }

            list.tail = node;
            list.count += 1;
        }

        pub inline fn pop(list: *DoublyLinkedList) ?*Node {
            if (list.tail) |tail_old| {
                assert(list.count > 0);
                assert(@field(tail_old, field_next) == null);

                if (@field(tail_old, field_back)) |prev| {
                    @branchHint(.likely);
                    @prefetch(prev, .{ .cache = .data, .locality = 3, .rw = .read });
                }

                list.tail = @field(tail_old, field_back);
                list.count -= 1;

                if (list.tail) |tail_new| {
                    assert(@field(tail_new, field_next) == tail_old);
                    @field(tail_new, field_next) = null;
                }

                @field(tail_old, field_back) = null;
                return tail_old;
            } else {
                assert(list.count == 0);
                return null;
            }
        }

        pub inline fn remove(list: *DoublyLinkedList, node: *Node) void {
            assert(list.count > 0);
            assert(list.tail != null);

            const tail = list.tail.?;

            if (node == tail) {
                // Pop the last element of the list.
                assert(@field(node, field_next) == null);
                list.tail = @field(node, field_back);
            }
            if (@field(node, field_back)) |node_back| {
                assert(@field(node_back, field_next).? == node);
                @field(node_back, field_next) = @field(node, field_next);
            }
            if (@field(node, field_next)) |node_next| {
                assert(@field(node_next, field_back).? == node);
                @field(node_next, field_back) = @field(node, field_back);
            }
            @field(node, field_back) = null;
            @field(node, field_next) = null;
            list.count -= 1;
            assert((list.count == 0) == (list.tail == null));
        }
    };
}

test "DoublyLinkedList LIFO" {
    const Node = struct { id: u32, back: ?*@This() = null, next: ?*@This() = null };
    const List = DoublyLinkedListType(Node, .back, .next);

    var nodes: [3]Node = undefined;
    for (&nodes, 0..) |*node, i| node.* = .{ .id = @intCast(i) };

    var list = List{};
    list.push(&nodes[0]);
    list.push(&nodes[1]);
    list.push(&nodes[2]);

    try std.testing.expectEqual(list.pop().?, &nodes[2]);
    try std.testing.expectEqual(list.pop().?, &nodes[1]);
    try std.testing.expectEqual(list.pop().?, &nodes[0]);
    try std.testing.expectEqual(list.pop(), null);
}

const QueueLink = extern struct {
    next: ?*QueueLink = null,
};

//  FIFO
/// An intrusive first in/first out linked list.
/// The element type T must have a field called "link" of type QueueType(T).Link.
pub fn QueueType(comptime T: type) type {
    return struct {
        any: QueueAny,

        pub const Link = QueueLink;
        const Queue = @This();

        pub inline fn init() Queue {
            return .{ .any = .{} };
        }

        pub inline fn push(self: *Queue, link: *T) void {
            self.any.push(&link.link);
        }

        pub inline fn pop(self: *Queue) ?*T {
            const link = self.any.pop() orelse return null;
            return @alignCast(@fieldParentPtr("link", link));
        }

        pub inline fn peek_last(self: *const Queue) ?*T {
            const link = self.any.peek_last() orelse return null;
            return @alignCast(@fieldParentPtr("link", link));
        }

        pub inline fn peek(self: *const Queue) ?*T {
            const link = self.any.peek() orelse return null;
            return @alignCast(@fieldParentPtr("link", link));
        }

        pub fn count(self: *const Queue) u64 {
            return self.any.count;
        }

        pub inline fn empty(self: *const Queue) bool {
            return self.any.empty();
        }

        /// Returns whether the linked list contains the given *exact element* (pointer comparison).
        pub inline fn contains(self: *const Queue, elem_needle: *const T) bool {
            return self.any.contains(&elem_needle.link);
        }

        /// Remove an element from the Queue. Asserts that the element is
        /// in the Queue. This operation is O(N), if this is done often you
        /// probably want a different data structure.
        pub inline fn remove(self: *Queue, to_remove: *T) void {
            self.any.remove(&to_remove.link);
        }

        pub inline fn reset(self: *Queue) void {
            self.any.reset();
        }

        pub inline fn iterate(self: *const Queue) Iterator {
            return .{ .any = self.any.iterate() };
        }

        pub const Iterator = struct {
            any: QueueAny.Iterator,

            pub inline fn next(iterator: *@This()) ?*T {
                const link = iterator.any.next() orelse return null;
                return @alignCast(@fieldParentPtr("link", link));
            }
        };
    };
}

// Non-generic implementation for smaller binary and faster compile times.
const QueueAny = struct {
    in: ?*QueueLink = null,
    out: ?*QueueLink = null,
    count: u64 = 0,

    pub inline fn push(self: *QueueAny, link: *QueueLink) void {
        assert(link.next == null);
        if (self.in) |in| {
            in.next = link;
            self.in = link;
        } else {
            assert(self.out == null);
            self.in = link;
            self.out = link;
        }
        self.count += 1;
    }

    pub inline fn pop(self: *QueueAny) ?*QueueLink {
        const result = self.out orelse return null;
        self.out = result.next;
        result.next = null;
        if (self.in == result) self.in = null;
        self.count -= 1;
        return result;
    }

    pub inline fn peek_last(self: *const QueueAny) ?*QueueLink {
        return self.in;
    }

    pub inline fn peek(self: *const QueueAny) ?*QueueLink {
        return self.out;
    }

    pub inline fn empty(self: *const QueueAny) bool {
        return self.peek() == null;
    }

    pub inline fn contains(self: *const QueueAny, needle: *const QueueLink) bool {
        var iterator = self.peek();
        while (iterator) |link| : (iterator = link.next) {
            if (link == needle) return true;
        }
        return false;
    }

    pub inline fn remove(self: *QueueAny, to_remove: *QueueLink) void {
        if (to_remove == self.out) {
            _ = self.pop();
            return;
        }
        var it = self.out;
        while (it) |link| : (it = link.next) {
            if (to_remove == link.next) {
                if (to_remove == self.in) self.in = link;
                link.next = to_remove.next;
                to_remove.next = null;
                self.count -= 1;
                break;
            }
        } else unreachable;
    }

    pub inline fn reset(self: *QueueAny) void {
        self.* = .{};
    }

    pub inline fn iterate(self: *const QueueAny) Iterator {
        return .{
            .head = self.out,
        };
    }

    const Iterator = struct {
        head: ?*QueueLink,

        fn next(iterator: *Iterator) ?*QueueLink {
            const head = iterator.head orelse return null;
            iterator.head = head.next;
            return head;
        }
    };
};

/// A single intrusive link node used by IntrusiveLifo.
///
/// This link is embedded *inside* the user type `T`.
/// The LIFO itself never allocates memory and never owns elements:
/// it only rewires these `next` pointers.
///
/// Invariants:
/// - `next == null` means the node is not currently in any LIFO
/// - a node MUST NOT be pushed twice without being popped
///
/// Layout:
///     T
///     ├─ user fields...
///     └─ link: IntrusiveLifoLink
///            └─ next ──► next element's link
///
pub const IntrusiveLifoLink = extern struct {
    next: ?*IntrusiveLifoLink = null,
};

///// An intrusive last-in / first-out stack (LIFO).
///
/// Characteristics:
/// - O(1) push
/// - O(1) pop
/// - O(1) peek
/// - O(n) contains
/// - zero allocations
/// - stable memory addresses
///
/// The element type `T` MUST embed a field:
///
///     link: IntrusiveLifo(T).Link
///
/// Example:
///
///     const Item = struct {
///         value: u32,
///         link: IntrusiveLifo(Item).Link = .{},
///     };
///
/// Memory ownership:
/// - The stack does NOT own elements
/// - Caller is responsible for element lifetime
///
/// Thread-safety:
/// - NOT thread-safe
/// - External synchronization required if used concurrently
pub fn IntrusiveLifo(comptime T: type) type {
    return struct {
        any: IntrusiveLifoAny,

        pub const Link = IntrusiveLifoLink;
        const Self = @This();

        pub inline fn init() Self {
            return .{ .any = .{} };
        }

        /// O(n) count - only use for debugging
        pub inline fn count(self: *const Self) u64 {
            var n: u64 = 0;
            var next = self.any.head;
            while (next) |link| {
                n += 1;
                next = link.next;
            }
            return n;
        }

        /// Pushes a new node to the first position of the Stack.
        pub inline fn push(self: *Self, node: *T) void {
            self.any.push(&node.link);
        }

        /// Returns the first element of the Stack list, and removes it.
        pub inline fn pop(self: *Self) ?*T {
            const link = self.any.pop() orelse return null;
            return @alignCast(@fieldParentPtr("link", link));
        }

        /// Returns the first element of the Stack list, but does not remove it.
        pub inline fn peek(self: *const Self) ?*T {
            const link = self.any.peek() orelse return null;
            return @alignCast(@fieldParentPtr("link", link));
        }

        /// Checks if the Stack is empty.
        pub inline fn empty(self: *const Self) bool {
            return self.any.empty();
        }

        /// Returns whether the linked list contains the given *exact element* (pointer comparison).
        pub inline fn contains(self: *const Self, needle: *const T) bool {
            return self.any.contains(&needle.link);
        }
    };
}

/// Non-generic intrusive LIFO implementation.
///
/// This struct operates purely on IntrusiveLifoLink nodes.
/// It knows nothing about the containing type `T`.
///
/// Data layout:
///
///     head ──► [link] ──► [link] ──► [link] ──► null
///               ▲
///               │
///             top of stack
///
///Before push:
//     head
//      │
//      ▼
//    [ A ] ──► [ B ] ──► null
//
// Push C:
//
//     C.next = head
//     head   = C
//
// After push:
//
//     head
//      │
//      ▼
//    [ C ] ──► [ A ] ──► [ B ] ──► null
//
//
// Pop:
//
//     result = head  (C)
//     head   = C.next (A)
//     C.next = null
//     pop() → C
// head → A → B → null
/// Optimized LIFO - minimal operations for hot path
const IntrusiveLifoAny = struct {
    head: ?*IntrusiveLifoLink = null,

    const Self = @This();

    inline fn push(self: *Self, link: *IntrusiveLifoLink) void {
        link.next = self.head;
        @prefetch(link.next, .{ .cache = .data, .locality = 3, .rw = .read });
        self.head = link;
    }

    inline fn pop(self: *Self) ?*IntrusiveLifoLink {
        const link = self.head orelse return null;
        self.head = link.next;
        if (link.next) |next_link| {
            @branchHint(.likely);
            @prefetch(next_link, .{ .cache = .data, .locality = 3, .rw = .read });
        }
        return link;
    }

    inline fn peek(self: *const Self) ?*IntrusiveLifoLink {
        return self.head;
    }

    inline fn empty(self: *const Self) bool {
        return self.head == null;
    }

    inline fn contains(self: *const Self, needle: *const IntrusiveLifoLink) bool {
        var next = self.head;
        while (next) |link| {
            if (link == needle) return true;
            next = link.next;
        }
        return false;
    }
};

test "Stack: push/pop/peek/empty" {
    const testing = @import("std").testing;
    const Item = struct { link: IntrusiveLifoLink = .{} };

    var one: Item = .{};
    var two: Item = .{};
    var three: Item = .{};

    var stack: IntrusiveLifo(Item) = IntrusiveLifo(Item).init();

    try testing.expect(stack.empty());

    // Push one element and verify
    stack.push(&one);
    try testing.expect(!stack.empty());
    try testing.expectEqual(@as(?*Item, &one), stack.peek());
    try testing.expect(stack.contains(&one));
    try testing.expect(!stack.contains(&two));
    try testing.expect(!stack.contains(&three));

    // Push two more elements
    stack.push(&two);
    stack.push(&three);
    try testing.expect(!stack.empty());
    try testing.expectEqual(@as(?*Item, &three), stack.peek());
    try testing.expect(stack.contains(&one));
    try testing.expect(stack.contains(&two));
    try testing.expect(stack.contains(&three));

    // Pop elements and check Stack order
    try testing.expectEqual(@as(?*Item, &three), stack.pop());
    try testing.expectEqual(@as(?*Item, &two), stack.pop());
    try testing.expectEqual(@as(?*Item, &one), stack.pop());
    try testing.expect(stack.empty());
    try testing.expectEqual(@as(?*Item, null), stack.pop());
}
