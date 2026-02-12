//! # Memory Identification
//!
//! Tracking structure for memory provenance and attributes.
//! Every allocated memory region has an associated MemID that
//! describes where it came from and its properties.
//!
//! ## Memory Kinds
//!
//! | Kind         | Source              | Can Decommit | Notes                |
//! |--------------|---------------------|--------------|----------------------|
//! | MEM_NONE     | Not allocated       | N/A          | Default/invalid      |
//! | MEM_EXTERNAL | External provider   | No           | User-provided memory |
//! | MEM_STATIC   | Static allocation   | No           | Compile-time memory  |
//! | MEM_OS       | Direct mmap         | Yes          | Normal OS pages      |
//! | MEM_OS_HUGE  | Huge pages (1 GiB)  | No           | Pinned, no decommit  |
//! | MEM_OS_REMAP | Remappable (mremap) | Yes          | Supports resize      |
//! | MEM_ARENA    | Arena allocator     | Yes          | Via arena's blocks   |
//!
//! ## Flags
//!
//! - `is_pinned`: Cannot decommit/reset (huge pages)
//! - `initially_committed`: Was originally backed by physical memory
//! - `initially_zero`: Was originally zero-initialized

const std = @import("std");
const builtin = @import("builtin");
const assert = @import("util.zig").assert;

/// Memory identification and provenance tracking
///
/// Stores information about where memory came from and its attributes.
/// Used to properly handle decommit, reset, and freeing operations.
pub const MemID = struct {
    memkind: MemKind = .MEM_NONE,
    flags: Flags = .{},

    // Memory info - interpretation depends on memkind
    mem: MemInfo = .{ .none = {} },

    pub const MemInfo = union(enum) {
        none: void, // MEM_NONE, MEM_EXTERNAL, MEM_STATIC
        os: []u8, // MEM_OS, MEM_OS_HUGE, MEM_OS_REMAP
        arena: MemArenaInfo, // MEM_ARENA
    };
    // Memory can reside in arena's, direct OS allocated, or statically allocated. The memid keeps track of this.
    pub const MemKind = enum(u8) {
        MEM_NONE, // not allocated
        MEM_EXTERNAL, // not owned by mimalloc but provided externally
        MEM_STATIC, // allocated in a static area
        MEM_OS, // allocated from the OS
        MEM_OS_HUGE, // allocated as huge OS pages (usually 1GiB)
        MEM_OS_REMAP, // allocated in a remapable area
        MEM_ARENA, // allocated from an arena

        /// check what is allocated from os
        pub inline fn memkind_is_os(self: MemKind) bool {
            return self >= .MEM_OS and self <= .MEM_OS_REMAP;
        }
    };

    pub const MemArenaInfo = struct {
        block_index: usize, // index in the arena
        id: usize, // arena id (>= 1)
        is_exclusive: bool, // this arena can only be used for specific arena allocations
    };

    pub inline fn none() MemID {
        return .{};
    }

    pub inline fn create_os(
        buf: []u8,
        committed: bool,
        is_zero: bool,
        is_large: bool,
    ) MemID {
        return .{
            .memkind = .MEM_OS,
            .mem = .{ .os = buf },
            .flags = .{
                .initially_committed = committed,
                .initially_zero = is_zero,
                .is_pinned = is_large,
            },
        };
    }

    pub inline fn create_arena(
        block_index: usize,
        arena_id: usize,
        is_exclusive: bool,
        committed: bool,
        is_zero: bool,
        is_pinned: bool,
    ) MemID {
        return .{
            .memkind = .MEM_ARENA,
            .mem = .{ .arena = .{
                .block_index = block_index,
                .id = arena_id,
                .is_exclusive = is_exclusive,
            } },
            .flags = .{
                .initially_committed = committed,
                .initially_zero = is_zero,
                .is_pinned = is_pinned,
            },
        };
    }

    pub const Flags = packed struct(u8) {
        is_pinned: bool = false, // `true` if we cannot decommit/reset/protect in this memory (e.g. when allocated using large (2Mib) or huge (1GiB) OS pages)
        initially_committed: bool = false, // `true` if the memory was originally allocated as committed
        initially_zero: bool = false, // `true` if the memory was originally zero initialized
        _pad: u5 = 0,
    };

    /// Get OS memory slice (asserts memkind is OS type)
    pub inline fn os_slice(self: MemID) []u8 {
        assert(self.memkind.memkind_is_os());
        return self.mem.os;
    }

    /// Get arena info (asserts memkind is MEM_ARENA)
    pub inline fn arena_info(self: MemID) MemArenaInfo {
        assert(self.memkind == .MEM_ARENA);
        return self.mem.arena;
    }
};
