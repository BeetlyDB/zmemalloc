//! # Thread-Local Heap
//!
//! Each thread has its own Heap for lock-free allocation.
//! The heap contains:
//! - Direct page pointers for O(1) small allocation lookup
//! - Bin queues for all size classes
//!
//! ## Size Classes and Bins
//!
//! Allocations are grouped into size classes (bins) for efficient reuse:
//! - Bins 1-8: exact word sizes (8, 16, 24, ... 64 bytes)
//! - Bins 9+: logarithmic spacing (96, 128, 192, 256, ...)
//!
//! ## Fast Path Optimization
//!
//! `pages_free_direct[wsize]` provides O(1) lookup for sizes <= 1KB:
//! ```
//! wsize = (size + 7) / 8  // Round up to word size
//! page = heap.pages_free_direct[wsize]
//! block = page.popFreeBlock()
//! ```
//!
//! For larger sizes, bin queue lookup is used.

const types = @import("types.zig");
const page = @import("page.zig");
const tld = @import("tld.zig");

/// Thread-local heap structure
///
/// Contains all allocation state for a single thread.
/// Lock-free for same-thread allocations.
pub const Heap = struct {
    /// Back-pointer to thread-local data
    tld: ?*tld.TLD = null,

    /// Thread ID (actually TLD address for fast comparison)
    thread_id: usize = 0,

    /// Direct page lookup for small sizes (wsize 0-128)
    /// `pages_free_direct[wsize]` points to a page with blocks of that word size
    /// Enables O(1) allocation for sizes <= 1KB without bin queue scanning
    pages_free_direct: [types.PAGES_DIRECT]?*page.Page = [_]?*page.Page{null} ** types.PAGES_DIRECT,

    /// Bin queues indexed by size class
    /// Each bin is a doubly-linked list of pages with free blocks
    /// `pages[binFromSize(size)]` gives the queue for that size class
    pages: [types.BIN_FULL + 1]page.Page.Queue = [_]page.Page.Queue{.{}} ** (types.BIN_FULL + 1),
};

/// Empty heap for initialization
pub const heap_empty: Heap = .{};

/// Per-thread heap instance
pub threadlocal var heap_main: Heap = heap_empty;
