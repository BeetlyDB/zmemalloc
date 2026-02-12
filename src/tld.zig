//! # Thread-Local Data (TLD)
//!
//! Centralized container for all thread-local allocation state.
//! Each thread has exactly one TLD instance that coordinates between
//! the heap, segment manager, and OS allocator.
//!
//! ## Architecture
//!
//! ```
//! ┌─────────────────────────────────────────────────────────┐
//! │                         TLD                             │
//! │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
//! │  │    Heap     │  │  SegmentsTLD │  │ OsAllocator   │   │
//! │  │  (bins,     │  │  (segment    │  │ (mmap/munmap  │   │
//! │  │   pages)    │  │   tracking)  │  │  interface)   │   │
//! │  └─────────────┘  └──────────────┘  └───────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Thread Safety
//!
//! TLD is designed for single-thread access only. Cross-thread
//! operations go through atomic `xthread_free` queues on pages,
//! never directly through another thread's TLD.
//!
//! ## Initialization
//!
//! TLD is lazily initialized on first allocation in each thread.
//! The `heap_backing` pointer links back to the thread's main heap.

const heap = @import("heap.zig");
const os = @import("os.zig");
const os_alloc = @import("os_allocator.zig");

pub const SegmentsTLD = @import("segment.zig").SegmentsTLD;

/// Thread-local data container
///
/// Aggregates all per-thread allocation state:
/// - `heap_backing`: Pointer to thread's heap for bin/page management
/// - `os_allocator`: Interface for requesting memory from OS (mmap)
/// - `segments`: Tracks all segments owned by this thread
///
/// ## Usage
///
/// Typically accessed through the thread-local `tld_main` variable.
/// The TLD coordinates memory requests: heap asks segments for pages,
/// segments ask os_allocator for raw memory from the kernel.
pub const TLD = struct {
    /// Back-pointer to the thread's main heap
    ///
    /// Used when pages need to reference their owning heap,
    /// for example during cross-thread free operations.
    heap_backing: ?*heap.Heap = null,

    /// OS memory interface for this thread
    ///
    /// Handles mmap/munmap calls with thread-local caching
    /// and configuration (huge pages, alignment, etc.)
    os_allocator: os_alloc.OsAllocator = .{ .config = os.mem_config_static },

    /// Segment manager for this thread
    ///
    /// Tracks all segments owned by this thread, manages
    /// page allocation within segments, and handles
    /// cross-thread segment reclamation.
    segments: SegmentsTLD = .{},
};
