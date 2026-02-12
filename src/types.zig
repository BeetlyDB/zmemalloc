//! # Core Type Definitions and Constants
//!
//! Compile-time constants defining the memory allocator layout and size classes.
//! All values are tuned for efficient memory usage on 32-bit and 64-bit platforms.
//!
//! ## Memory Hierarchy (64-bit values)
//!
//! ```
//! ┌────────────────────────────────────────────────────────────────┐
//! │ Segment (32 MiB)                                               │
//! │ ┌──────────┬──────────┬──────────┬─────┬──────────┐            │
//! │ │ Slice 0  │ Slice 1  │ Slice 2  │ ... │ Slice 511│            │
//! │ │ (64 KiB) │ (64 KiB) │ (64 KiB) │     │ (64 KiB) │            │
//! │ └──────────┴──────────┴──────────┴─────┴──────────┘            │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Object Size Classes
//!
//! | Class  | Max Size | Page Type    | Description                    |
//! |--------|----------|--------------|--------------------------------|
//! | Small  | 8 KiB    | Small (64K)  | 8+ objects per page            |
//! | Medium | 64 KiB   | Medium (512K)| 8+ objects per page            |
//! | Large  | 16 MiB   | Multi-slice  | Dedicated page(s) per object   |
//! | Huge   | > 16 MiB | Full segment | One object per segment         |
//!
//! ## Bin System
//!
//! - Bins 1-8: Exact word sizes (8, 16, 24, ... 64 bytes)
//! - Bins 9+: Logarithmic spacing (96, 128, 192, 256, ...)
//! - BIN_HUGE (73): Huge allocations marker
//! - BIN_FULL (74): Full pages bin (pages with no free blocks)
//!
//! ## Commit Mask
//!
//! Each segment has a commit mask with one bit per slice (512 bits on 64-bit).
//! This tracks which slices have been committed (backed by physical memory).

const std = @import("std");
const builtin = @import("builtin");

pub const DEBUG = builtin.mode == .Debug;
pub const SAFETY = builtin.mode == .Debug or builtin.mode == .ReleaseSafe;

pub const KiB: usize = 1024;
pub const MiB: usize = 1024 * KiB;
pub const GiB: usize = 1024 * MiB;
pub const TiB: usize = 1024 * GiB;
pub const PiB: usize = 1024 * TiB;

/// Size of pointer in bytes (4 on 32-bit, 8 on 64-bit)
pub const INTPTR_SIZE: usize = @sizeOf(usize);

/// Size of pointer in bits (32 or 64)
pub const INTPTR_BITS: usize = @bitSizeOf(usize);

/// Log2 of pointer size (2 on 32-bit, 3 on 64-bit)
pub const SIZE_SHIFT: usize = switch (INTPTR_SIZE) {
    4 => 2, // 32-bit
    8 => 3, // 64-bit
    16 => 4, // 128-bit (future)
    else => @compileError("Unsupported pointer size"),
};

/// Maximum value for usize
pub const SIZE_MAX: usize = std.math.maxInt(usize);

/// Maximum natural alignment for any type
pub const MAX_ALIGN_SIZE: usize = @alignOf(std.c.max_align_t);

// =============================================================================
// Segment Layout
//
// Segment (32 MiB on 64-bit):
// ┌─────────────────────────────────────────────────────────────────────────┐
// │ Slice 0   │ Slice 1   │ Slice 2   │ ... │ Slice 511                     │
// │ (64 KiB)  │ (64 KiB)  │ (64 KiB)  │     │ (64 KiB)                      │
// └─────────────────────────────────────────────────────────────────────────┘
//
// ============================================================================

/// Slice shift: 16 on 64-bit (64 KiB), 15 on 32-bit (32 KiB)
pub const SEGMENT_SLICE_SHIFT: usize = 13 + SIZE_SHIFT;

/// Slice size: 64 KiB on 64-bit, 32 KiB on 32-bit
pub const SEGMENT_SLICE_SIZE: usize = 1 << SEGMENT_SLICE_SHIFT;

/// Segment shift: 25 on 64-bit (32 MiB), 22 on 32-bit (4 MiB)
pub const SEGMENT_SHIFT: usize = SEGMENT_SLICE_SHIFT + if (INTPTR_SIZE > 4) 9 else 7;

/// Segment size: 32 MiB on 64-bit, 4 MiB on 32-bit
pub const SEGMENT_SIZE: usize = 1 << SEGMENT_SHIFT;

/// Segment alignment (same as size for power-of-2 alignment)
pub const SEGMENT_ALIGN: usize = SEGMENT_SIZE;

/// Mask to get offset within segment
pub const SEGMENT_MASK: usize = SEGMENT_SIZE - 1;

/// Number of slices per segment: 512 on 64-bit, 128 on 32-bit
pub const SLICES_PER_SEGMENT: usize = SEGMENT_SIZE / SEGMENT_SLICE_SIZE;

// =============================================================================
// Page Sizes
// =============================================================================

/// Small page = 1 slice (64 KiB on 64-bit)
pub const SMALL_PAGE_SHIFT: usize = SEGMENT_SLICE_SHIFT;
pub const SMALL_PAGE_SIZE: usize = 1 << SMALL_PAGE_SHIFT;

/// Medium page = 8 slices (512 KiB on 64-bit)
pub const MEDIUM_PAGE_SHIFT: usize = SMALL_PAGE_SHIFT + 3;
pub const MEDIUM_PAGE_SIZE: usize = 1 << MEDIUM_PAGE_SHIFT;

// =============================================================================
// Object Size Classes
//
// Small:  <= SMALL_PAGE_SIZE / 8   =  8 KiB on 64-bit
// Medium: <= MEDIUM_PAGE_SIZE / 8  = 64 KiB on 64-bit
// Large:  <= SEGMENT_SIZE / 2      = 16 MiB on 64-bit
// Huge:   > SEGMENT_SIZE / 2
//
// =============================================================================

/// Maximum small object size (fits 8+ per small page)
pub const SMALL_OBJ_SIZE_MAX: usize = SMALL_PAGE_SIZE / 8;

/// Maximum medium object size (fits 8+ per medium page)
pub const MEDIUM_OBJ_SIZE_MAX: usize = MEDIUM_PAGE_SIZE / 8;

/// Maximum large object size (half segment)
pub const LARGE_OBJ_SIZE_MAX: usize = SEGMENT_SIZE / 2;

/// Word-size versions (for bin calculations)
pub const SMALL_OBJ_WSIZE_MAX: usize = SMALL_OBJ_SIZE_MAX / INTPTR_SIZE;
pub const MEDIUM_OBJ_WSIZE_MAX: usize = MEDIUM_OBJ_SIZE_MAX / INTPTR_SIZE;
pub const LARGE_OBJ_WSIZE_MAX: usize = LARGE_OBJ_SIZE_MAX / INTPTR_SIZE;

/// Maximum guaranteed alignment for allocations
pub const MAX_ALIGN_GUARANTEE: usize = MEDIUM_OBJ_SIZE_MAX;

/// Maximum block alignment we support
pub const BLOCK_ALIGNMENT_MAX: usize = SEGMENT_SIZE / 2;

/// Maximum slice offset for aligned blocks
pub const MAX_SLICE_OFFSET_COUNT: usize = (BLOCK_ALIGNMENT_MAX / SEGMENT_SLICE_SIZE) - 1;

// =============================================================================
// Bin
// =============================================================================

/// Maximum word size for small allocations (direct lookup)
pub const SMALL_WSIZE_MAX: usize = 128;

/// Padding words for alignment
pub const PADDING_WSIZE: usize = if (DEBUG) 2 else 0;

/// Number of direct page lookup entries
pub const PAGES_DIRECT: usize = SMALL_WSIZE_MAX + 1;

/// Number of size class bins (excluding huge)
pub const BIN_HUGE: usize = 73;

/// Total bins including full pages bin
pub const BIN_FULL: usize = BIN_HUGE + 1;

/// Maximum segment bins for TLD spans
pub const SEGMENTS_BIN_MAX: usize = 35;

// =============================================================================
// Commit
// =============================================================================

/// Minimum commit granularity
pub const MINIMAL_COMMIT_SIZE: usize = SEGMENT_SLICE_SIZE;

/// Commit unit size (same as slice)
pub const COMMIT_SIZE: usize = SEGMENT_SLICE_SIZE;

/// Bits in commit mask (512 on 64-bit = one bit per slice)
pub const COMMIT_MASK_BITS: usize = SEGMENT_SIZE / COMMIT_SIZE;

/// Bits per mask field
pub const COMMIT_MASK_FIELD_BITS: usize = INTPTR_BITS;

/// Number of fields in commit mask array
pub const COMMIT_MASK_FIELD_COUNT: usize = COMMIT_MASK_BITS / COMMIT_MASK_FIELD_BITS;

// =============================================================================
// Allocation
// =============================================================================

/// Maximum allocation size we can handle
pub const MAX_ALLOC_SIZE: usize = blk: {
    if (INTPTR_BITS > 32) {
        // 64-bit: limited by slice count fitting in u32
        break :blk SEGMENT_SLICE_SIZE * (std.math.maxInt(u32) - 1);
    } else {
        // 32-bit: limited by address space
        break :blk SIZE_MAX;
    }
};

comptime {
    // Verify segment layout
    if (SEGMENT_SIZE != SLICES_PER_SEGMENT * SEGMENT_SLICE_SIZE) {
        @compileError("Segment size mismatch");
    }

    // Verify commit mask fits evenly
    if (COMMIT_MASK_BITS != COMMIT_MASK_FIELD_COUNT * COMMIT_MASK_FIELD_BITS) {
        @compileError("Commit mask bits must be exactly divisible by field bits");
    }

    // Verify object size hierarchy
    if (SMALL_OBJ_SIZE_MAX >= MEDIUM_OBJ_SIZE_MAX) {
        @compileError("Small objects must be smaller than medium");
    }
    if (MEDIUM_OBJ_SIZE_MAX >= LARGE_OBJ_SIZE_MAX) {
        @compileError("Medium objects must be smaller than large");
    }

    // Verify page sizes
    if (SMALL_PAGE_SIZE != SEGMENT_SLICE_SIZE) {
        @compileError("Small page must equal slice size");
    }
    if (MEDIUM_PAGE_SIZE != 8 * SMALL_PAGE_SIZE) {
        @compileError("Medium page must be 8 small pages");
    }

    // Verify 64-bit expected values
    if (INTPTR_SIZE == 8) {
        if (SEGMENT_SIZE != 32 * MiB) @compileError("Expected 32 MiB segments on 64-bit");
        if (SEGMENT_SLICE_SIZE != 64 * KiB) @compileError("Expected 64 KiB slices on 64-bit");
        if (SLICES_PER_SEGMENT != 512) @compileError("Expected 512 slices per segment on 64-bit");
        if (SMALL_OBJ_SIZE_MAX != 8 * KiB) @compileError("Expected 8 KiB max small on 64-bit");
        if (MEDIUM_OBJ_SIZE_MAX != 64 * KiB) @compileError("Expected 64 KiB max medium on 64-bit");
    }
}

// =============================================================================
// Tests
// =============================================================================

test "types: size constants" {
    const testing = std.testing;

    try testing.expectEqual(@as(usize, 1024), KiB);
    try testing.expectEqual(@as(usize, 1024 * 1024), MiB);
    try testing.expectEqual(@as(usize, 1024 * 1024 * 1024), GiB);
}

test "types: segment layout on 64-bit" {
    if (INTPTR_SIZE != 8) return error.SkipZigTest;

    const testing = std.testing;

    try testing.expectEqual(@as(usize, 32 * MiB), SEGMENT_SIZE);
    try testing.expectEqual(@as(usize, 64 * KiB), SEGMENT_SLICE_SIZE);
    try testing.expectEqual(@as(usize, 512), SLICES_PER_SEGMENT);
    try testing.expectEqual(@as(usize, 64 * KiB), SMALL_PAGE_SIZE);
    try testing.expectEqual(@as(usize, 512 * KiB), MEDIUM_PAGE_SIZE);
}

test "types: object size limits on 64-bit" {
    if (INTPTR_SIZE != 8) return error.SkipZigTest;

    const testing = std.testing;

    try testing.expectEqual(@as(usize, 8 * KiB), SMALL_OBJ_SIZE_MAX);
    try testing.expectEqual(@as(usize, 64 * KiB), MEDIUM_OBJ_SIZE_MAX);
    try testing.expectEqual(@as(usize, 16 * MiB), LARGE_OBJ_SIZE_MAX);
}

test "types: commit mask" {
    const testing = std.testing;

    // Commit mask should have one bit per slice
    try testing.expectEqual(SLICES_PER_SEGMENT, COMMIT_MASK_BITS);

    // Field count should divide evenly
    try testing.expectEqual(@as(usize, 0), COMMIT_MASK_BITS % COMMIT_MASK_FIELD_BITS);
}
