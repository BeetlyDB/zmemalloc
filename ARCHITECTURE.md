# zmemalloc Architecture

A high-performance memory allocator inspired by mimalloc, written in Zig.

## Overview

zmemalloc is a thread-local memory allocator using segment-based memory management. Designed for high throughput and low fragmentation in multi-threaded applications.

**Key optimizations:**
- 64-byte Page struct (fits in single cache line)
- Three-level free list sharding (free, local_free, xthread_free)
- Bump pointer allocation for fresh pages
- Lock-free cross-thread free
- Comptime page shifts 

## Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEMORY HIERARCHY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Application                                                               │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                               │
│   │ Thread 1│     │ Thread 2│     │ Thread N│                               │
│   └────┬────┘     └────┬────┘     └────┬────┘                               │
│        │               │               │                                    │
│        ▼               ▼               ▼                                    │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                               │
│   │  Heap   │     │  Heap   │     │  Heap   │   Thread-Local Heaps          │
│   │ (bins)  │     │ (bins)  │     │ (bins)  │                               │
│   └────┬────┘     └────┬────┘     └────┬────┘                               │
│        │               │               │                                    │
│        ▼               ▼               ▼                                    │
│   ┌─────────────────────────────────────────┐                               │
│   │              SegmentsTLD                │   Per-thread segment cache    │
│   │  small_free | medium_free | large_free  │                               │
│   └─────────────────────────────────────────┘                               │
│                        │                                                    │
│        ┌───────────────┼───────────────┐                                    │
│        ▼               ▼               ▼                                    │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                                 │
│   │ Segment │    │ Segment │    │ Segment │    32MB aligned                 │
│   │ (small) │    │ (medium)│    │ (large) │                                 │
│   │ 64KB pg │    │ 512KB pg│    │ full pg │                                 │
│   └────┬────┘    └────┬────┘    └────┬────┘                                 │
│        │              │              │                                      │
│        ▼              ▼              ▼                                      │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐                                   │
│   │  Pages  │   │  Pages  │   │  Page   │   Fixed-size blocks               │
│   │ blocks  │   │ blocks  │   │ block   │                                   │
│   └─────────┘   └─────────┘   └─────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Data Structures

### Page (64 bytes - single cache line)

```zig
Page = struct {
    // Hot path fields (32 bytes):
    free: IntrusiveLifo(Block),     // 8: recycled blocks
    page_start: ?[*]u8,             // 8: start of data area
    block_size: usize,              // 8: size of each block
    capacity: u16,                  // 2: max blocks
    reserved: u16,                  // 2: bump pointer position
    used: u16,                      // 2: blocks in use
    flags_and_idx: u16,             // 2: packed flags + segment_idx

    // Queue fields (32 bytes):
    local_free: IntrusiveLifo(Block), // 8: same-thread freed blocks
    xthread_free: Atomic(?*Block),    // 8: cross-thread freed (atomic)
    next: ?*Page,                     // 8: bin queue link
    prev: ?*Page,                     // 8: bin queue link
}

// Packed flags (6 bits) + segment_idx (10 bits) in u16:
PackedInfo = packed struct(u16) {
    in_full: bool,        // page is full, not in bin queue
    in_bin: bool,         // page is in heap's bin queue
    is_huge: bool,        // huge allocation (> 16MB)
    segment_in_use: bool, // page claimed by segment
    is_commited: bool,    // memory is committed
    is_zero_init: bool,   // memory is zeroed
    segment_idx: u10,     // index in segment.pages[]
}
```

### Segment (32MB aligned)

```
SEGMENT LAYOUT (32MB)
┌─────────────────────────────────────────────────────────────────┐
│                    Segment Header (~40KB)                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ memid, segment_size, page_kind, capacity, used              ││
│  │ thread_id (cache-line aligned for cross-thread check)       ││
│  │ pages[512] - array of 64-byte Page structs                  ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                       Page Data Area                            │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐                        │
│  │Page 0 │ │Page 1 │ │Page 2 │ │  ...  │  (64KB or 512KB each)  │
│  │ data  │ │ data  │ │ data  │ │       │                        │
│  └───────┘ └───────┘ └───────┘ └───────┘                        │
└─────────────────────────────────────────────────────────────────┘

Pointer → Segment:  segment = ptr & ~(32MB - 1)
Pointer → Page:     idx = (ptr - segment) >> page_shift
```

### Block (intrusive free list)

```
Allocated:                    Free:
┌──────────────────────┐     ┌──────────────────────┐
│                      │     │ link.next ──────────►│ next free block
│     User Data        │     │                      │
│                      │     │    (unused space)    │
└──────────────────────┘     └──────────────────────┘

When free: first 8 bytes = pointer to next free block
When allocated: user owns all memory
```

## Allocation Algorithm

### Fast Path (< 1KB) - O(1)

```
malloc(size):
    wsize = (size + 7) / 8
    page = heap.pages_free_direct[wsize]
    if page:
        block = page.free.pop()      ──► return block
        OR
        block = bump_allocate()       ──► return block
```

### Medium Path (bin queue lookup) - O(1)

```
    bin = binFromSize(size)
    page = heap.pages[bin].tail
    if page and page.hasFree():
        block = page.popFreeBlock()
        if page exhausted:
            remove from bin queue
        return block
```

### Slow Path (new page allocation) - O(n)

```
    1. Try purge expired pages (if memory pressure)
    2. Try reclaim abandoned segments
    3. Allocate page:
       a. Try existing segment from free queue
       b. Allocate new segment from arena/OS
    4. Initialize page (set page_start, capacity)
    5. Add page to bin queue
    6. return page.popFreeBlock()
```

### Page Block Allocation (popFreeBlock)

```
popFreeBlock():
    // Hot path 1: recycled blocks
    if free.pop():
        used++
        return block

    // Hot path 2: bump allocation (contiguous, cache-friendly)
    if reserved < capacity:
        block = page_start + reserved * block_size
        reserved++
        used++
        return block

    // Cold path: collect from other lists
    return popFreeBlockSlow()

popFreeBlockSlow():
    // Try local_free (same-thread freed blocks)
    if !local_free.empty():
        swap(free, local_free)
        return free.pop()

    // Try xthread_free (cross-thread freed blocks)
    if collectXthreadFree() > 0:
        return free.pop()

    // Final fallback: bump allocate
    if reserved < capacity:
        return bump_allocate()

    return null
```

## Free Algorithm

### Same-Thread Free - O(1)

```
free(ptr):
    segment = ptr & ~SEGMENT_MASK
    page = segment.pageFromPtr(ptr)

    if segment.thread_id == current_thread:
        page.local_free.push(block)
        page.used--
```

### Cross-Thread Free - O(1) lock-free

```
    else:
        // Atomic CAS loop
        loop:
            old = page.xthread_free.load()
            block.next = old
            if CAS(page.xthread_free, old, block):
                break
```

### Cross-Thread Collection (batch splice)

```
collectXthreadFree():
    head = xthread_free.swap(null)    // atomic
    if !head: return 0

    // Find tail and count
    tail = head
    count = 1
    while tail.next:
        tail = tail.next
        count++

    // Splice entire list to free in O(1)
    tail.next = free.head
    free.head = head

    used -= count
    return count
```

## Page Types

| Type | Page Size | Block Size | Pages/Segment |
|------|-----------|------------|---------------|
| Small | 64 KB | 8 - 1024 B | ~500 |
| Medium | 512 KB | 1KB - 128KB | ~60 |
| Large | Full segment | 128KB - 16MB | 1 |
| Huge | Variable | > 16MB | 1 (custom segment) |

## Size Class Binning

```
wsize = (size + 7) / 8    // size in 8-byte words

Bin assignment:
  wsize 1-8:  bin = wsize        (linear: 8, 16, 24, 32, 40, 48, 56, 64 bytes)
  wsize > 8:  bin = logarithmic  (exponential growth)

Total: 74 bins (0-73, where 73 = BIN_HUGE)
```

## Memory Reclamation

### Implicit Page Retirement
Pages stay in bin queue even when `used == 0`. Only removed during:
- `collectUnusedMemory()` - explicit collection
- Memory pressure - when segment count > threshold

### Purge Delay (MADV_DONTNEED)
Unused page memory marked with `madvise(MADV_DONTNEED)`:
- OS can reclaim physical memory if needed
- If page reused quickly, memory still available
- Delayed purge check during allocation (time-based)

### Segment Lifecycle

```
    NEW ───► ACTIVE ───► has free pages ───► in free queue
                │                                  │
                │ all pages freed                  │
                ▼                                  ▼
             EMPTY ◄───────────────────────────────┘
                │
                │ (memory pressure OR force)
                ▼
           FREE (returned to OS/arena)
```

## Thread Safety

| Operation | Mechanism |
|-----------|-----------|
| Same-thread alloc/free | No synchronization needed |
| Cross-thread free | Lock-free atomic CAS on xthread_free |
| Segment reclaim | Atomic CAS on thread_id |
| Abandoned segments | Global lock for queue operations |

```

## Configuration Constants

```zig
SEGMENT_SIZE        = 32 MiB
SEGMENT_SHIFT       = 25
SMALL_PAGE_SIZE     = 64 KiB
SMALL_PAGE_SHIFT    = 16
MEDIUM_PAGE_SIZE    = 512 KiB
MEDIUM_PAGE_SHIFT   = 19

SMALL_OBJ_SIZE_MAX  = 1 KiB
MEDIUM_OBJ_SIZE_MAX = 128 KiB
LARGE_OBJ_SIZE_MAX  = 16 MiB

BIN_HUGE            = 73
SLICES_PER_SEGMENT  = 512
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| malloc (fast) | O(1) | direct page + pop/bump |
| malloc (medium) | O(1) | bin queue + pop/bump |
| malloc (slow) | O(n) | segment allocation |
| free (same-thread) | O(1) | push to local_free |
| free (cross-thread) | O(1) | atomic CAS |
| fromPtr | O(1) | bitmask (huge: registry lookup) |

**Cache optimizations:**
- Page struct = 64 bytes = 1 cache line
- Hot fields in first 32 bytes
- Bump allocation = sequential memory access
- Prefetch hints for block access
