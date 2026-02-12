# zmemalloc Architecture

A high-performance memory allocator inspired by mimalloc, written in Zig.

## Overview

zmemalloc is a thread-local memory allocator using segment-based memory management. Designed for high throughput and low fragmentation in multi-threaded applications.

**Key optimizations:**
- 64-byte Page struct (fits in single cache line)
- Three-level free list sharding (free, local_free, xthread_free)
- Bump pointer allocation for fresh pages
- Lock-free cross-thread free
- Direct pointer array for O(1) small allocations
- Automatic page recycling via full->non-full transitions

## Memory Hierarchy

```
+-----------------------------------------------------------------------------+
|                           MEMORY HIERARCHY                                  |
+-----------------------------------------------------------------------------+
|                                                                             |
|   Application                                                               |
|       |                                                                     |
|       v                                                                     |
|   +----------+     +----------+     +----------+                            |
|   | Thread 1 |     | Thread 2 |     | Thread N |                            |
|   +----+-----+     +----+-----+     +----+-----+                            |
|        |                |                |                                  |
|        v                v                v                                  |
|   +----------+     +----------+     +----------+                            |
|   |   Heap   |     |   Heap   |     |   Heap   |   Thread-Local Heaps       |
|   |  (bins)  |     |  (bins)  |     |  (bins)  |                            |
|   +----+-----+     +----+-----+     +----+-----+                            |
|        |                |                |                                  |
|        v                v                v                                  |
|   +---------------------------------------------+                           |
|   |              SegmentsTLD                    |   Per-thread segment      |
|   |  small_free | medium_free | large_free      |   cache                   |
|   +---------------------------------------------+                           |
|                        |                                                    |
|        +---------------+---------------+                                    |
|        v               v               v                                    |
|   +----------+   +----------+   +----------+                                |
|   | Segment  |   | Segment  |   | Segment  |    32MB aligned                |
|   | (small)  |   | (medium) |   | (large)  |                                |
|   | 64KB pg  |   | 512KB pg |   | full pg  |                                |
|   +----+-----+   +----+-----+   +----+-----+                                |
|        |              |              |                                      |
|        v              v              v                                      |
|   +----------+  +----------+  +----------+                                  |
|   |  Pages   |  |  Pages   |  |   Page   |   Fixed-size blocks              |
|   | blocks   |  | blocks   |  |  block   |                                  |
|   +----------+  +----------+  +----------+                                  |
|                                                                             |
+-----------------------------------------------------------------------------+
```

## Core Data Structures

### Page (64 bytes - single cache line)

```zig
Page = struct {
    // Hot path fields (32 bytes):
    free_head: ?*anyopaque,         // 8: head of free list (linked blocks)
    page_start: ?[*]u8,             // 8: start of data area
    block_size: usize,              // 8: size of each block
    capacity: u16,                  // 2: max blocks
    reserved: u16,                  // 2: bump pointer position
    used: u16,                      // 2: blocks in use
    flags_and_idx: u16,             // 2: packed flags + segment_idx

    // Queue fields (32 bytes):
    local_free_head: ?*anyopaque,   // 8: same-thread freed blocks
    xthread_free: Atomic(?*Block),  // 8: cross-thread freed (atomic)
    next: ?*Page,                   // 8: bin queue link
    prev: ?*Page,                   // 8: bin queue link
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
+-----------------------------------------------------------------+
|                    Segment Header (~40KB)                       |
|  +------------------------------------------------------------+ |
|  | memid, segment_size, page_kind, capacity, used             | |
|  | thread_id (cache-line aligned for cross-thread check)      | |
|  | heap (pointer to owning heap)                              | |
|  | all_next, all_prev (for xthread_free collection)           | |
|  | pages[512] - array of 64-byte Page structs                 | |
|  +------------------------------------------------------------+ |
+----------------------------------------------------------------+
|                       Page Data Area                            |
|  +--------+ +--------+ +--------+ +--------+                    |
|  | Page 0 | | Page 1 | | Page 2 | |  ...   |  (64KB or 512KB)   |
|  |  data  | |  data  | |  data  | |        |                    |
|  +--------+ +--------+ +--------+ +--------+                    |
+----------------------------------------------------------------+

Pointer -> Segment:  segment = ptr & ~(32MB - 1)
Pointer -> Page:     idx = (ptr - segment) >> page_shift
```

### Block (intrusive free list)

```
Allocated:                    Free:
+----------------------+     +----------------------+
|                      |     | next ---------------+---> next free block
|     User Data        |     |                      |
|                      |     |    (unused space)    |
+----------------------+     +----------------------+

When free: first 8 bytes = pointer to next free block
When allocated: user owns all memory
```

### Heap (per-thread)

```zig
Heap = struct {
    pages_free_direct: [129]?*Page,  // Direct pointers for wsize 0-128
    pages: [74]Page.Queue,           // Bin queues for size classes
    tld: ?*TLD,                      // Thread-local data
    thread_id: usize,
}
```

## Allocation Flow

### Overview

```
malloc(size)
    |
    v
+-------------------+
| size <= 1KB?      |--NO--> mallocSlowPath()
+-------------------+
    |YES
    v
+-------------------+
| mallocSmallFast() |
| pages_free_direct |
| [wsize] lookup    |
+-------------------+
    |
    +-- page found --> popFreeBlock() --> SUCCESS
    |
    +-- page null --> mallocSlowPath()
    |
    +-- page exhausted --> mark full, remove from bin --> mallocSlowPath()
```

### Fast Path: mallocSmallFast (< 1KB) - O(1)

This is the hottest path, optimized for minimal instructions:

```zig
fn mallocSmallFast(heap: *Heap, size: usize) ?[*]u8 {
    // 1. Calculate word size (8-byte granularity)
    const wsize = (size + 7) / 8;

    // 2. Direct O(1) lookup - no bin calculation needed
    const page = heap.pages_free_direct[wsize] orelse return null;

    // 3. Try to pop from free list or bump allocate
    if (page.popFreeBlock()) |block| {
        return @ptrCast(block);  // SUCCESS - fast path complete
    }

    // 4. Page exhausted - mark as full, remove from bin queue
    //    This prevents slow path from trying the same page
    page.set_in_full(true);
    page.pageRemoveFromBin(heap, bin);
    heap.pages_free_direct[wsize] = null;
    return null;  // Fall through to slow path
}
```

**Key insight:** The `pages_free_direct` array provides O(1) lookup without bin calculation. Each entry points directly to a page that can satisfy allocations of that word size.

### Slow Path: mallocSlowPath

Called when fast path fails (no page or page exhausted):

```zig
fn mallocSlowPath(heap: *Heap, size: usize) ?[*]u8 {
    // 1. Try medium path (bin queue lookup)
    if (size <= MEDIUM_OBJ_SIZE_MAX) {
        if (mallocMedium(heap, size)) |ptr| {
            return ptr;
        }
    }

    // 2. Fall back to generic allocation (may allocate new page)
    return mallocGeneric(heap, size, false);
}
```

### Medium Path: mallocMedium - O(1)

```zig
fn mallocMedium(heap: *Heap, size: usize) ?[*]u8 {
    const bin = binFromSize(size);
    const page = heap.pages[bin].tail orelse return null;

    // Try to allocate from this page
    if (tryAllocFromPage(heap, page, bin)) |ptr| {
        return ptr;
    }

    return null;
}
```

### Generic Path: mallocGeneric - O(n)

Allocates new pages when needed:

```zig
fn mallocGeneric(heap: *Heap, size: usize, zero: bool) ?[*]u8 {
    // 1. Collect pending cross-thread frees if threshold exceeded
    if (pending_xthread_free > threshold) {
        collectAllXthreadFree();
        reclaimFullPages();
    }

    // 2. Try bin queue again
    if (bin queue has page with free blocks) {
        return allocate from it;
    }

    // 3. Allocate new page from segment
    page = segments.allocPage(block_size);

    // 4. Initialize page (set page_start, capacity)
    page.init(block_size, page_start, page_size);

    // 5. Add to bin queue and set direct pointers
    page.pagePushBin(heap, bin);
    setDirectPointerForBlockSize(heap, page);

    // 6. Allocate block
    return page.popFreeBlock();
}
```

### Direct Pointer Management

**Critical for performance:** When a page is added to a bin, we set direct pointers for ALL word sizes that map to that bin:

```zig
fn setDirectPointerForBlockSize(heap: *Heap, page: *Page) void {
    const block_wsize = (page.block_size + 7) / 8;

    // For small bins (1-8), wsize maps 1:1 to bin
    heap.pages_free_direct[block_wsize] = page;

    // For larger bins, multiple wsizes map to same bin
    // Set direct pointers for ALL of them
    if (bin > 8) {
        // Find range of wsizes that map to this bin
        for (wsize in range that maps to this bin) {
            heap.pages_free_direct[wsize] = page;
        }
    }
}
```

**Why this matters:** Without this, allocations of different sizes in the same bin would miss the direct pointer and fall through to slow path. This was a major performance bug - 83% of allocations were hitting slow path unnecessarily!

### Page Block Allocation: popFreeBlock

```zig
fn popFreeBlock(self: *Page) ?*Block {
    // Hot path 1: pop from free list (recycled blocks)
    if (self.popFromFreeList()) |block| {
        self.used += 1;
        return block;
    }

    // Hot path 2: bump allocation (contiguous, cache-friendly)
    if (self.reserved < self.capacity) {
        block = self.page_start + self.reserved * self.block_size;
        self.reserved += 1;
        self.used += 1;
        return block;
    }

    // Cold path: collect from local_free/xthread_free
    return self.popFreeBlockSlow();
}

fn popFreeBlockSlow(self: *Page) ?*Block {
    // Try local_free (same-thread freed blocks)
    if (!self.localFreeEmpty()) {
        self.swapFreeLists();  // local_free becomes free
        return self.popFromFreeList();
    }

    // Try xthread_free (cross-thread freed blocks)
    if (self.collectXthreadFree() > 0) {
        return self.popFromFreeList();
    }

    // Final fallback: bump allocate
    if (self.reserved < self.capacity) {
        return bump_allocate();
    }

    return null;
}
```

## Free Flow

### Overview

```
free(ptr)
    |
    v
+-------------------+
| Find segment/page |  segment = ptr & ~(32MB-1)
| from pointer      |  page = segment.pageFromPtr(ptr)
+-------------------+
    |
    v
+-------------------+
| Same thread?      |--NO--> xthread_free (atomic CAS)
+-------------------+
    |YES
    v
+-------------------+
| Push to           |
| local_free        |
+-------------------+
    |
    v
+-------------------+
| Page was full?    |--NO--> DONE
+-------------------+
    |YES
    v
+-------------------+
| freeColdPath()    |  Re-add page to bin queue
+-------------------+
```

### Same-Thread Free - O(1)

```zig
fn freeImpl(ptr: ?*anyopaque) void {
    const segment = Segment.fromPtr(ptr);
    const page = Segment.pageFromPtrWithSegment(segment, ptr);
    const block: *Block = @ptrCast(ptr);

    if (segment.thread_id == currentThreadId()) {
        // Hot path: same thread
        page.pushFreeBlock(block);  // Push to local_free, decrement used

        // Handle full -> non-full transition
        if (page.is_in_full()) {
            freeColdPath(page);  // Re-add to bin queue
        }
    } else {
        // Cross-thread free (see below)
        page.xthreadFree(block);
    }
}
```

### Cross-Thread Free - O(1) lock-free

```zig
fn xthreadFree(self: *Page, block: *Block) void {
    while (true) {
        const old_head = self.xthread_free.load(.acquire);
        block.link.next = old_head;

        if (self.xthread_free.cmpxchgWeak(old_head, block, .release, .acquire) == null) {
            // Success - increment global pending counter
            pending_xthread_free.fetchAdd(1, .monotonic);
            break;
        }
        // CAS failed, retry
    }
}
```

### Page Reactivation: freeColdPath

**Critical for page reuse:** When a full page receives a free, it must be re-added to the bin queue:

```zig
fn freeColdPath(page: *Page) void {
    const bin = binFromSize(page.block_size);

    // Page is completely empty - return to segment
    if (page.used == 0) {
        page.set_in_full(false);
        segments.freePage(page);
        return;
    }

    // Page has space now - re-add to bin queue for allocation
    page.set_in_full(false);
    if (!page.is_in_bin()) {
        page.pagePushBin(heap, bin);
        setDirectPointerForBlockSize(heap, page);  // Set ALL direct pointers!
    }
}
```

### Cross-Thread Collection (batch splice)

Collected during allocation when threshold exceeded:

```zig
fn collectXthreadFree(self: *Page) usize {
    // Atomic swap - get entire list at once
    const head = self.xthread_free.swap(null, .acquire) orelse return 0;

    // Find tail and count in single pass
    var tail = head;
    var count: usize = 1;
    while (tail.link.next) |next| {
        tail = next;
        count += 1;
    }

    // Splice entire list to free in O(1)
    tail.link.next = self.free_head;
    self.free_head = head;

    self.used -= count;
    pending_xthread_free.fetchSub(count, .monotonic);
    return count;
}
```

## Page Lifecycle

```
                    +---------------------------+
                    |                           |
                    v                           |
+-------+      +---------+      +----------+    |
| FRESH | ---> | IN_BIN  | ---> |   FULL   | ---+
+-------+      +---------+      +----------+
   |                |                |
   |                |                | free() called
   |                |                v
   |                |          +----------+
   |                +--------> | REUSED   | (back to IN_BIN)
   |                           +----------+
   |                                |
   |                                | used == 0
   v                                v
+-------+                      +---------+
| FREED |  <------------------ | EMPTY   |
+-------+                      +---------+
```

**State transitions:**
1. **FRESH -> IN_BIN**: Page allocated, added to bin queue, direct pointers set
2. **IN_BIN -> FULL**: All blocks allocated, removed from bin queue, marked `in_full`
3. **FULL -> REUSED**: Block freed via `freeColdPath`, re-added to bin queue
4. **IN_BIN/REUSED -> EMPTY**: All blocks freed (`used == 0`)
5. **EMPTY -> FREED**: Page returned to segment

## Memory Deallocation Hierarchy

Memory flows back through the hierarchy in stages:

```
Block → Page → Segment → Arena/OS

free(ptr)
    |
    v
+------------------+
| Push to          |  Block added to page's free list
| page.local_free  |  (or xthread_free if cross-thread)
+------------------+
    |
    | page.used == 0
    v
+------------------+
| freePage()       |  Page cleared, returned to segment
+------------------+
    |
    | segment.used == 0
    v
+------------------+
| freeSegment()    |  Segment returned to arena or OS
+------------------+
    |
    v
+------------------+     +------------------+
| Arena free()     | OR  | OS munmap()      |
| (bitmap update)  |     | (physical free)  |
+------------------+     +------------------+
    |
    | tryPurge()
    v
+------------------+
| MADV_FREE        |  Physical pages released to OS
+------------------+
```

### freePage

Called when a page becomes completely empty (`used == 0`):

```zig
fn freePage(self: *SegmentsTLD, pg: *Page, force: bool) void {
    const segment = Segment.fromPtr(pg);
    segment.pageClear(pg);  // Reset all page fields
    segment.used -= 1;

    if (segment.used == 0) {
        if (force) {
            // Forced free - return segment immediately
            self.freeSegment(segment, true);
        } else if (self.empty_segment_count >= EMPTY_SEGMENT_CACHE_MAX * 3) {
            // Too many empty segments cached
            self.freeSegment(segment, true);
        } else {
            // Cache empty segment for reuse
            self.empty_segment_count += 1;
            self.insertInFreeQueue(segment);
        }
    } else if (was_full) {
        // Segment was full, now has free space
        self.insertInFreeQueue(segment);
    }
}
```

**pageClear** resets all page fields:
```zig
fn pageClear(self: *Segment, pg: *Page) void {
    pg.flags_and_idx = @bitCast(PackedInfo{ .segment_idx = pg.segment_idx() });
    pg.xthread_free.store(null, .release);
    pg.free_head = null;
    pg.local_free_head = null;
    pg.block_size = 0;
    pg.page_start = null;
    pg.capacity = 0;
    pg.reserved = 0;
    pg.used = 0;
    pg.next = null;
    pg.prev = null;
}
```

### freeSegment

Called when a segment becomes completely empty:

```zig
fn freeSegment(self: *SegmentsTLD, segment: *Segment, force: bool) void {
    // Remove from tracking lists
    self.removeFromFreeQueue(segment);
    self.all_segments.remove(segment);

    // Purge uncommitted pages (MADV_FREE)
    if (force and !segment.memid.flags.is_pinned) {
        for (segment.pages[0..segment.capacity]) |*pg| {
            if (!pg.is_segment_in_use() and pg.is_commited()) {
                segment.pagePurge(pg);
            }
        }
    }

    segment.thread_id.store(0, .release);

    // Return memory based on source
    if (segment.memid.memkind == .MEM_ARENA) {
        // Return to arena (bitmap update, no syscall)
        const arena = getArena(segment.memid.arena_id);
        arena.free(block_index, block_count, initially_committed);
    } else {
        // Direct OS free (munmap syscall)
        os_alloc.free(segment[0..segment_size]);
    }
}
```

### pagePurge

Releases physical memory while keeping virtual address mapping:

```zig
fn pagePurge(self: *Segment, pg: *Page) void {
    if (!self.allow_purge) return;
    if (!pg.is_commited()) return;

    const start = self.rawPageStart(pg, &psize);

    // MADV_FREE: Mark pages as reusable
    // Kernel can reclaim physical memory when needed
    // Faster than MADV_DONTNEED (lazy decommit)
    std.posix.madvise(start, psize, MADV.FREE);
    pg.set_commited(false);
}
```

## OS Memory Operations

| Operation | Syscall | Effect |
|-----------|---------|--------|
| **commit** | `mprotect(RW)` | Make memory accessible |
| **decommit** | `madvise(MADV_DONTNEED)` | Release physical pages immediately |
| **purge/reset** | `madvise(MADV_FREE)` | Mark pages as reusable (lazy free) |
| **protect** | `mprotect(NONE)` | Make memory inaccessible (guard pages) |

### MADV_FREE vs MADV_DONTNEED

```
MADV_FREE (preferred):
  - Lazy decommit: kernel reclaims when under memory pressure
  - If memory accessed before reclaim: data preserved (no page fault)
  - Lower syscall overhead
  - Used for page purge

MADV_DONTNEED:
  - Immediate decommit: RSS decreases immediately
  - Next access causes page fault + zero-fill
  - Used for arena block decommit
  - Fallback if MADV_FREE not supported
```

### Fallback Mechanism

```zig
var reset_advice: Atomic(u32) = MADV.FREE;

fn prim_reset(mem: []u8) !void {
    const advice = reset_advice.load(.monotonic);
    const result = madvise(mem, advice);

    if (result == EINVAL and advice == MADV.FREE) {
        // MADV_FREE not supported, fall back to DONTNEED
        reset_advice.store(MADV.DONTNEED, .release);
        return madvise(mem, MADV.DONTNEED);
    }
}
```

## Arena Purge System

Arenas use deferred purge for efficiency:

```zig
Arena = struct {
    blocks_inuse: DynamicBitmap,     // Which blocks are allocated
    blocks_dirty: DynamicBitmap,     // Which blocks have been used
    blocks_purge: DynamicBitmap,     // Which blocks need purging
    purge_expire: Atomic(i64),       // When to execute purge
}
```

### Purge Flow

```
Segment freed to arena
    |
    v
+------------------+
| arena.free()     |  Mark blocks in blocks_purge bitmap
| Set purge_expire |  Schedule purge for later (delay = 10ms)
+------------------+
    |
    | Time passes...
    v
+------------------+
| tryPurge()       |  Called during allocation
| if now > expire  |
+------------------+
    |
    v
+------------------+
| For each bit in  |  Scan blocks_purge bitmap
| blocks_purge:    |
|   MADV_DONTNEED  |  Decommit physical memory
|   Clear bit      |
+------------------+
```

### tryPurgeDelayed (Throttled)

Called during `mallocGeneric` to avoid syscall overhead:

```zig
const PURGE_DELAY_MS: i64 = 10;
var last_purge_check: i64 = 0;

fn tryPurgeDelayed() void {
    const now = std.time.milliTimestamp();
    if (now - last_purge_check < PURGE_DELAY_MS) return;
    last_purge_check = now;

    // 1. Free empty segments from cache
    tld.segments.collect(false);

    // 2. Decommit freed arena blocks
    globalArenas().tryPurge(false);
}
```

## Segment Caching

Empty segments are cached for reuse instead of immediately freed:

```zig
const SEGMENT_CACHE_THRESHOLD: usize = 8;    // Start freeing after 8 segments
const EMPTY_SEGMENT_CACHE_MAX: usize = 2;    // Keep up to 2 empty per queue

// Three queues by page kind:
SegmentsTLD = struct {
    small_free: SegmentQueue,   // Segments with 64KB pages
    medium_free: SegmentQueue,  // Segments with 512KB pages
    large_free: SegmentQueue,   // Segments with full pages
    empty_segment_count: usize, // Total empty segments cached
}
```

**Caching benefits:**
- Avoids expensive `mmap`/`munmap` syscalls
- Segments already have page tables set up
- Quick reuse for same-sized allocations

**Eviction policy:**
- When `empty_segment_count >= EMPTY_SEGMENT_CACHE_MAX * 3` (6)
- AND `total_segments > SEGMENT_CACHE_THRESHOLD` (8)
- → Free segment immediately instead of caching

## Page Types

| Type | Page Size | Block Size | Pages/Segment | Use Case |
|------|-----------|------------|---------------|----------|
| Small | 64 KB | 8 - 1024 B | ~500 | Most allocations |
| Medium | 512 KB | 1KB - 128KB | ~60 | Medium objects |
| Large | Full segment | 128KB - 16MB | 1 | Large objects |
| Huge | Variable | > 16MB | 1 (custom) | Huge allocations |

## Size Class Binning

```
wsize = (size + 7) / 8    // size in 8-byte words

Bin assignment:
  wsize 1-8:  bin = wsize        (linear: 8, 16, 24, 32, 40, 48, 56, 64 bytes)
  wsize > 8:  bin = logarithmic  (exponential growth)

Total: 74 bins (0-73, where 73 = BIN_HUGE)

Direct pointer array: 129 entries (wsize 0-128)
  - Provides O(1) lookup for sizes up to 1024 bytes
  - Multiple wsizes may point to same page (same bin)
```

## Thread Safety

| Operation | Mechanism |
|-----------|-----------|
| Same-thread alloc/free | No synchronization needed |
| Cross-thread free | Lock-free atomic CAS on xthread_free |
| Segment reclaim | Atomic CAS on thread_id |
| Abandoned segments | Global lock for queue operations |

## Thread Exit and Abandoned Segments

When a thread exits without calling `threadExit()`, its segments become orphaned. The allocator handles this through lazy reclamation.

### Global Abandoned List

```zig
// In Subproc (global process state)
Subproc = struct {
    abandoned_os_list: SegmentAbandonedQueue,  // Queue of abandoned segments
    abandoned_os_list_count: Atomic(usize),     // Count for quick check
    abandoned_os_lock: Mutex,                   // Protects list operations
}
```

**Flow:**
```
Thread Exit (without cleanup)
    |
    v
+-------------------+
| Segment still has |
| thread_id set     |
+-------------------+
    |
    | (Another thread tries cross-thread free)
    v
+-------------------+
| Detects orphaned  |---> markAbandoned() sets thread_id = 0
| segment           |---> Adds to abandoned_os_list
+-------------------+
    |
    v
+-------------------+
| reclaimAbandoned()|  Called during allocation when
| on other threads  |  abandoned_count > 0
+-------------------+
    |
    v
+-------------------+
| tryReclaim() via  |  Atomic CAS: 0 -> new_thread_id
| CAS on thread_id  |
+-------------------+
```

### Reclamation Process

```zig
fn reclaimAbandoned(max_count: usize) usize {
    // Quick check without lock
    if (global_subproc.abandoned_os_list_count.load(.monotonic) == 0) {
        return 0;
    }

    // Lock and iterate abandoned list
    global_subproc.abandoned_os_lock.lock();
    defer global_subproc.abandoned_os_lock.unlock();

    var reclaimed: usize = 0;
    var segment = abandoned_os_list.head;

    while (segment) |seg| {
        if (seg.canReclaim()) {  // has free pages + is abandoned
            abandoned_os_list.remove(seg);
            if (tryReclaimSegment(seg)) {
                reclaimed += 1;
                // Segment now owned by current thread
            }
        }
        if (reclaimed >= max_count) break;
    }

    return reclaimed;
}
```

**Reclamation triggers:**
- During `mallocGeneric` when `abandoned_count > 0`
- When allocation fails and segments need to be reclaimed

### all_segments List

Each thread maintains a list of ALL its segments for cross-thread free collection:

```zig
// In SegmentsTLD (per-thread segment cache)
SegmentsTLD = struct {
    all_segments: AllSegmentsList,  // Doubly-linked list of all owned segments
    // ...
}

// In Segment
Segment = struct {
    all_next: ?*Segment,  // Link for all_segments list
    all_prev: ?*Segment,
    // ...
}
```

**Purpose:** When collecting `xthread_free`, we need to iterate ALL segments (including those with full pages). The `all_segments` list provides O(1) insertion/removal and O(n) iteration:

```zig
fn collectAllXthreadFree(self: *SegmentsTLD) usize {
    var total_collected: usize = 0;

    // Iterate ALL segments owned by this thread
    var segment = self.all_segments.tail;
    while (segment) |seg| {
        const next = seg.all_prev;

        // Collect xthread_free from ALL pages (including full ones)
        for (seg.pages[0..seg.capacity]) |*page| {
            if (page.xthread_free.load(.acquire) != null) {
                total_collected += page.collectXthreadFree();
            }
        }

        segment = next;
    }

    return total_collected;
}
```

**Why not use bin queues?** Bin queues only contain non-full pages. But cross-thread frees can happen to ANY page, including full ones. Without `all_segments`, we'd have no way to find and collect blocks from full pages.

### Segment Lifecycle with Abandonment

```
+-------------+
|   ACTIVE    |  thread_id = owner_thread
+------+------+
       |
       | Thread exits without cleanup
       v
+-------------+
|  ORPHANED   |  thread_id still set, but thread is gone
+------+------+
       |
       | Detected during xthread_free or allocation
       v
+-------------+
|  ABANDONED  |  thread_id = 0, in abandoned_os_list
+------+------+
       |
       | Another thread calls reclaimAbandoned()
       v
+-------------+
|  RECLAIMED  |  thread_id = new_owner, removed from abandoned list
+-------------+
```

## Configuration Constants

```zig
SEGMENT_SIZE        = 32 MiB
SEGMENT_SHIFT       = 25
SMALL_PAGE_SIZE     = 64 KiB
SMALL_PAGE_SHIFT    = 16
MEDIUM_PAGE_SIZE    = 512 KiB
MEDIUM_PAGE_SHIFT   = 19

SMALL_OBJ_SIZE_MAX  = 1 KiB      // Fast path threshold
MEDIUM_OBJ_SIZE_MAX = 128 KiB
LARGE_OBJ_SIZE_MAX  = 16 MiB

BIN_HUGE            = 73
PAGES_DIRECT        = 129        // Direct pointer array size
SMALL_WSIZE_MAX     = 128        // Max wsize for direct lookup
```

## Performance Characteristics

| Operation | Complexity | Cache Misses | Notes |
|-----------|------------|--------------|-------|
| malloc (fast) | O(1) | 1-2 | direct pointer + free list pop |
| malloc (medium) | O(1) | 2-3 | bin queue + pop |
| malloc (slow) | O(n) | many | segment/page allocation |
| free (same-thread) | O(1) | 1-2 | push to local_free |
| free (cross-thread) | O(1) | 1 | atomic CAS |
| ptr -> segment | O(1) | 0 | bitmask only |
| ptr -> page | O(1) | 1 | bitmask + array lookup |

**Cache optimizations:**
- Page struct = 64 bytes = 1 cache line
- Hot fields in first 32 bytes
- Bump allocation = sequential memory access
- Prefetch hints for next block in free list
- Direct pointer array eliminates bin calculation in hot path

## Common Pitfalls for Contributors

1. **Direct pointer mismatch**: When adding a page to a bin, ALL wsizes that map to that bin must have their direct pointers updated. Otherwise allocations of different sizes in the same bin will miss the fast path.

1. **Page full flag**: When a page is exhausted in fast path, it must be marked `in_full` and removed from bin queue. When it receives a free, `freeColdPath` must re-add it.

1. **Cross-thread free accounting**: The `used` counter must be decremented when collecting `xthread_free`, and `pending_xthread_free` global counter must be updated.

1. **Page size constraint**: Page struct must remain exactly 64 bytes (one cache line). Adding fields requires removing or compressing existing ones.

1. **Bin vs wsize**: For small allocations, `bin == wsize`. For larger allocations, multiple wsizes map to the same bin. `blockSizeForBin(bin)` returns the maximum size for that bin.
