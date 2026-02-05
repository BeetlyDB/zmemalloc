const std = @import("std");
const zmemalloc = @import("zmemalloc");

export fn malloc(len: usize) ?*anyopaque {
    return zmemalloc.c_malloc(len);
}

export fn realloc(ptr: ?*anyopaque, size: usize) ?*anyopaque {
    return zmemalloc.c_realloc(ptr, size);
}

export fn free(ptr: ?*anyopaque) void {
    zmemalloc.c_free(ptr);
}

export fn calloc(count: usize, size: usize) ?*anyopaque {
    return zmemalloc.c_calloc(count, size);
}

export fn aligned_alloc(alignment: usize, size: usize) ?*anyopaque {
    return zmemalloc.c_aligned_alloc(alignment, size);
}

export fn posix_memalign(ptr: *?*anyopaque, alignment: usize, size: usize) c_int {
    return zmemalloc.c_posix_memalign(ptr, alignment, size);
}

export fn memalign(alignment: usize, size: usize) ?*anyopaque {
    return zmemalloc.c_memalign(alignment, size);
}

export fn valloc(size: usize) ?*anyopaque {
    return zmemalloc.c_valloc(size);
}

export fn pvalloc(size: usize) ?*anyopaque {
    return zmemalloc.c_pvalloc(size);
}

export fn malloc_usable_size(ptr: ?*anyopaque) usize {
    return zmemalloc.c_malloc_usable_size(ptr);
}

/// Collect unused memory and return to OS
/// force=true to aggressively return all unused memory
export fn collect(force: bool) usize {
    return zmemalloc.collect(force);
}
