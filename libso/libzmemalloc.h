#include <stdint.h>

extern void *malloc(uintptr_t const a0);
extern void *realloc(void *const a0, uintptr_t const a1);
extern void free(void *const a0);
extern void *calloc(uintptr_t const a0, uintptr_t const a1);
extern void *aligned_alloc(uintptr_t const a0, uintptr_t const a1);
extern int posix_memalign(void **const a0, uintptr_t const a1, uintptr_t const a2);
extern void *memalign(uintptr_t const a0, uintptr_t const a1);
extern uintptr_t malloc_usable_size(void *const a0);
extern uintptr_t collect(bool force);
