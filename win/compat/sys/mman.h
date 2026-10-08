// win/compat/sys/mman.h - read-only file mappings (PROT_READ, MAP_PRIVATE / MAP_SHARED) on CreateFileMapping.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define MAP_SHARED 1
#define MAP_PRIVATE 2
#define MAP_ANON 0x20
#define MAP_ANONYMOUS MAP_ANON
#define MAP_FAILED ((void*) -1)
#define MADV_WILLNEED 3
#define MADV_SEQUENTIAL 2
#define MADV_RANDOM 1

void* nslm_mmap(void* addr, size_t len, int prot, int flags, int fd, int64_t off);
int nslm_munmap(void* addr, size_t len);
int nslm_madvise(void* addr, size_t len, int advice);
#define mmap nslm_mmap
#define munmap nslm_munmap
#define madvise nslm_madvise

#ifdef __cplusplus
}
#endif
