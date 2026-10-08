// win/compat/unistd.h - the POSIX file and process calls the library uses, on the UCRT (binary mode always).
#pragma once
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif

int nslm_open(const char* path, int flags, ...);
intptr_t nslm_pwrite(int fd, const void* buf, size_t n, int64_t off);
intptr_t nslm_pread(int fd, void* buf, size_t n, int64_t off);
int nslm_ftruncate(int fd, int64_t len);
void nslm_usleep(unsigned us);
long nslm_sysconf(int name);
int nslm_mkdir(const char* path, int mode);

#define open nslm_open
#define close _close
#define read _read
#define write _write
#define lseek _lseeki64
#define pwrite nslm_pwrite
#define pread nslm_pread
#define ftruncate nslm_ftruncate
#define usleep nslm_usleep
#define getpid _getpid
#define unlink _unlink
#define access _access
#define isatty _isatty
#define fileno _fileno
#define fsync _commit
#define sysconf nslm_sysconf
#define _SC_NPROCESSORS_ONLN 84
#define _SC_CLK_TCK 2
#ifndef F_OK
#define F_OK 0
#define R_OK 4
#define W_OK 2
#endif
typedef int64_t off_t_compat;
#define off_t off_t_compat

#ifdef __cplusplus
}
#endif
