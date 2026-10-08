// win/compat/posix_compat.h - force-included (/FI) into every C and C++ translation unit of the Windows build.
// The small POSIX surface the C99 library and the tools use, on the Win32 API and the UCRT: clocks, strdup, ssize_t,
// the GCC va_list builtins, and the Clang/GCC attributes MSVC lacks.  The shim headers next to this file (unistd.h,
// sys/mman.h, dirent.h, pthread.h) cover the rest.
#pragma once
#ifdef _WIN32

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <stdarg.h>
#include <stdio.h>
#include <io.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <direct.h>
#include <sys/stat.h>
#define mkdir(path, mode) _mkdir(path)
#define rmdir _rmdir

#if !defined(__clang__) && !defined(__GNUC__)
#define __builtin_va_list va_list
#define __builtin_va_start va_start
#define __builtin_va_end va_end
#endif

#ifndef _SSIZE_T_DEFINED
#define _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#endif
typedef int pid_t_compat;

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_REALTIME 0
#define CLOCK_MONOTONIC 1
#define CLOCK_MONOTONIC_RAW 4
#endif
typedef int clockid_t_compat;
int nslm_clock_gettime(int clk, struct timespec* ts);
int nslm_rename(const char* from, const char* to);   // replaces an existing target, as POSIX rename
#define rename nslm_rename
uint64_t nslm_clock_gettime_nsec_np(int clk);
#define clock_gettime nslm_clock_gettime
#define clock_gettime_nsec_np nslm_clock_gettime_nsec_np

#ifdef __cplusplus
}
#endif

#endif
