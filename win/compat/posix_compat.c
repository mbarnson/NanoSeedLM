// win/compat/posix_compat.c - the Win32 implementations behind win/compat/*.h.
#include <windows.h>

#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "dirent.h"
#include "pthread.h"
#include "sys/mman.h"
#include "unistd.h"

// ---- clocks ---------------------------------------------------------------------------------------------------------

int nslm_clock_gettime(int clk, struct timespec* ts) {
    if (clk == CLOCK_REALTIME) return timespec_get(ts, TIME_UTC) == TIME_UTC ? 0 : -1;
    static LARGE_INTEGER f;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    ts->tv_sec = (time_t) (c.QuadPart / f.QuadPart);
    ts->tv_nsec = (long) ((c.QuadPart % f.QuadPart) * 1000000000LL / f.QuadPart);
    return 0;
}
uint64_t nslm_clock_gettime_nsec_np(int clk) {
    struct timespec ts;
    nslm_clock_gettime(clk, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

#undef rename
int nslm_rename(const char* from, const char* to) {
    return MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED) ? 0 : -1;
}

// ---- files ----------------------------------------------------------------------------------------------------------

#undef open
int nslm_open(const char* path, int flags, ...) {
    int mode = 0;
    if (flags & _O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = va_arg(ap, int);
        va_end(ap);
    }
    (void) mode;
    int fd = -1;
    _sopen_s(&fd, path, flags | _O_BINARY, _SH_DENYNO, _S_IREAD | _S_IWRITE);
    return fd;
}

static intptr_t rw_at(int fd, void* buf, size_t n, int64_t off, int wr) {
    HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }
    size_t done = 0;
    while (done < n) {
        OVERLAPPED ov;
        memset(&ov, 0, sizeof ov);
        const uint64_t o = (uint64_t) off + done;
        ov.Offset = (DWORD) o;
        ov.OffsetHigh = (DWORD) (o >> 32);
        const DWORD chunk = n - done > (1u << 30) ? (1u << 30) : (DWORD) (n - done);
        DWORD got = 0;
        const BOOL ok = wr ? WriteFile(h, (const char*) buf + done, chunk, &got, &ov) : ReadFile(h, (char*) buf + done, chunk, &got, &ov);
        if (!ok) {
            if (!wr && GetLastError() == ERROR_HANDLE_EOF) break;
            errno = EIO;
            return done ? (intptr_t) done : -1;
        }
        if (!got) break;
        done += got;
    }
    return (intptr_t) done;
}
intptr_t nslm_pwrite(int fd, const void* buf, size_t n, int64_t off) { return rw_at(fd, (void*) buf, n, off, 1); }
intptr_t nslm_pread(int fd, void* buf, size_t n, int64_t off) { return rw_at(fd, buf, n, off, 0); }
int nslm_ftruncate(int fd, int64_t len) { return _chsize_s(fd, len) ? -1 : 0; }
void nslm_usleep(unsigned us) { Sleep((us + 999) / 1000); }
long nslm_sysconf(int name) {
    if (name == _SC_NPROCESSORS_ONLN) {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (long) si.dwNumberOfProcessors;
    }
    if (name == _SC_CLK_TCK) return 100;
    return -1;
}
int nslm_mkdir(const char* path, int mode) { (void) mode; return _mkdir(path); }

// ---- mmap: read-only views of whole files ---------------------------------------------------------------------------

void* nslm_mmap(void* addr, size_t len, int prot, int flags, int fd, int64_t off) {
    (void) addr;
    if (flags & MAP_ANON) {
        void* p = VirtualAlloc(NULL, len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        return p ? p : MAP_FAILED;
    }
    HANDLE h = (HANDLE) _get_osfhandle(fd);
    if (h == INVALID_HANDLE_VALUE || len == 0) return MAP_FAILED;
    const int w = (prot & PROT_WRITE) != 0;
    // MAP_PRIVATE + PROT_WRITE = copy on write
    const DWORD pp = w ? ((flags & MAP_PRIVATE) ? PAGE_WRITECOPY : PAGE_READWRITE) : PAGE_READONLY;
    const DWORD acc = w ? ((flags & MAP_PRIVATE) ? FILE_MAP_COPY : FILE_MAP_WRITE) : FILE_MAP_READ;
    const uint64_t end = (uint64_t) off + len;
    HANDLE m = CreateFileMappingA(h, NULL, pp, (DWORD) (end >> 32), (DWORD) end, NULL);
    if (!m) return MAP_FAILED;
    void* p = MapViewOfFile(m, acc, (DWORD) ((uint64_t) off >> 32), (DWORD) off, len);
    CloseHandle(m);   // the view keeps the mapping alive
    return p ? p : MAP_FAILED;
}
int nslm_munmap(void* addr, size_t len) {
    (void) len;
    if (UnmapViewOfFile(addr)) return 0;
    return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
}
int nslm_madvise(void* addr, size_t len, int advice) {
    if (advice == MADV_WILLNEED) {
        WIN32_MEMORY_RANGE_ENTRY r = {addr, len};
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &r, 0);
    }
    return 0;
}

// ---- dirent ---------------------------------------------------------------------------------------------------------

struct NslmDir {
    HANDLE h;
    WIN32_FIND_DATAA fd;
    int first;
    struct dirent de;
};
DIR* nslm_opendir(const char* path) {
    char pat[1100];
    snprintf(pat, sizeof pat, "%s\\*", path);
    DIR* d = (DIR*) calloc(1, sizeof *d);
    d->h = FindFirstFileA(pat, &d->fd);
    if (d->h == INVALID_HANDLE_VALUE) { free(d); errno = ENOENT; return NULL; }
    d->first = 1;
    return d;
}
struct dirent* nslm_readdir(DIR* d) {
    if (!d->first && !FindNextFileA(d->h, &d->fd)) return NULL;
    d->first = 0;
    snprintf(d->de.d_name, sizeof d->de.d_name, "%s", d->fd.cFileName);
    return &d->de;
}
int nslm_closedir(DIR* d) {
    FindClose(d->h);
    free(d);
    return 0;
}

// ---- pthreads -------------------------------------------------------------------------------------------------------

typedef struct {
    void* (*fn)(void*);
    void* arg;
    void* ret;
} Start;
static DWORD WINAPI thread_main(LPVOID p) {
    Start* s = (Start*) p;
    s->ret = s->fn(s->arg);
    return 0;
}
// The handle carries its Start block: pthread_join frees it.
typedef struct {
    HANDLE h;
    Start* s;
} Thr;
int pthread_create(pthread_t* t, const pthread_attr_t* attr, void* (*fn)(void*), void* arg) {
    (void) attr;
    Thr* th = (Thr*) calloc(1, sizeof *th);
    th->s = (Start*) calloc(1, sizeof *th->s);
    th->s->fn = fn;
    th->s->arg = arg;
    th->h = CreateThread(NULL, 8u << 20, thread_main, th->s, 0, NULL);
    if (!th->h) { free(th->s); free(th); return EAGAIN; }
    t->h = th;
    return 0;
}
int pthread_join(pthread_t t, void** ret) {
    Thr* th = (Thr*) t.h;
    WaitForSingleObject(th->h, INFINITE);
    CloseHandle(th->h);
    if (ret) *ret = th->s->ret;
    free(th->s);
    free(th);
    return 0;
}
int pthread_detach(pthread_t t) {
    Thr* th = (Thr*) t.h;
    CloseHandle(th->h);   // the Start block leaks with the thread (detached threads here live for the process)
    free(th);
    return 0;
}
int pthread_mutex_init(pthread_mutex_t* m, const pthread_mutexattr_t* a) { (void) a; InitializeSRWLock((PSRWLOCK) &m->p); return 0; }
int pthread_mutex_destroy(pthread_mutex_t* m) { (void) m; return 0; }
int pthread_mutex_lock(pthread_mutex_t* m) { AcquireSRWLockExclusive((PSRWLOCK) &m->p); return 0; }
int pthread_mutex_unlock(pthread_mutex_t* m) { ReleaseSRWLockExclusive((PSRWLOCK) &m->p); return 0; }
int pthread_cond_init(pthread_cond_t* c, const pthread_condattr_t* a) { (void) a; InitializeConditionVariable((PCONDITION_VARIABLE) &c->p); return 0; }
int pthread_cond_destroy(pthread_cond_t* c) { (void) c; return 0; }
int pthread_cond_wait(pthread_cond_t* c, pthread_mutex_t* m) {
    return SleepConditionVariableSRW((PCONDITION_VARIABLE) &c->p, (PSRWLOCK) &m->p, INFINITE, 0) ? 0 : EINVAL;
}
int pthread_cond_signal(pthread_cond_t* c) { WakeConditionVariable((PCONDITION_VARIABLE) &c->p); return 0; }
int pthread_cond_broadcast(pthread_cond_t* c) { WakeAllConditionVariable((PCONDITION_VARIABLE) &c->p); return 0; }
