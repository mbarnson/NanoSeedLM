// harness/platform.c - see platform.h.  Windows (Win32), macOS (Mach, libproc, IOKit) and Linux (/proc).
#define _DEFAULT_SOURCE   // glibc under -std=c11: utimensat, st_mtim
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#include <psapi.h>
#include <shellapi.h>
#elif defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#include <libproc.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <unistd.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

char* plat_slurp(const char* path, size_t* len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = (char*) malloc((size_t) (n < 0 ? 0 : n) + 1);
    const size_t got = fread(b, 1, (size_t) (n < 0 ? 0 : n), f);
    fclose(f);
    b[got] = 0;
    if (len) *len = got;
    return b;
}

// File times to the file system's precision (the cold KV cache orders uses within a second by them).  Windows: 100 ns
// units since 1601, in integers (a double of them would round to microseconds).
#if defined(_WIN32)
#define FT_1970 11644473600ull   // seconds from 1601 to 1970
int plat_set_mtime(const char* path, double t) {
    HANDLE h = CreateFileA(path, FILE_WRITE_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) return -1;
    const uint64_t s = (uint64_t) t, u = (s + FT_1970) * 10000000ull + (uint64_t) ((t - (double) s) * 1e7);
    FILETIME ft = {(DWORD) u, (DWORD) (u >> 32)};
    const BOOL ok = SetFileTime(h, NULL, &ft, &ft);
    CloseHandle(h);
    return ok ? 0 : -1;
}
double plat_mtime(const char* path) {
    WIN32_FILE_ATTRIBUTE_DATA a;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &a)) return -1;
    const uint64_t u = (uint64_t) a.ftLastWriteTime.dwHighDateTime << 32 | a.ftLastWriteTime.dwLowDateTime;
    return (double) (u / 10000000ull - FT_1970) + 1e-7 * (double) (u % 10000000ull);
}
#else
int plat_set_mtime(const char* path, double t) {
    const time_t s = (time_t) t;
    struct timespec ts[2];
    ts[0].tv_sec = s;
    ts[0].tv_nsec = (long) ((t - (double) s) * 1e9);
    ts[1] = ts[0];
    return utimensat(AT_FDCWD, path, ts, 0) ? -1 : 0;
}
double plat_mtime(const char* path) {
    struct stat st;
    if (stat(path, &st)) return -1;
#if defined(__APPLE__)
    return (double) st.st_mtimespec.tv_sec + 1e-9 * (double) st.st_mtimespec.tv_nsec;
#else
    return (double) st.st_mtim.tv_sec + 1e-9 * (double) st.st_mtim.tv_nsec;
#endif
}
#endif

#if defined(_WIN32)

const void* plat_map(const char* path, size_t* len) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return NULL;
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    HANDLE m = sz.QuadPart ? CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL) : NULL;
    CloseHandle(f);
    if (!m) return NULL;
    const void* p = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(m);
    if (p && len) *len = (size_t) sz.QuadPart;
    return p;
}
uint64_t plat_random_u64(void) {
    uint64_t v = 0;
    BCryptGenRandom(NULL, (PUCHAR) &v, sizeof v, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    return v;
}
void plat_init(int* argc, char*** argv) {
    SetConsoleOutputCP(CP_UTF8);
    int n = 0;
    wchar_t** w = CommandLineToArgvW(GetCommandLineW(), &n);
    if (!w) return;
    char** a = (char**) calloc((size_t) n + 1, sizeof(char*));
    for (int i = 0; i < n; ++i) {
        const int k = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, NULL, 0, NULL, NULL);
        a[i] = (char*) malloc((size_t) k);
        WideCharToMultiByte(CP_UTF8, 0, w[i], -1, a[i], k, NULL, NULL);
    }
    LocalFree(w);
    *argc = n;
    *argv = a;   // lives for the process
}
double now_s(void) {
    static LARGE_INTEGER f;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double) c.QuadPart / (double) f.QuadPart;
}
ProcMem proc_mem(void) {
    ProcMem m = {-1, -1, -1};
    PROCESS_MEMORY_COUNTERS_EX pc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*) &pc, sizeof pc)) {
        m.phys_footprint_mb = (double) pc.PrivateUsage / 1048576.0;
        m.phys_footprint_peak_mb = (double) pc.PeakPagefileUsage / 1048576.0;
        m.rss_mb = (double) pc.WorkingSetSize / 1048576.0;
    }
    return m;
}
static double ft_s(FILETIME t) { return (double) (((uint64_t) t.dwHighDateTime << 32) | t.dwLowDateTime) * 1e-7; }
double host_cpu_busy_s(void) {
    FILETIME idle, kern, user;
    if (!GetSystemTimes(&idle, &kern, &user)) return -1;
    return ft_s(kern) + ft_s(user) - ft_s(idle);   // kernel time includes idle time
}
double self_cpu_s(void) {
    FILETIME c, x, k, u;
    if (!GetProcessTimes(GetCurrentProcess(), &c, &x, &k, &u)) return -1;
    return ft_s(k) + ft_s(u);
}
uint64_t plat_mem_available(void) {
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullAvailPhys : 0;
}
MachineState machine_state(void) {
    MachineState s;
    memset(&s, 0, sizeof s);
    s.load1 = s.load5 = s.load15 = -1;
    s.thermal = -1;
    SYSTEM_POWER_STATUS ps;
    snprintf(s.power, sizeof s.power, "unknown");
    if (GetSystemPowerStatus(&ps)) {
        if (ps.ACLineStatus == 1) snprintf(s.power, sizeof s.power, "AC");
        else if (ps.ACLineStatus == 0) snprintf(s.power, sizeof s.power, "Battery");
        s.low_power = ps.SystemStatusFlag == 1;   // battery saver
    }
    return s;
}

#else   // macOS and Linux

const void* plat_map(const char* path, size_t* len) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    fstat(fd, &st);
    void* m = st.st_size ? mmap(NULL, (size_t) st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    close(fd);
    if (m == MAP_FAILED) return NULL;
    if (len) *len = (size_t) st.st_size;
    return m;
}
uint64_t plat_random_u64(void) {
    uint64_t v = 0;
#if defined(__APPLE__)
    arc4random_buf(&v, sizeof v);
#else
    FILE* f = fopen("/dev/urandom", "rb");
    if (f) { if (fread(&v, sizeof v, 1, f) != 1) v = 0; fclose(f); }
    if (!v) v = (uint64_t) time(NULL) * 0x9E3779B97F4A7C15ull;
#endif
    return v;
}
void plat_init(int* argc, char*** argv) { (void) argc; (void) argv; }
double now_s(void) {
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}
double self_cpu_s(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

#if defined(__APPLE__)
ProcMem proc_mem(void) {
    ProcMem m = {-1, -1, -1};
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t*) &ri) == 0) {
        m.phys_footprint_mb = (double) ri.ri_phys_footprint / 1048576.0;
        m.phys_footprint_peak_mb = (double) ri.ri_lifetime_max_phys_footprint / 1048576.0;
        m.rss_mb = (double) ri.ri_resident_size / 1048576.0;
    }
    return m;
}
double host_cpu_busy_s(void) {
    host_cpu_load_info_data_t c;
    mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
    static mach_port_t host = MACH_PORT_NULL;
    if (host == MACH_PORT_NULL) host = mach_host_self();
    if (host_statistics(host, HOST_CPU_LOAD_INFO, (host_info_t) &c, &n) != KERN_SUCCESS) return -1;
    const double busy = (double) c.cpu_ticks[CPU_STATE_USER] + c.cpu_ticks[CPU_STATE_SYSTEM] + c.cpu_ticks[CPU_STATE_NICE];
    return busy / (double) sysconf(_SC_CLK_TCK);
}
uint64_t plat_mem_available(void) {   // free, inactive, purgeable and speculative pages
    vm_statistics64_data_t v;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &v, &n) != KERN_SUCCESS) return 0;
    return ((uint64_t) v.free_count + v.inactive_count + v.purgeable_count + v.speculative_count) * (uint64_t) vm_page_size;
}
MachineState machine_state(void) {
    MachineState s;
    memset(&s, 0, sizeof s);
    double la[3] = {-1, -1, -1};
    getloadavg(la, 3);
    s.load1 = la[0]; s.load5 = la[1]; s.load15 = la[2];
    s.thermal = -1;
    s.low_power = 0;
    snprintf(s.power, sizeof s.power, "unknown");
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (info) {
        CFStringRef src = IOPSGetProvidingPowerSourceType(info);
        if (src) {
            if (CFStringCompare(src, CFSTR(kIOPMACPowerKey), 0) == kCFCompareEqualTo) snprintf(s.power, sizeof s.power, "AC");
            else if (CFStringCompare(src, CFSTR(kIOPMBatteryPowerKey), 0) == kCFCompareEqualTo) snprintf(s.power, sizeof s.power, "Battery");
        }
        CFRelease(info);
    }
    return s;
}
#else
ProcMem proc_mem(void) {
    ProcMem m = {-1, -1, -1};
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return m;
    char line[256];
    while (fgets(line, sizeof line, f)) {
        long kb;
        if (sscanf(line, "VmRSS: %ld", &kb) == 1) m.rss_mb = m.phys_footprint_mb = kb / 1024.0;
        if (sscanf(line, "VmHWM: %ld", &kb) == 1) m.phys_footprint_peak_mb = kb / 1024.0;
    }
    fclose(f);
    return m;
}
double host_cpu_busy_s(void) {
    FILE* f = fopen("/proc/stat", "r");
    if (!f) return -1;
    unsigned long long u, n, s;
    const int ok = fscanf(f, "cpu %llu %llu %llu", &u, &n, &s) == 3;
    fclose(f);
    return ok ? (double) (u + n + s) / (double) sysconf(_SC_CLK_TCK) : -1;
}
uint64_t plat_mem_available(void) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %llu", &kb) == 1) break;
    fclose(f);
    return (uint64_t) kb * 1024;
}
MachineState machine_state(void) {
    MachineState s;
    memset(&s, 0, sizeof s);
    double la[3] = {-1, -1, -1};
    getloadavg(la, 3);
    s.load1 = la[0]; s.load5 = la[1]; s.load15 = la[2];
    s.thermal = -1;
    snprintf(s.power, sizeof s.power, "unknown");
    return s;
}
#endif
#endif

void print_machine_state(const char* prefix, MachineState s) {
    printf("%spower=%s\n%sload1=%.2f\n%sload5=%.2f\n%sthermal_state=%d\n%slow_power_mode=%d\n", prefix, s.power, prefix,
           s.load1, prefix, s.load5, prefix, s.thermal, prefix, s.low_power);
}
