// harness/platform.h - what the command-line tools need from the OS, in C, for macOS, Windows and Linux: a monotonic
// clock, UTF-8 command lines and console output, file slurping, process memory and CPU, and machine state.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Call first in main: on Windows, replaces argv with the UTF-8 command line and makes the console print UTF-8.
void plat_init(int* argc, char*** argv);
double now_s(void);   // monotonic seconds
// The whole file, NUL-terminated (*len without the NUL); NULL if unreadable.  The caller frees.
char* plat_slurp(const char* path, size_t* len);
uint64_t plat_random_u64(void);   // from the OS's random source
// The whole file mapped read-only (*len bytes); NULL if unreadable.  Lives for the process.
const void* plat_map(const char* path, size_t* len);

typedef struct {
    double phys_footprint_mb;       // current physical footprint (macOS: phys_footprint; Windows: private working set)
    double phys_footprint_peak_mb;  // lifetime maximum
    double rss_mb;                  // resident size (working set)
} ProcMem;
ProcMem proc_mem(void);
double host_cpu_busy_s(void);   // CPU seconds used by every core (-1 if unavailable)
double self_cpu_s(void);        // CPU seconds used by this process

typedef struct {
    double load1, load5, load15;   // -1 where the OS has no load average
    int thermal;                   // -1: not tracked
    int low_power;
    char power[32];                // "AC", "Battery" or "unknown"
} MachineState;
MachineState machine_state(void);
void print_machine_state(const char* prefix, MachineState s);

static inline const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static inline int opt_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}
static inline int cmp_double(const void* a, const void* b) {
    const double x = *(const double*) a, y = *(const double*) b;
    return x < y ? -1 : x > y;
}

#ifdef __cplusplus
}
#endif
