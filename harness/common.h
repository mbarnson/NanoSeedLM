// harness/common.h - header-only helpers shared by the harness tools: option parsing, clocks, and the
// process / machine-state readers recorded with every measurement.
#pragma once
#import <Foundation/Foundation.h>
#include <IOKit/ps/IOPowerSources.h>
#include <IOKit/ps/IOPSKeys.h>
#include <libproc.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#include "engine_api.h"

static inline const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static inline int opt_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}

static inline double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

// ---- process and machine state --------------------------------------------------------------------------------------

typedef struct {
    double phys_footprint_mb;       // current physical footprint (what Activity Monitor calls Memory; includes Metal)
    double phys_footprint_peak_mb;  // lifetime maximum
    double rss_mb;                  // resident size
} ProcMem;

static inline ProcMem proc_mem(pid_t pid) {
    ProcMem m = {-1, -1, -1};
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t*) &ri) == 0) {
        m.phys_footprint_mb = (double) ri.ri_phys_footprint / 1048576.0;
        m.phys_footprint_peak_mb = (double) ri.ri_lifetime_max_phys_footprint / 1048576.0;
        m.rss_mb = (double) ri.ri_resident_size / 1048576.0;
    }
    return m;
}

// CPU seconds used by every core (host_statistics) and by this process (getrusage): their difference over a window is
// the CPU that other processes used while we measured.
static inline double host_cpu_busy_s(void) {
    host_cpu_load_info_data_t c;
    mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
    static mach_port_t host = MACH_PORT_NULL;
    if (host == MACH_PORT_NULL) host = mach_host_self();
    if (host_statistics(host, HOST_CPU_LOAD_INFO, (host_info_t) &c, &n) != KERN_SUCCESS) return -1;
    const double busy = (double) c.cpu_ticks[CPU_STATE_USER] + c.cpu_ticks[CPU_STATE_SYSTEM] + c.cpu_ticks[CPU_STATE_NICE];
    return busy / (double) sysconf(_SC_CLK_TCK);
}
static inline double self_cpu_s(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec + 1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

// "AC" or "Battery" (or "unknown").
static inline const char* power_source(void) {
    static char buf[32];
    snprintf(buf, sizeof buf, "unknown");
    CFTypeRef info = IOPSCopyPowerSourcesInfo();
    if (!info) return buf;
    CFStringRef src = IOPSGetProvidingPowerSourceType(info);
    if (src) {
        if (CFStringCompare(src, CFSTR(kIOPMACPowerKey), 0) == kCFCompareEqualTo) snprintf(buf, sizeof buf, "AC");
        else if (CFStringCompare(src, CFSTR(kIOPMBatteryPowerKey), 0) == kCFCompareEqualTo) snprintf(buf, sizeof buf, "Battery");
    }
    CFRelease(info);
    return buf;
}

typedef struct {
    double load1, load5, load15;
    int thermal;          // NSProcessInfoThermalState (0 nominal .. 3 critical)
    int low_power;
    char power[32];
} MachineState;

static inline MachineState machine_state(void) {
    MachineState s;
    memset(&s, 0, sizeof s);
    double la[3] = {-1, -1, -1};
    getloadavg(la, 3);
    s.load1 = la[0]; s.load5 = la[1]; s.load15 = la[2];
    s.thermal = (int) NSProcessInfo.processInfo.thermalState;
    s.low_power = (int) NSProcessInfo.processInfo.lowPowerModeEnabled;
    snprintf(s.power, sizeof s.power, "%s", power_source());
    return s;
}

static inline void print_machine_state(const char* prefix, MachineState s) {
    printf("%spower=%s\n%sload1=%.2f\n%sload5=%.2f\n%sthermal_state=%d\n%slow_power_mode=%d\n", prefix, s.power, prefix,
           s.load1, prefix, s.load5, prefix, s.thermal, prefix, s.low_power);
}

static inline int cmp_double(const void* a, const void* b) {
    const double x = *(const double*) a, y = *(const double*) b;
    return x < y ? -1 : x > y;
}
