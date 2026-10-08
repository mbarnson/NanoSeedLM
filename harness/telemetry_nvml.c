// harness/telemetry_nvml.c - telemetry.h on NVIDIA GPUs (Windows, Linux) through NVML, loaded at run time (the driver
// ships it; nothing to link).  GPU busy % = the mean of NVML's GPU-utilization samples over the window; this
// process's GPU time is not exposed by NVML, so telem_self_gpu_ns reports -1.  No ANE: -1.
#include "telemetry.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define LIBOPEN() ((void*) LoadLibraryA("nvml.dll"))
#define LIBSYM(h, n) ((void*) GetProcAddress((HMODULE) (h), n))
#else
#include <dlfcn.h>
#define LIBOPEN() dlopen("libnvidia-ml.so.1", RTLD_LAZY)
#define LIBSYM(h, n) dlsym(h, n)
#endif

typedef void* nvmlDevice_t;
typedef union { unsigned int ui; unsigned long ul; unsigned long long ull; long long sll; double d; } NvmlValue;
typedef struct { unsigned long long ts; NvmlValue v; } NvmlSample;
enum { NVML_GPU_UTILIZATION_SAMPLES = 1 };

static struct {
    int tried, ok;
    nvmlDevice_t dev;
    int (*samples)(nvmlDevice_t, int, unsigned long long, int*, unsigned int*, NvmlSample*);
} g;

static int telem_init(void) {
    if (g.tried) return g.ok;
    g.tried = 1;
    void* h = LIBOPEN();
    if (!h) return 0;
    int (*init)(void) = (int (*)(void)) LIBSYM(h, "nvmlInit_v2");
    int (*byidx)(unsigned int, nvmlDevice_t*) = (int (*)(unsigned int, nvmlDevice_t*)) LIBSYM(h, "nvmlDeviceGetHandleByIndex_v2");
    g.samples = (int (*)(nvmlDevice_t, int, unsigned long long, int*, unsigned int*, NvmlSample*)) LIBSYM(h, "nvmlDeviceGetSamples");
    if (!init || !byidx || !g.samples || init() != 0 || byidx(0, &g.dev) != 0) return 0;
    g.ok = 1;
    return 1;
}

// The mark is the host time stamp (microseconds) of the newest sample at the window's start.
TelemMark telem_mark(void) {
    TelemMark m = {NULL};
    if (!telem_init()) return m;
    int type;
    unsigned int n = 0;
    if (g.samples(g.dev, NVML_GPU_UTILIZATION_SAMPLES, 0, &type, &n, NULL) != 0 || !n) return m;
    NvmlSample* s = (NvmlSample*) calloc(n, sizeof *s);
    unsigned long long last = 0;
    if (g.samples(g.dev, NVML_GPU_UTILIZATION_SAMPLES, 0, &type, &n, s) == 0)
        for (unsigned int i = 0; i < n; ++i) if (s[i].ts > last) last = s[i].ts;
    free(s);
    unsigned long long* p = (unsigned long long*) malloc(sizeof *p);
    *p = last;
    m.sample = p;
    return m;
}

TelemResult telem_since(TelemMark m) {
    TelemResult r = {-1, -1, 0};
    if (!m.sample) return r;
    const unsigned long long since = *(unsigned long long*) m.sample;
    free(m.sample);
    int type;
    unsigned int n = 0;
    if (g.samples(g.dev, NVML_GPU_UTILIZATION_SAMPLES, since, &type, &n, NULL) != 0 || !n) return r;
    NvmlSample* s = (NvmlSample*) calloc(n, sizeof *s);
    if (g.samples(g.dev, NVML_GPU_UTILIZATION_SAMPLES, since, &type, &n, s) == 0 && n) {
        double sum = 0;
        for (unsigned int i = 0; i < n; ++i) sum += (double) s[i].v.ui;
        r.gpu_busy_pct = sum / n;
        r.ok = 1;
    }
    free(s);
    return r;
}

long long telem_self_gpu_ns(void) { return -1; }
