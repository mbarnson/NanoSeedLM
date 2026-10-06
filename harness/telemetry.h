// harness/telemetry.h - GPU / ANE activity over a window (IOReport, no sudo), and this process's GPU time.
#pragma once

typedef struct { void* sample; } TelemMark;
typedef struct {
    double gpu_busy_pct;   // GPUPH residency not OFF; -1 if unavailable
    double ane_busy_pct;   // share of PMP AF BW ANE0 L1 samples above the idle bin; -1 if unavailable.
                           // A proxy only: it has crosstalk from GPU activity.
    int ok;
} TelemResult;

TelemMark telem_mark(void);            // start a window
TelemResult telem_since(TelemMark m);  // close it (consumes the mark)

// GPU time used by THIS process, in ns: the sum of accumulatedGPUTime over this pid's IOAccelerator user clients
// (the per-process counter Activity Monitor shows; sudoless).  0 if the process has no GPU client, -1 on an IOKit
// failure.  Unlike the system-wide gpu_busy_pct, this excludes other processes (e.g. WindowServer).
long long telem_self_gpu_ns(void);
