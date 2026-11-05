// engine/vram.h - the VRAM that a CUDA engine keeps free for other users of the GPU (the desktop, other programs).
#pragma once
#include <stdlib.h>

#include <cuda_runtime_api.h>

// NSLM_VRAM_RESERVE_MB; else 2560 MB on a GPU that drives a display (it has a kernel time limit: WDDM, X11); else 512
static inline double nslm_vram_reserve_mb(void) {
    const char* s = getenv("NSLM_VRAM_RESERVE_MB");
    int dev = 0, limit = 0;
    if (s) return atof(s);
    cudaGetDevice(&dev);
    return cudaDeviceGetAttribute(&limit, cudaDevAttrKernelExecTimeout, dev) == cudaSuccess && limit ? 2560 : 512;
}
