// harness/nslm-handoff-bench.c - the cost of a GPU-host hand-off inside a CUDA graph (plan D4): 48 a forward, as one
// for each layer of Qwen3.8-Flash-Next. A host thread answers each signal after --work microseconds.
//   nslm-handoff-bench [--layers 48] [--forwards 200] [--work 0]
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "kernels_qwen4.h"
#include "platform.h"

typedef struct { volatile uint32_t *flag, *ack; volatile int stop; double work_us; } Host;

static void* answer(void* arg) {   // the CPU side: wait for each signal, work, then acknowledge it
    Host* h = (Host*) arg;
    uint32_t done = 0;
    while (!h->stop) {
        const uint32_t v = *h->flag;
        if (v == done) continue;
        if (h->work_us > 0) { const double t = now_s() + h->work_us * 1e-6; while (now_s() < t) {} }
        done = v;
        *h->ack = v;
    }
    return NULL;
}

int main(int argc, char** argv) {
    int layers = 48, forwards = 200;
    double work = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--forwards") && i + 1 < argc) forwards = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--work") && i + 1 < argc) work = atof(argv[++i]);
        else { fprintf(stderr, "usage: nslm-handoff-bench [--layers N] [--forwards N] [--work US]\n"); return 2; }
    }
    uint32_t *hf, *df, *seq;
    if (cudaHostAlloc((void**) &hf, 128, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer((void**) &df, hf, 0) != cudaSuccess ||
        cudaMalloc((void**) &seq, 4) != cudaSuccess) { fprintf(stderr, "CUDA setup failed\n"); return 1; }
    memset(hf, 0, 128);
    cudaMemset(seq, 0, 4);
    Host h = {hf, hf + 16, 0, work};   // flag and ack on separate cache lines
    pthread_t th;
    pthread_create(&th, NULL, answer, &h);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaGraphExec_t ge[2];
    for (int w = 0; w < 2; ++w) {   // 0: the hand-offs, 1: the same graph without them (its launch cost)
        cudaGraph_t g;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        for (int l = 0; l < layers; ++l) {
            if (w == 0) { kq4_signal(s, seq, df); kq4_wait(s, seq, df + 16); }
            else { kq4_signal(s, seq + 0, df + 32); kq4_signal(s, seq, df + 32); }
        }
        cudaStreamEndCapture(s, &g);
        if (cudaGraphInstantiate(&ge[w], g, 0) != cudaSuccess) { fprintf(stderr, "graph failed\n"); return 1; }
        cudaGraphDestroy(g);
    }
    double ms[2];
    for (int w = 0; w < 2; ++w) {
        cudaGraphLaunch(ge[w], s);   // warm-up
        cudaStreamSynchronize(s);
        const double t0 = now_s();
        for (int f = 0; f < forwards; ++f) cudaGraphLaunch(ge[w], s);
        cudaStreamSynchronize(s);
        ms[w] = (now_s() - t0) * 1e3 / forwards;
    }
    h.stop = 1;
    pthread_join(th, NULL);
    const cudaError_t err = cudaGetLastError();
    printf("%d hand-offs a forward, host work %.0f us: %.3f ms a forward, %.3f ms without the waits; %.1f us a hand-off%s\n", layers,
           work, ms[0], ms[1], 1e3 * (ms[0] - ms[1]) / layers, err == cudaSuccess ? "" : " (CUDA error)");
    return err != cudaSuccess;
}
