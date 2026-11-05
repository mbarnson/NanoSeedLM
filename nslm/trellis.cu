// nslm/trellis.cu - the trellis encoder of nslm/trellis.h on CUDA (nslm/trellis_gpu.h): every expression is
// nslm/lib_trellis.c's, in the same order, so the path, scales and error match nslm_tq_encode_row bit for bit.
// Compiled with --fmad=false; the double square root and divisions are IEEE.
//
// One thread block encodes one block of T weights of one row (TQ_TPB threads).  The Viterbi search never stores the
// 2^L state costs: the cost of state s at step t is m_{t-1}[s mod Q] + h_t (w_t - a q(s))^2, Q = 2^(L-K), where
// m_{t-1}[p] is the least cost of the 2^K states (p << K) | j at step t - 1 (lib_trellis.c's z->m).  So shared memory
// holds m (Q floats), and step t computes m_t[q] = min_j of the costs of the states (q << K) | j and its arg min j
// (the back pointer, one byte, in global memory).  Ties go to the lowest j and, at the end, the lowest state, as in C.
//
// Thread items: item u in [0, Q) is the group q = (u >> K) | ((u & (2^K - 1)) << (L - 2K)).  The 2^K items that share
// u >> K read the same 2^K entries of m (a broadcast), so a warp's reads hit distinct banks; back pointers are stored
// in item order, so a warp's stores are coalesced.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "trellis.h"
#include "trellis_gpu.h"

#define TQ_TPB 512
#define TQ_BACK_BYTES ((size_t) 512 << 20)   // back pointers of one launch, at most
#define TQ_MAX_CTAS 2048                      // thread blocks a launch, at most (Windows' display watchdog)

typedef struct { int cols, T, L, K, nb, idx0, has_h; } TqArgs;

static __device__ __forceinline__ int tq_code_d(unsigned s) {   // nslm_tq_code
    const unsigned x = s * 34038481u + 76625530u;
    return (int) ((x & 255u) + (x >> 8 & 255u) + (x >> 16 & 255u) + (x >> 24)) - 510;
}

// item u's group q, and back
static __device__ __forceinline__ int tq_item_q(int u, int L, int K) { return (u >> K) | ((u & ((1 << K) - 1)) << (L - 2 * K)); }
static __device__ __forceinline__ int tq_q_item(int q, int L, int K) { return ((q & ((1 << (L - 2 * K)) - 1)) << K) | (q >> (L - 2 * K)); }

template <int QPT>
__global__ void __launch_bounds__(TQ_TPB) k_tq_encode(TqArgs A, const float* __restrict__ W, const float* __restrict__ Hm,
                                                      unsigned char* __restrict__ back_all, unsigned short* __restrict__ path,
                                                      float* __restrict__ aout) {
    extern __shared__ float sm[];
    __shared__ float red_v[TQ_TPB / 32];
    __shared__ int red_q[TQ_TPB / 32];
    __shared__ float a_sh;
    const int L = A.L, K = A.K, Q = 1 << (L - K), H = 1 << K, qm = Q - 1;
    float* m = sm;                          // [Q]
    float* wsh = m + Q;                     // [T]
    float* hsh = wsh + A.T;                 // [T]
    int* st = (int*) (hsh + A.T);           // [T]
    const int idx = A.idx0 + (int) blockIdx.x, r = idx / A.nb, b = idx % A.nb, j0 = b * A.T;
    const int n = A.cols - j0 < A.T ? A.cols - j0 : A.T;
    const int tid = (int) threadIdx.x;
    unsigned char* back = back_all + (size_t) blockIdx.x * (size_t) A.T * (size_t) Q;
    for (int t = tid; t < n; t += TQ_TPB) {
        wsh[t] = W[(size_t) r * A.cols + j0 + t];
        hsh[t] = A.has_h ? Hm[j0 + t] : 1.0f;
    }
    __syncthreads();
    if (tid == 0) {
        double sw = 0, sh = 0;
        for (int t = 0; t < n; ++t) { const double ht = (double) hsh[t]; sw += ht * (double) wsh[t] * (double) wsh[t]; sh += ht; }
        a_sh = sh > 0 ? (float) (sqrt(sw / sh) / 147.8) : 0.0f;   // q has standard deviation about 147.8
    }
    __syncthreads();
    float a = a_sh;
    for (int pass = 0; pass < 3 && a != 0.0f; ++pass) {
        for (int q = tid; q < Q; q += TQ_TPB) m[q] = 0.0f;   // step 0's costs are h_0 d^2 alone: 0 + x is x
        __syncthreads();
        float reg[QPT];
        for (int t = 0; t < n; ++t) {
            const float wt = wsh[t], ht = hsh[t];
            unsigned char* bk = back + (size_t) t * (size_t) Q;
#pragma unroll
            for (int i = 0; i < QPT; ++i) {
                const int u = tid + i * TQ_TPB;
                if (u < Q) {
                    const int q = tq_item_q(u, L, K);
                    float best = 0.0f;
                    int bj = 0;
                    for (int j = 0; j < H; ++j) {
                        const unsigned s = ((unsigned) q << K) | (unsigned) j;
                        const float ac = a * (float) tq_code_d(s);
                        const float d = wt - ac;
                        const float c = m[s & (unsigned) qm] + ht * d * d;
                        if (j == 0 || c < best) { best = c; bj = j; }
                    }
                    reg[i] = best;
                    bk[u] = (unsigned char) bj;
                }
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < QPT; ++i) {
                const int u = tid + i * TQ_TPB;
                if (u < Q) m[tq_item_q(u, L, K)] = reg[i];
            }
            __syncthreads();
        }
        // the lowest state of least cost: the lowest group q of least m, then its back pointer
        float bv = 0.0f;
        int bq = -1;
        for (int q = tid; q < Q; q += TQ_TPB)
            if (bq < 0 || m[q] < bv) { bv = m[q]; bq = q; }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int oq = __shfl_down_sync(0xffffffffu, bq, off);
            if (oq >= 0 && (bq < 0 || ov < bv || (ov == bv && oq < bq))) { bv = ov; bq = oq; }
        }
        if ((tid & 31) == 0) { red_v[tid >> 5] = bv; red_q[tid >> 5] = bq; }
        __syncthreads();
        if (tid == 0) {
            bv = red_v[0]; bq = red_q[0];
            for (int k = 1; k < TQ_TPB / 32; ++k)
                if (red_q[k] >= 0 && (bq < 0 || red_v[k] < bv || (red_v[k] == bv && red_q[k] < bq))) { bv = red_v[k]; bq = red_q[k]; }
            // back-track: the back pointer of step t picks the predecessor at step t - 1
            int s = (bq << K) | back[(size_t) (n - 1) * Q + tq_q_item(bq, L, K)];
            for (int t = n - 1; t > 0; --t) {
                st[t] = s;
                const int q = s & qm;
                s = (q << K) | back[(size_t) (t - 1) * Q + tq_q_item(q, L, K)];
            }
            st[0] = s;
            // the weighted least-squares scale of this path
            double num = 0, den = 0;
            for (int t = 0; t < n; ++t) {
                const double h = (double) hsh[t], q = (double) tq_code_d((unsigned) st[t]);
                num += h * (double) wsh[t] * q;
                den += h * q * q;
            }
            if (den > 0) a_sh = (float) (num / den);
        }
        __syncthreads();
        a = a_sh;
    }
    if (a == 0.0f)
        for (int t = tid; t < n; t += TQ_TPB) st[t] = 0;
    __syncthreads();
    for (int t = tid; t < n; t += TQ_TPB) path[(size_t) idx * A.T + t] = (unsigned short) st[t];
    if (tid == 0) aout[idx] = a;
}

static uint16_t f2bf_h(float f) {   // nslm_f2bf (nslm/lfsr.h)
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static float bf2f_h(uint16_t v) {   // nslm_bf2f
    const uint32_t u = (uint32_t) v << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
static void put_bits_h(uint32_t* row, uint64_t bit, uint32_t v, int nb) {   // lib_trellis.c's put_bits
    for (int i = 0; i < nb; ++i, ++bit)
        if (v >> i & 1u) row[bit >> 5] |= 1u << (bit & 31); else row[bit >> 5] &= ~(1u << (bit & 31));
}

struct NslmTqGpu {
    cudaStream_t st;
    int smem_optin;
};

// The shared-memory limit is the kernel's, for every thread of the process: set the most the device allows (its opt-in
// maximum less the kernel's static arrays), the same from every caller, so a T 256 launch cannot cut a T 320 one short
template <int QPT>
static cudaError_t launch(const TqArgs& a, int ctas, size_t smem, int smem_max, cudaStream_t st, const float* W, const float* Hm,
                          unsigned char* back, unsigned short* path, float* aout) {
    cudaFuncAttributes fa;
    cudaError_t e = cudaFuncGetAttributes(&fa, k_tq_encode<QPT>);
    if (e == cudaSuccess)
        e = cudaFuncSetAttribute(k_tq_encode<QPT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_max - (int) fa.sharedSizeBytes);
    if (e != cudaSuccess) return e;
    k_tq_encode<QPT><<<(unsigned) ctas, TQ_TPB, smem, st>>>(a, W, Hm, back, path, aout);
    return cudaGetLastError();
}

extern "C" {

NslmTqGpu* nslm_tq_gpu_open(char* err, int errlen) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return NULL; }
    NslmTqGpu* g = (NslmTqGpu*) calloc(1, sizeof(NslmTqGpu));
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g->smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) { snprintf(err, (size_t) errlen, "CUDA stream"); free(g); return NULL; }
    return g;
}

int nslm_tq_gpu_encode(NslmTqGpu* g, const float* w, const float* h, int rows, int cols, int T, int L, int K, uint32_t* words,
                       uint16_t* scales, double* err, char* msg, int msglen) {
    if (rows < 1 || cols < 1 || T < 1 || L > NSLM_TQ_MAXL || K < 1 || K > NSLM_TQ_MAXK || 2 * K > L) {
        snprintf(msg, (size_t) msglen, "bad shape or trellis (L %d, K %d: needs L <= 16, 2 K <= L, K <= 8)", L, K);
        return -1;
    }
    const int Q = 1 << (L - K), nb = (cols + T - 1) / T, qpt = Q > TQ_TPB ? Q / TQ_TPB : 1;
    const size_t smem = sizeof(float) * (size_t) Q + 12 * (size_t) T, total = (size_t) rows * nb;
    if (smem > (size_t) g->smem_optin) {
        snprintf(msg, (size_t) msglen, "L %d, K %d, T %d needs %zu bytes of shared memory (the GPU has %d)", L, K, T, smem, g->smem_optin);
        return -1;
    }
    size_t per = (size_t) T * Q, chunk = TQ_BACK_BYTES / per;
    if (chunk < 1) chunk = 1;
    if (chunk > TQ_MAX_CTAS) chunk = TQ_MAX_CTAS;
    if (chunk > total) chunk = total;
    float *W = NULL, *Hd = NULL, *Ad = NULL;
    unsigned short* P = NULL;
    unsigned char* B = NULL;
    unsigned short* path = (unsigned short*) malloc(sizeof(unsigned short) * total * (size_t) T);
    float* av = (float*) malloc(sizeof(float) * total);
    int rc = -1;
    cudaError_t ce = cudaSuccess;
    if (!path || !av) { snprintf(msg, (size_t) msglen, "out of memory"); goto done; }
    if (cudaMalloc((void**) &W, sizeof(float) * (size_t) rows * cols) || (h && cudaMalloc((void**) &Hd, sizeof(float) * (size_t) cols)) ||
        cudaMalloc((void**) &Ad, sizeof(float) * total) || cudaMalloc((void**) &P, sizeof(unsigned short) * total * (size_t) T) ||
        cudaMalloc((void**) &B, per * chunk)) {
        snprintf(msg, (size_t) msglen, "out of GPU memory");
        goto done;
    }
    cudaMemcpyAsync(W, w, sizeof(float) * (size_t) rows * cols, cudaMemcpyHostToDevice, g->st);
    if (h) cudaMemcpyAsync(Hd, h, sizeof(float) * (size_t) cols, cudaMemcpyHostToDevice, g->st);
    for (size_t i0 = 0; i0 < total && ce == cudaSuccess; i0 += chunk) {
        const int ctas = (int) (total - i0 < chunk ? total - i0 : chunk);
        const TqArgs a = {cols, T, L, K, nb, (int) i0, h != NULL};
        switch (qpt) {
            case 1: ce = launch<1>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
            case 2: ce = launch<2>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
            case 4: ce = launch<4>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
            case 8: ce = launch<8>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
            case 16: ce = launch<16>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
            case 32: ce = launch<32>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
            default: ce = launch<64>(a, ctas, smem, g->smem_optin, g->st, W, Hd, B, P, Ad); break;
        }
    }
    if (ce == cudaSuccess) {
        cudaMemcpyAsync(path, P, sizeof(unsigned short) * total * (size_t) T, cudaMemcpyDeviceToHost, g->st);
        cudaMemcpyAsync(av, Ad, sizeof(float) * total, cudaMemcpyDeviceToHost, g->st);
        ce = cudaStreamSynchronize(g->st);
    }
    if (ce == cudaSuccess) ce = cudaGetLastError();
    if (ce != cudaSuccess) { snprintf(msg, (size_t) msglen, "GPU error: %s", cudaGetErrorString(ce)); goto done; }
    {   // lib_trellis.c's packing and error, block by block (a block: the gotos above must not cross an initialisation)
        const uint64_t nbits = nslm_tq_block_bits(T, L, K);
        const size_t nw = nslm_tq_row_words(cols, T, L, K);
        for (int r = 0; r < rows; ++r) {
            uint32_t* row = words + nw * (size_t) r;
            const float* wr = w + (size_t) r * cols;
            memset(row, 0, sizeof(uint32_t) * nw);
            double e = 0;
            for (int b = 0, j0 = 0; j0 < cols; ++b, j0 += T) {
                const int n = cols - j0 < T ? cols - j0 : T;
                const size_t ib = (size_t) r * nb + b;
                const unsigned short* stp = path + ib * (size_t) T;
                scales[ib] = f2bf_h(av[ib]);
                const float ar = bf2f_h(scales[ib]);
                const uint64_t base = (uint64_t) b * nbits;
                put_bits_h(row, base, (uint32_t) stp[0], L);
                for (int t = 1; t < n; ++t) put_bits_h(row, base + (uint64_t) K * t + (uint64_t) (L - K), (uint32_t) stp[t] >> (L - K), K);
                for (int t = 0; t < n; ++t) {
                    const float v = ar * (float) nslm_tq_code((uint32_t) stp[t]), d = wr[j0 + t] - v;
                    e += (h ? h[j0 + t] : 1.0) * d * d;
                }
            }
            if (err) err[r] = e;
        }
    }
    rc = 0;
done:
    cudaFree(W); cudaFree(Hd); cudaFree(Ad); cudaFree(P); cudaFree(B);
    free(path);
    free(av);
    return rc;
}

void nslm_tq_gpu_close(NslmTqGpu* g) {
    if (!g) return;
    cudaStreamDestroy(g->st);
    free(g);
}

}   // extern "C"
