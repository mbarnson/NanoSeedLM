// nslm/searchp.cu - the P = 3 / 8 seed search on CUDA (nslm/searchp_gpu.h), the counterpart of nslm/searchp.metal: every
// expression is nslm/lib_searchp.c's, in the same order.  Compiled with --fmad=false (no contraction); square roots
// and divisions are IEEE (__fsqrt_rn, __fdiv_rn): bit-identical results.
// Grid (column groups from a.g0, row tiles of SP_ROWS), SP_TPB threads, SP_BPT rows per thread; the seed table is
// built SP_NCH seeds at a time in shared memory.  FULL: SH holds a lower-triangular 8 x 8 transform A per column group.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "searchp_gpu.h"

#define MAGIC 12582912.0f
#define kR32 (1.0f / 32767.0f)
#define INF_F __int_as_float(0x7f800000)
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))

static __device__ __forceinline__ float pow2f(int e) { return __uint_as_float((unsigned) (e + 127) << 23); }
static __device__ __forceinline__ int flog2f(float x) { return (int) ((__float_as_uint(x) >> 23) & 255u) - 127; }
static __device__ __forceinline__ float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }
static __device__ __forceinline__ unsigned short lfsr_step_d(unsigned short s) {
    const unsigned short b = (unsigned short) ((s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1u);
    return (unsigned short) ((s >> 1) | (b << 15));
}
static __device__ __forceinline__ float f2bf_f(float f) {   // round to BF16 (nearest even), as f32
    unsigned u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}

template <bool FULL, int P>
static __device__ __forceinline__ void scaled_u(int s, const float* sh, const float* A, float* U) {
    unsigned short st = (unsigned short) s;
    float u0[8 * P];
    for (int k = 0; k < 8 * P; ++k) { st = lfsr_step_d(st); u0[k] = (float) ((int) st - 32768) * kR32; }
    for (int c = 0; c < 8; ++c)
        for (int p = 0; p < P; ++p) {
            if (FULL) {
                float acc = 0.0f;
                for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[P * k + p];
                U[P * c + p] = acc;
            } else U[P * c + p] = u0[P * c + p] * sh[c];
        }
}

// nslmp_seed_entry: U[8 P], L, 1/diag, ok at ent[SP_OK]
template <bool FULL, int P>
static __device__ void seed_entry(int s, const float* sh, const float* A, float* ent) {
    float U[8 * P];
    scaled_u<FULL, P>(s, sh, A, U);
    for (int k = 0; k < 8 * P; ++k) ent[k] = U[k];
    float* L = ent + 8 * P, *D = L + P * (P + 1) / 2;
    ent[SP_OK] = 0.0f;
    for (int i = 0; i < P; ++i)
        for (int j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (int c = 0; c < 8; ++c) g = g + U[P * c + i] * U[P * c + j];
            for (int k = 0; k < j; ++k) g = g - L[LI(i, k)] * L[LI(j, k)];
            if (i == j) {
                if (!(g > 0.0f)) return;
                L[LI(i, i)] = __fsqrt_rn(g);
                D[i] = __fdiv_rn(1.0f, L[LI(i, i)]);
            } else L[LI(i, j)] = g * D[j];
        }
    ent[SP_OK] = 1.0f;
}

template <int P>
static __device__ __forceinline__ void solvep(const float* L, const float* D, const float* b, float* t) {
    float y[P];
    for (int i = 0; i < P; ++i) {
        float v = b[i];
        for (int k = 0; k < i; ++k) v = v - L[LI(i, k)] * y[k];
        y[i] = v * D[i];
    }
    for (int i = P - 1; i >= 0; --i) {
        float v = y[i];
        for (int k = i + 1; k < P; ++k) v = v - L[LI(k, i)] * t[k];
        t[i] = v * D[i];
    }
}

template <int P>
static __device__ __forceinline__ float cand_err(const float* L, const float* q, const float* b, float wn, float sc) {
    float qb = 0.0f, qgq = 0.0f;
    for (int p = 0; p < P; ++p) qb = qb + q[p] * b[p];
    for (int i = 0; i < P; ++i) {
        float z = 0.0f;
        for (int k = i; k < P; ++k) z = z + L[LI(k, i)] * q[k];
        qgq = qgq + z * z;
    }
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INF_F : (wn - (2.0f * sc) * qb) + rec;
}

template <bool FULL, int P>
static __device__ float decoded_err(const float* x, const float* sh, const float* A, int seed, int e, const int* q) {
    unsigned short st = (unsigned short) seed;
    const float sc = kR32 * pow2f(e);
    float wv[8];
    for (int c = 0; c < 8; ++c) {
        int isum = 0;
        for (int p = 0; p < P; ++p) { st = lfsr_step_d(st); isum += ((int) st - 32768) * q[p]; }
        wv[c] = f2bf_f((float) isum * sc);
    }
    float er = 0.0f;
    for (int c = 0; c < 8; ++c) {
        float v;
        if (FULL) { v = 0.0f; for (int k = 0; k <= c; ++k) v = v + A[c * 8 + k] * wv[k]; }
        else v = sh[c] * wv[c];
        const float d = x[c] - v;
        er = er + d * d;
    }
    return er;
}

template <bool FULL, int P>
__global__ void __launch_bounds__(SP_TPB) k_seed_searchp(SearchPArgs a, const float* W, const float* SH, unsigned short* seed_out,
                                                         unsigned* coef_out, unsigned char* ecode_out, float* err_out) {
    __shared__ float tab[SP_NCH * SP_ENT];
    const int tid = threadIdx.x;
    const int g = a.g0 + (int) blockIdx.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15, nn = P == 8 ? 6561 : 27;
    float sh[8], A[FULL ? 64 : 1];
    for (int c = 0; c < 8; ++c) sh[c] = FULL ? 1.0f : SH[g * 8 + c];
    if (FULL)
        for (int k = 0; k < 64; ++k) A[k] = SH[(size_t) g * 64 + k];
    float x[SP_BPT][8], wn[SP_BPT], best[SP_BPT];
    int bs[SP_BPT], be[SP_BPT], row[SP_BPT];
    unsigned bw[SP_BPT];   // the winner's coefficients (as the C reference recomputes them from its table entry)
    for (int j = 0; j < SP_BPT; ++j) {
        row[j] = (int) blockIdx.y * SP_ROWS + j * SP_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        for (int c = 0; c < 8; ++c) {
            if (FULL) {
                float acc = 0.0f;
                for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * W[(size_t) r * a.cols + g * 8 + k];
                x[j][c] = acc;
            } else x[j][c] = W[(size_t) r * a.cols + g * 8 + c] * sh[c];
        }
        float n = 0.0f;
        for (int c = 0; c < 8; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INF_F; bs[j] = 1; be[j] = lo; bw[j] = 0;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += SP_NCH) {
        __syncthreads();
        if (tid < SP_NCH && s0 + tid <= a.n_seeds) seed_entry<FULL, P>(s0 + tid, sh, A, tab + tid * SP_ENT);
        __syncthreads();
        const int nk = min(SP_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            const float* U = tab + k * SP_ENT;
            if (U[SP_OK] == 0.0f) continue;
            const float* L = U + 8 * P, *D = L + P * (P + 1) / 2;
            for (int j = 0; j < SP_BPT; ++j) {
                float b[P];
                for (int p = 0; p < P; ++p) b[p] = 0.0f;
                for (int c = 0; c < 8; ++c)
                    for (int p = 0; p < P; ++p) b[p] = b[p] + U[P * c + p] * x[j][c];
                float t[P];
                solvep<P>(L, D, b, t);
                float m = 0.0f;
                for (int p = 0; p < P; ++p) m = fmaxf(m, fabsf(t[p]));
                int e0 = flog2f(m) - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[P];
                    for (int p = 0; p < P; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    const float er = cand_err<P>(L, q, b, wn[j], sc);
                    if (er < best[j]) {
                        best[j] = er; bs[j] = s0 + k; be[j] = e;
                        unsigned w = 0;
                        for (int p = 0; p < P; ++p) w |= (unsigned) ((int) q[p] & 15) << (4 * p);
                        bw[j] = w;
                    }
                }
            }
        }
    }
    __syncthreads();
    for (int j = 0; j < SP_BPT; ++j) {
        const int e = be[j];
        int q[P], bq[P];
        for (int p = 0; p < P; ++p) bq[p] = q[p] = (int) (bw[j] << (28 - 4 * p)) >> 28;
        float bde = decoded_err<FULL, P>(x[j], sh, A, bs[j], e, q);
        if (a.refit)
            for (int d = 0; d < nn; ++d) {
                int cq[P], dd = d;
                bool okq = true;
                for (int p = 0; p < P; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq = okq && cq[p] >= -8 && cq[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err<FULL, P>(x[j], sh, A, bs[j], e, cq);
                if (ee < bde) { bde = ee; for (int p = 0; p < P; ++p) bq[p] = cq[p]; }
            }
        if (row[j] >= a.rows) continue;
        const size_t k = (size_t) row[j] * (size_t) ng + (size_t) g;
        unsigned cw = 0;
        for (int p = 0; p < P; ++p) cw |= (unsigned) (bq[p] & 15) << (4 * p);
        seed_out[k] = (unsigned short) bs[j];
        coef_out[k] = cw;
        ecode_out[k] = (unsigned char) (e - a.bias);
        err_out[k] = bde;
    }
}

// ---- the host side (searchp_gpu.h) ---------------------------------------------------------------------------------

struct NslmPGpu {
    cudaStream_t st;
};

extern "C" {

NslmPGpu* nslmp_gpu_open(const char* library, char* err, int errlen) {
    (void) library;   // the kernel is linked in (Metal loads it from a library file)
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return NULL; }
    NslmPGpu* g = (NslmPGpu*) calloc(1, sizeof(NslmPGpu));
    if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) { snprintf(err, (size_t) errlen, "CUDA stream"); free(g); return NULL; }
    return g;
}

int nslmp_gpu_search(NslmPGpu* g, int P, const float* w, int rows, int cols, const float* sh, const float* A, int bias,
                     const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen) {
    if (cols % 8 || rows < 1 || o->n_exp < 1 || o->n_exp > 3 || (P != 3 && P != 8)) { snprintf(msg, (size_t) msglen, "bad shape or options"); return -1; }
    const int ng = cols / 8;
    const size_t nb = (size_t) rows * ng, nsh = A ? (size_t) 64 * ng : (size_t) cols;   // SH: sqrt(h) per column, or A per group
    float *W = NULL, *SH = NULL, *Ro = NULL;
    unsigned short* So = NULL;
    unsigned* Co = NULL;
    unsigned char* Eo = NULL;
    float* shp = (float*) malloc(sizeof(float) * nsh);
    if (A) memcpy(shp, A, sizeof(float) * nsh);
    else for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;
    int rc = -1;
    if (cudaMalloc((void**) &W, sizeof(float) * (size_t) rows * cols) || cudaMalloc((void**) &SH, sizeof(float) * nsh) ||
        cudaMalloc((void**) &So, 2 * nb) || cudaMalloc((void**) &Co, 4 * nb) || cudaMalloc((void**) &Eo, nb) || cudaMalloc((void**) &Ro, 4 * nb)) {
        snprintf(msg, (size_t) msglen, "out of GPU memory");
        goto done;
    }
    cudaMemcpyAsync(W, w, sizeof(float) * (size_t) rows * cols, cudaMemcpyHostToDevice, g->st);
    cudaMemcpyAsync(SH, shp, sizeof(float) * nsh, cudaMemcpyHostToDevice, g->st);
    {
        // column groups per launch: about 5e9 block-seeds at P = 3 (1.5e9 at P = 8), so no launch runs long
        int gstep = (int) ((P == 8 ? 1.5e9 : 5e9) / ((double) rows * o->n_seeds));
        if (gstep < 1) gstep = 1;
        if (gstep > ng) gstep = ng;
        for (int g0 = 0; g0 < ng; g0 += gstep) {
            const int gn = ng - g0 < gstep ? ng - g0 : gstep;
            SearchPArgs a = {rows, cols, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}};
            const dim3 grid((unsigned) gn, (unsigned) ((rows + SP_ROWS - 1) / SP_ROWS));
            if (P == 8) {
                if (A) k_seed_searchp<true, 8><<<grid, SP_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
                else k_seed_searchp<false, 8><<<grid, SP_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
            } else {
                if (A) k_seed_searchp<true, 3><<<grid, SP_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
                else k_seed_searchp<false, 3><<<grid, SP_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
            }
        }
    }
    cudaMemcpyAsync(seed, So, 2 * nb, cudaMemcpyDeviceToHost, g->st);
    cudaMemcpyAsync(coef, Co, 4 * nb, cudaMemcpyDeviceToHost, g->st);
    cudaMemcpyAsync(ecode, Eo, nb, cudaMemcpyDeviceToHost, g->st);
    if (err) cudaMemcpyAsync(err, Ro, 4 * nb, cudaMemcpyDeviceToHost, g->st);
    {   // a block: the gotos above must not cross an initialisation (C++)
        cudaError_t ce = cudaStreamSynchronize(g->st);
        if (ce == cudaSuccess) ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            snprintf(msg, (size_t) msglen, "GPU error: %s", cudaGetErrorString(ce));
            goto done;
        }
    }
    rc = 0;
done:
    cudaFree(W); cudaFree(SH); cudaFree(So); cudaFree(Co); cudaFree(Eo); cudaFree(Ro);
    free(shp);
    return rc;
}

void nslmp_gpu_close(NslmPGpu* g) {
    if (!g) return;
    cudaStreamDestroy(g->st);
    free(g);
}

}   // extern "C"
