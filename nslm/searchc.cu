// nslm/searchc.cu - the seed search of nslm/searchc.h on CUDA (nslm/searchc_gpu.h): every expression is
// nslm/lib_searchc.c's, in the same order.  Compiled with --fmad=false; square roots and divisions are IEEE.
// Grid (column groups from g0, row tiles of SC_ROWS), SC_TPB threads, SC_BPT rows per thread; the seed table is built
// SC_NCH seeds at a time in shared memory, one entry per seed: U[C P], L[P (P + 1) / 2], 1/diag[P], ok.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "searchc_gpu.h"

#define MAGIC 12582912.0f
#define kR32 (1.0f / 32767.0f)
#define INF_F __int_as_float(0x7f800000)
#define LI(i, j) ((i) * ((i) + 1) / 2 + (j))
#define SC_NCH 32
#define SC_TPB 256
#define SC_BPT 2
#define SC_ROWS (SC_TPB * SC_BPT)

typedef struct { int rows, cols, g0, bias, n_seeds, n_exp, refit, exp_delta[3]; } SearchCArgs;   // cols: padded, C | cols

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

template <int C, int P> struct Ent { enum { L = C * P, D = C * P + P * (P + 1) / 2, OK = C * P + P * (P + 1) / 2 + P, N = OK + 1 }; };

template <int C, int P>
static __device__ void seed_entry(int s, const float* sh, float* ent) {
    float* U = ent, *L = ent + Ent<C, P>::L, *D = ent + Ent<C, P>::D;
    unsigned short st = (unsigned short) s;
    for (int c = 0; c < C; ++c)
        for (int p = 0; p < P; ++p) { st = lfsr_step_d(st); U[P * c + p] = ((float) ((int) st - 32768) * kR32) * sh[c]; }
    ent[Ent<C, P>::OK] = 0.0f;
    for (int i = 0; i < P; ++i)
        for (int j = 0; j <= i; ++j) {
            float g = 0.0f;
            for (int c = 0; c < C; ++c) g = g + U[P * c + i] * U[P * c + j];
            for (int k = 0; k < j; ++k) g = g - L[LI(i, k)] * L[LI(j, k)];
            if (i == j) {
                if (!(g > 0.0f)) return;
                L[LI(i, i)] = __fsqrt_rn(g);
                D[i] = __fdiv_rn(1.0f, L[LI(i, i)]);
            } else L[LI(i, j)] = g * D[j];
        }
    ent[Ent<C, P>::OK] = 1.0f;
}

template <int C, int P>
static __device__ float decoded_err(const float* x, const float* sh, int seed, int e, const int* q) {
    unsigned short st = (unsigned short) seed;
    const float sc = kR32 * pow2f(e);
    float er = 0.0f;
    for (int c = 0; c < C; ++c) {
        int isum = 0;
        for (int p = 0; p < P; ++p) { st = lfsr_step_d(st); isum += ((int) st - 32768) * q[p]; }
        const float d = x[c] - sh[c] * f2bf_f((float) isum * sc);
        er = er + d * d;
    }
    return er;
}

template <int C, int P>
__global__ void __launch_bounds__(SC_TPB) k_seed_searchc(SearchCArgs a, const float* W, const float* SH, unsigned short* seed_out,
                                                         unsigned* coef_out, unsigned char* ecode_out, float* err_out) {
    __shared__ float tab[SC_NCH * Ent<C, P>::N];
    const int tid = threadIdx.x, g = a.g0 + (int) blockIdx.x, ng = a.cols / C, lo = a.bias, hi = a.bias + 15;
    int nn = 1;
    for (int p = 0; p < P; ++p) nn *= 3;
    float sh[C], x[SC_BPT][C], wn[SC_BPT], best[SC_BPT];
    int bs[SC_BPT], be[SC_BPT], row[SC_BPT];
    unsigned bw[SC_BPT];   // the winner's coefficients (as the C reference recomputes them from its entry)
    for (int c = 0; c < C; ++c) sh[c] = SH[g * C + c];
    for (int j = 0; j < SC_BPT; ++j) {
        row[j] = (int) blockIdx.y * SC_ROWS + j * SC_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        float n = 0.0f;
        for (int c = 0; c < C; ++c) x[j][c] = W[(size_t) r * a.cols + g * C + c] * sh[c];
        for (int c = 0; c < C; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INF_F; bs[j] = 1; be[j] = lo; bw[j] = 0;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += SC_NCH) {
        __syncthreads();
        if (tid < SC_NCH && s0 + tid <= a.n_seeds) seed_entry<C, P>(s0 + tid, sh, tab + tid * Ent<C, P>::N);
        __syncthreads();
        const int nk = min(SC_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            const float* U = tab + k * Ent<C, P>::N, *L = U + Ent<C, P>::L, *D = U + Ent<C, P>::D;
            if (U[Ent<C, P>::OK] == 0.0f) continue;
            for (int j = 0; j < SC_BPT; ++j) {
                float b[P], y[P], t[P];
                for (int p = 0; p < P; ++p) b[p] = 0.0f;
                for (int c = 0; c < C; ++c)
                    for (int p = 0; p < P; ++p) b[p] = b[p] + U[P * c + p] * x[j][c];
                for (int i = 0; i < P; ++i) {   // t = (U^T U)^-1 b: L y = b, L^T t = y
                    float v = b[i];
                    for (int m = 0; m < i; ++m) v = v - L[LI(i, m)] * y[m];
                    y[i] = v * D[i];
                }
                for (int i = P - 1; i >= 0; --i) {
                    float v = y[i];
                    for (int m = i + 1; m < P; ++m) v = v - L[LI(m, i)] * t[m];
                    t[i] = v * D[i];
                }
                float mx = 0.0f;
                for (int p = 0; p < P; ++p) mx = fmaxf(mx, fabsf(t[p]));
                int e0 = flog2f(mx) - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[P], qb = 0.0f, qgq = 0.0f;
                    for (int p = 0; p < P; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    for (int p = 0; p < P; ++p) qb = qb + q[p] * b[p];
                    for (int i = 0; i < P; ++i) {   // z_i = (L^T q)_i
                        float z = 0.0f;
                        for (int m = i; m < P; ++m) z = z + L[LI(m, i)] * q[m];
                        qgq = qgq + z * z;
                    }
                    const float rec = (sc * sc) * qgq, er = rec > 4.0f * wn[j] ? INF_F : (wn[j] - (2.0f * sc) * qb) + rec;
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
    for (int j = 0; j < SC_BPT; ++j) {
        const int e = be[j];
        int q[P], bq[P];
        for (int p = 0; p < P; ++p) bq[p] = q[p] = (int) (bw[j] << (28 - 4 * p)) >> 28;
        float bde = decoded_err<C, P>(x[j], sh, bs[j], e, q);
        if (a.refit)
            for (int d = 0; d < nn; ++d) {   // the 3^P neighbours of q by the decoded error
                int cq[P], dd = d;
                bool okq = true;
                for (int p = 0; p < P; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq = okq && cq[p] >= -8 && cq[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err<C, P>(x[j], sh, bs[j], e, cq);
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

// ---- the host side (searchc_gpu.h) ---------------------------------------------------------------------------------

struct NslmCGpu {
    cudaStream_t st;
};

extern "C" {

NslmCGpu* nslmc_gpu_open(char* err, int errlen) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return NULL; }
    NslmCGpu* g = (NslmCGpu*) calloc(1, sizeof(NslmCGpu));
    if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) { snprintf(err, (size_t) errlen, "CUDA stream"); free(g); return NULL; }
    return g;
}

int nslmc_gpu_search(NslmCGpu* g, int C, int P, const float* w, int rows, int cols, const float* sh, int bias, const Search4Opts* o,
                     uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err, char* msg, int msglen) {
    const int lay = C == 12 && P == 4 ? 0 : C == 16 && P == 7 ? 1 : C == 8 && P == 3 ? 2 : -1;
    if (lay < 0 || rows < 1 || cols < 1 || o->n_exp < 1 || o->n_exp > 3) { snprintf(msg, (size_t) msglen, "bad layout, shape or options"); return -1; }
    const int ng = (cols + C - 1) / C, cp = ng * C;   // padded columns
    const size_t nb = (size_t) rows * ng;
    float* wp = (float*) calloc((size_t) rows * cp, sizeof(float)), *shp = (float*) calloc((size_t) cp, sizeof(float));
    for (int r = 0; r < rows; ++r) memcpy(wp + (size_t) r * cp, w + (size_t) r * cols, sizeof(float) * (size_t) cols);
    for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;
    float *W = NULL, *SH = NULL, *Ro = NULL;
    unsigned short* So = NULL;
    unsigned* Co = NULL;
    unsigned char* Eo = NULL;
    int rc = -1;
    if (cudaMalloc((void**) &W, sizeof(float) * (size_t) rows * cp) || cudaMalloc((void**) &SH, sizeof(float) * (size_t) cp) ||
        cudaMalloc((void**) &So, 2 * nb) || cudaMalloc((void**) &Co, 4 * nb) || cudaMalloc((void**) &Eo, nb) || cudaMalloc((void**) &Ro, 4 * nb)) {
        snprintf(msg, (size_t) msglen, "out of GPU memory");
        goto done;
    }
    cudaMemcpyAsync(W, wp, sizeof(float) * (size_t) rows * cp, cudaMemcpyHostToDevice, g->st);
    cudaMemcpyAsync(SH, shp, sizeof(float) * (size_t) cp, cudaMemcpyHostToDevice, g->st);
    {
        // column groups per launch: about 1.2e11 / (C P) block-seeds, so no launch runs long
        int gstep = (int) (1.2e11 / (C * P) / ((double) rows * o->n_seeds));
        if (gstep < 1) gstep = 1;
        if (gstep > ng) gstep = ng;
        for (int g0 = 0; g0 < ng; g0 += gstep) {
            const int gn = ng - g0 < gstep ? ng - g0 : gstep;
            SearchCArgs a = {rows, cp, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}};
            const dim3 grid((unsigned) gn, (unsigned) ((rows + SC_ROWS - 1) / SC_ROWS));
            if (lay == 0) k_seed_searchc<12, 4><<<grid, SC_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
            else if (lay == 1) k_seed_searchc<16, 7><<<grid, SC_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
            else k_seed_searchc<8, 3><<<grid, SC_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
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
    free(wp);
    free(shp);
    return rc;
}

void nslmc_gpu_close(NslmCGpu* g) {
    if (!g) return;
    cudaStreamDestroy(g->st);
    free(g);
}

}   // extern "C"
