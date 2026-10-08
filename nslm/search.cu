// nslm/search.cu - the P = 3 seed search on CUDA (contract: nslm/search_gpu.h), the counterpart of nslm/search.metal.
//
// The seed table (U(s) * sqrt(h), inverse Gram, Cholesky factor) is built per column group by k_seedtab in IEEE double,
// with nslm/lib_search.c seedtab_one's operations in its order (Metal has no double and emulates it; CUDA has it), so
// every entry is the CPU's bit for bit.  The search streams the table SG_NCH seeds at a time through shared memory and
// evaluates every candidate with the CPU's f32 expressions (compiled with --fmad=false: no contraction).
//
// Prune: LS(s) = |x|^2 - b^T Gi b (b = U^T x), the least-squares residual, lower-bounds any quantized candidate.
// Computed with explicit FMAs; a seed is skipped only if LS(s) > best + (pm_s + 1e-5) |x|^2, pm_s = 4e-6 (sum |G|)
// (sum |Gi|), so the arg-min and every running best are the CPU's.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "lfsr.h"
#include "search_gpu.h"

#define MAGIC 12582912.0f
#define INF_F __int_as_float(0x7f800000)
#define kR32 __int_as_float(0x38000100)   // fl(1 / 32767), NSLM_R32

static __device__ __forceinline__ float pow2f(int e) { return __uint_as_float((unsigned) (e + 127) << 23); }
static __device__ __forceinline__ float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }

// The 24 states after seed s from the per-seed stream table (nslm/lfsr.h lfsr_stream24).
static __device__ __forceinline__ void states(unsigned s, const unsigned* G, unsigned* v) {
    const unsigned g = G[s], lo = s | (g << 16);
    for (int k = 1; k <= 16; ++k) v[k - 1] = (lo >> k) & 0xFFFFu;
    for (int k = 17; k <= 24; ++k) v[k - 1] = (g >> (k - 16)) & 0xFFFFu;
}
// U(s) scaled by sh, exactly as seedtab_one: fl(fl(S * R32) * sh_c).  With a full transform A: U' = A U, rows summed
// in order k = 0 .. c (A lower triangular).
static __device__ void scaled_u(unsigned s, const unsigned* G, const float* sh, unsigned* v, float* U, int full, const float* A) {
    states(s, G, v);
    if (!full) {
        for (int k = 0; k < 24; ++k) { U[k] = (float) ((int) v[k] - 32768) * kR32; U[k] = U[k] * sh[k / 3]; }
        return;
    }
    float u0[24];
    for (int k = 0; k < 24; ++k) u0[k] = (float) ((int) v[k] - 32768) * kR32;
    for (int c = 0; c < 8; ++c)
        for (int p = 0; p < 3; ++p) {
            float acc = 0.0f;
            for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[k * 3 + p];
            U[c * 3 + p] = acc;
        }
}

// seedtab_one's double block, operation for operation: Gi and R rounded to f32 (Gi zero when not positive definite),
// and the prune margin.
static __device__ void gram(const float* U, float* gi, float* R, float* pm) {
    double g[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double a = 0;
            for (int c = 0; c < 8; ++c) a = __dadd_rn(a, __dmul_rn((double) U[c * 3 + i], (double) U[c * 3 + j]));
            g[i][j] = a;
        }
    const double det = __dadd_rn(__dsub_rn(__dmul_rn(g[0][0], __dsub_rn(__dmul_rn(g[1][1], g[2][2]), __dmul_rn(g[1][2], g[2][1]))),
                                           __dmul_rn(g[0][1], __dsub_rn(__dmul_rn(g[1][0], g[2][2]), __dmul_rn(g[1][2], g[2][0])))),
                                 __dmul_rn(g[0][2], __dsub_rn(__dmul_rn(g[1][0], g[2][1]), __dmul_rn(g[1][1], g[2][0]))));
    const double r00 = __dsqrt_rn(g[0][0]), r01 = __ddiv_rn(g[0][1], r00), r02 = __ddiv_rn(g[0][2], r00);
    const double r11 = __dsqrt_rn(fmax(__dsub_rn(g[1][1], __dmul_rn(r01, r01)), 0.0));
    const double r12 = r11 > 0 ? __ddiv_rn(__dsub_rn(g[1][2], __dmul_rn(r01, r02)), r11) : 0.0;
    const double r22 = __dsqrt_rn(fmax(__dsub_rn(__dsub_rn(g[2][2], __dmul_rn(r02, r02)), __dmul_rn(r12, r12)), 0.0));
    R[0] = (float) r00; R[1] = (float) r01; R[2] = (float) r02; R[3] = (float) r11; R[4] = (float) r12; R[5] = (float) r22;
    for (int k = 0; k < 6; ++k) gi[k] = 0.0f;
    if (det > 0) {
        gi[0] = (float) __ddiv_rn(__dsub_rn(__dmul_rn(g[1][1], g[2][2]), __dmul_rn(g[1][2], g[2][1])), det);
        gi[1] = (float) __ddiv_rn(__dsub_rn(__dmul_rn(g[0][2], g[2][1]), __dmul_rn(g[0][1], g[2][2])), det);
        gi[2] = (float) __ddiv_rn(__dsub_rn(__dmul_rn(g[0][1], g[1][2]), __dmul_rn(g[0][2], g[1][1])), det);
        gi[3] = (float) __ddiv_rn(__dsub_rn(__dmul_rn(g[0][0], g[2][2]), __dmul_rn(g[0][2], g[2][0])), det);
        gi[4] = (float) __ddiv_rn(__dsub_rn(__dmul_rn(g[0][2], g[1][0]), __dmul_rn(g[0][0], g[1][2])), det);
        gi[5] = (float) __ddiv_rn(__dsub_rn(__dmul_rn(g[0][0], g[1][1]), __dmul_rn(g[0][1], g[1][0])), det);
    }
    float sg = 0.0f;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) sg += fabsf((float) g[i][j]);
    const float si = fabsf(gi[0]) + fabsf(gi[3]) + fabsf(gi[5]) + 2.0f * (fabsf(gi[1]) + fabsf(gi[2]) + fabsf(gi[4]));
    *pm = 4e-6f * sg * si + 1e-5f;
}

// The table of column group g0 + blockIdx.x: entries [x][seed][SG_ENT] = U[24], Gi[6], R[6], prune margin.
// Grid (column groups, 65536 / 256), 256 threads.  full: A for column group x at SH[64 x]; else sqrt(h) at SH[8 g].
__global__ void __launch_bounds__(256) k_seedtab(SearchArgs a, const float* SH, const unsigned* G, float* tab) {
    const unsigned s = blockIdx.y * 256 + threadIdx.x;
    float* e = tab + ((size_t) blockIdx.x * 65536 + s) * SG_ENT;
    if (s < 1u || s > (unsigned) a.n_seeds) {
        if (s < 65536u) for (int k = 0; k < SG_ENT; ++k) e[k] = 0.0f;
        return;
    }
    const int g = a.g0 + (int) blockIdx.x;
    float sh[8], A[64];
    for (int c = 0; c < 8; ++c) sh[c] = a.full ? 1.0f : SH[g * 8 + c];
    for (int k = 0; k < 64; ++k) A[k] = a.full ? SH[blockIdx.x * 64 + k] : 0.0f;
    unsigned v[24];
    float U[24], gi[6], R[6], pm;
    scaled_u(s, G, sh, v, U, a.full, A);
    gram(U, gi, R, &pm);
    for (int k = 0; k < 24; ++k) e[k] = U[k];
    for (int k = 0; k < 6; ++k) { e[24 + k] = gi[k]; e[30 + k] = R[k]; }
    e[36] = pm;
    e[37] = e[38] = e[39] = 0.0f;
}

// The CPU's decoded error of a (seed, exponent, q) candidate: BF16 decode exactly as nslm_decode_block, then
// sum_c (x_c - sh_c w_hat_c)^2 in order (full: (A w_hat)_c).
static __device__ float decoded_err(const float* x, const float* sh, const unsigned* v, int e, int q0, int q1, int q2, int full, const float* A) {
    const float sc = kR32 * pow2f(e);
    float bf[8];
    for (int c = 0; c < 8; ++c) {
        const int isum = ((int) v[c * 3] - 32768) * q0 + ((int) v[c * 3 + 1] - 32768) * q1 + ((int) v[c * 3 + 2] - 32768) * q2;
        unsigned u = __float_as_uint((float) isum * sc);
        u += 0x7FFFu + ((u >> 16) & 1u);
        bf[c] = __uint_as_float(u & 0xFFFF0000u);
    }
    float er = 0.0f;
    for (int c = 0; c < 8; ++c) {
        float ab;
        if (full) { ab = 0.0f; for (int k = 0; k <= c; ++k) ab = ab + A[c * 8 + k] * bf[k]; }
        else ab = sh[c] * bf[c];
        const float d = x[c] - ab;
        er = er + d * d;
    }
    return er;
}

__global__ void __launch_bounds__(SG_TPB) k_seed_search(SearchArgs a, const float* W, const float* SH, const unsigned* G,
                                                        const float* TAB, unsigned short* seed_out, unsigned short* nib_out, float* err_out) {
    __shared__ float tab[SG_NCH * SG_ENT];
    const int tid = threadIdx.x;
    const float* gtab = TAB + (size_t) blockIdx.x * 65536 * SG_ENT;
    const int g = a.g0 + (int) blockIdx.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15;
    const int full = a.full;
    float sh[8], A[64];
    for (int c = 0; c < 8; ++c) sh[c] = full ? 1.0f : SH[g * 8 + c];
    for (int k = 0; k < 64; ++k) A[k] = full ? SH[blockIdx.x * 64 + k] : 0.0f;
    float x[SG_BPT][8], wn[SG_BPT], best[SG_BPT];
    int bs[SG_BPT], be[SG_BPT], row[SG_BPT];
    for (int j = 0; j < SG_BPT; ++j) {
        row[j] = (int) blockIdx.y * SG_ROWS + j * SG_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        if (full) {
            float w[8];
            for (int c = 0; c < 8; ++c) w[c] = W[(size_t) r * a.cols + g * 8 + c];
            for (int c = 0; c < 8; ++c) { float acc = 0.0f; for (int k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * w[k]; x[j][c] = acc; }
        } else
            for (int c = 0; c < 8; ++c) x[j][c] = W[(size_t) r * a.cols + g * 8 + c] * sh[c];
        float n = 0.0f;
        for (int c = 0; c < 8; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INF_F; bs[j] = 1; be[j] = lo;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += SG_NCH) {
        const int nk = min(SG_NCH, a.n_seeds - s0 + 1);
        __syncthreads();
        for (int i = tid; i < nk * SG_ENT; i += SG_TPB) tab[i] = gtab[(size_t) s0 * SG_ENT + i];
        __syncthreads();
        for (int k = 0; k < nk; ++k) {
            const float* T = tab + k * SG_ENT;
            const float* U = T;
            const float i00 = T[24], i01 = T[25], i02 = T[26], i11 = T[27], i12 = T[28], i22 = T[29], pm = T[36];
            for (int j = 0; j < SG_BPT; ++j) {
                if (a.prune) {   // fused lower bound: skip seeds that cannot reach the running best
                    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
                    for (int c = 0; c < 8; ++c) { b0 = fmaf(U[c * 3], x[j][c], b0); b1 = fmaf(U[c * 3 + 1], x[j][c], b1); b2 = fmaf(U[c * 3 + 2], x[j][c], b2); }
                    const float t0 = fmaf(i00, b0, fmaf(i01, b1, i02 * b2));
                    const float t1 = fmaf(i01, b0, fmaf(i11, b1, i12 * b2));
                    const float t2 = fmaf(i02, b0, fmaf(i12, b1, i22 * b2));
                    const float ls = wn[j] - fmaf(b0, t0, fmaf(b1, t1, b2 * t2));
                    if (ls > best[j] + pm * wn[j]) continue;
                }
                // the CPU's candidate loop, expression for expression (nslm_search_ref)
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
                for (int c = 0; c < 8; ++c) {
                    b0 = b0 + U[c * 3] * x[j][c];
                    b1 = b1 + U[c * 3 + 1] * x[j][c];
                    b2 = b2 + U[c * 3 + 2] * x[j][c];
                }
                const float t0 = (i00 * b0 + i01 * b1) + i02 * b2;
                const float t1 = (i01 * b0 + i11 * b1) + i12 * b2;
                const float t2 = (i02 * b0 + i12 * b1) + i22 * b2;
                const float m = fmaxf(fmaxf(fabsf(t0), fabsf(t1)), fabsf(t2));
                int e0 = (int) ((__float_as_uint(m) >> 23) & 255u) - 127 - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                const float r00 = T[30], r01 = T[31], r02 = T[32], r11 = T[33], r12 = T[34], r22 = T[35];
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    const float q0 = clampq((t0 * inv + MAGIC) - MAGIC);
                    const float q1 = clampq((t1 * inv + MAGIC) - MAGIC);
                    const float q2 = clampq((t2 * inv + MAGIC) - MAGIC);
                    const float qb = (q0 * b0 + q1 * b1) + q2 * b2;
                    const float z0 = (r00 * q0 + r01 * q1) + r02 * q2, z1 = r11 * q1 + r12 * q2, z2 = r22 * q2;
                    const float qgq = (z0 * z0 + z1 * z1) + z2 * z2;
                    const float rec = (sc * sc) * qgq;
                    const float er = rec > 4.0f * wn[j] ? INF_F : (wn[j] - (2.0f * sc) * qb) + rec;
                    if (er < best[j]) { best[j] = er; bs[j] = s0 + k; be[j] = e; }
                }
            }
        }
    }
    // finish_block: the winner's entry from the table, then plain rounding or the best of the 3^P neighbourhood by
    // decoded error
    for (int j = 0; j < SG_BPT; ++j) {
        if (row[j] >= a.rows) continue;
        const float* T = gtab + (size_t) bs[j] * SG_ENT;
        unsigned v[24];
        states((unsigned) bs[j], G, v);
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
        for (int c = 0; c < 8; ++c) { b0 = b0 + T[c * 3] * x[j][c]; b1 = b1 + T[c * 3 + 1] * x[j][c]; b2 = b2 + T[c * 3 + 2] * x[j][c]; }
        const float t0 = (T[24] * b0 + T[25] * b1) + T[26] * b2;
        const float t1 = (T[25] * b0 + T[27] * b1) + T[28] * b2;
        const float t2 = (T[26] * b0 + T[28] * b1) + T[29] * b2;
        const int e = be[j];
        const float inv = pow2f(-e);
        const int q0 = (int) clampq((t0 * inv + MAGIC) - MAGIC), q1 = (int) clampq((t1 * inv + MAGIC) - MAGIC),
                  q2 = (int) clampq((t2 * inv + MAGIC) - MAGIC);
        int bq0 = q0, bq1 = q1, bq2 = q2;
        float bde = decoded_err(x[j], sh, v, e, q0, q1, q2, full, A);
        if (a.refit) {
            for (int d0 = -1; d0 <= 1; ++d0)
                for (int d1 = -1; d1 <= 1; ++d1)
                    for (int d2 = -1; d2 <= 1; ++d2) {
                        const int c0 = q0 + d0, c1 = q1 + d1, c2 = q2 + d2;
                        if (c0 < -8 || c0 > 7 || c1 < -8 || c1 > 7 || c2 < -8 || c2 > 7) continue;
                        const float ee = decoded_err(x[j], sh, v, e, c0, c1, c2, full, A);
                        if (ee < bde) { bde = ee; bq0 = c0; bq1 = c1; bq2 = c2; }
                    }
        }
        const size_t k = (size_t) row[j] * (size_t) ng + (size_t) g;
        seed_out[k] = (unsigned short) bs[j];
        nib_out[k] = (unsigned short) (((e - a.bias) & 15) | ((bq0 & 15) << 4) | ((bq1 & 15) << 8) | ((bq2 & 15) << 12));
        err_out[k] = bde;
    }
}

// ---- the host side (search_gpu.h) -----------------------------------------------------------------------------------

#define GSTEP_MAX 16   // column groups per batch: the tables take 65536 x 160 bytes each

struct NslmGpu {
    cudaStream_t st;
    unsigned* G;   // per-seed stream table (lfsr_stream24)
    float* tab;    // GSTEP_MAX column groups' tables
};

static int run(NslmGpu* g, const float* w, int rows, int cols, const float* shbuf, int nsh, int bias, const SearchOpts* o, int prune,
               int full, uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen) {
    const int ng = cols / 8;
    const size_t nb = (size_t) rows * ng;
    float *W = NULL, *SH = NULL, *Eo = NULL;
    unsigned short *So = NULL, *No = NULL;
    int rc = -1;
    if (cudaMalloc((void**) &W, sizeof(float) * (size_t) rows * cols) || cudaMalloc((void**) &SH, sizeof(float) * (size_t) nsh) ||
        cudaMalloc((void**) &So, 2 * nb) || cudaMalloc((void**) &No, 2 * nb) || cudaMalloc((void**) &Eo, 4 * nb)) {
        snprintf(msg, (size_t) msglen, "out of GPU memory");
        goto done;
    }
    cudaMemcpyAsync(W, w, sizeof(float) * (size_t) rows * cols, cudaMemcpyHostToDevice, g->st);
    cudaMemcpyAsync(SH, shbuf, sizeof(float) * (size_t) nsh, cudaMemcpyHostToDevice, g->st);
    for (int g0 = 0; g0 < ng; g0 += GSTEP_MAX) {
        const int gn = ng - g0 < GSTEP_MAX ? ng - g0 : GSTEP_MAX;
        SearchArgs a = {rows, cols, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}, prune,
                        0.0f, full};
        k_seedtab<<<dim3((unsigned) gn, 256), 256, 0, g->st>>>(a, SH, g->G, g->tab);
        // row tiles of one batch per launch keep each launch short (display watchdogs)
        const int rtiles = (rows + SG_ROWS - 1) / SG_ROWS;
        k_seed_search<<<dim3((unsigned) gn, (unsigned) rtiles), SG_TPB, 0, g->st>>>(a, W, SH, g->G, g->tab, So, No, Eo);
    }
    cudaMemcpyAsync(seed, So, 2 * nb, cudaMemcpyDeviceToHost, g->st);
    cudaMemcpyAsync(nib, No, 2 * nb, cudaMemcpyDeviceToHost, g->st);
    if (err) cudaMemcpyAsync(err, Eo, 4 * nb, cudaMemcpyDeviceToHost, g->st);
    if (cudaStreamSynchronize(g->st) != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        snprintf(msg, (size_t) msglen, "GPU error: %s", cudaGetErrorString(cudaGetLastError()));
        goto done;
    }
    rc = 0;
done:
    cudaFree(W); cudaFree(SH); cudaFree(So); cudaFree(No); cudaFree(Eo);
    return rc;
}

extern "C" {

NslmGpu* nslm_gpu_open(const char* library, char* err, int errlen) {
    (void) library;   // the kernels are linked in (Metal loads them from a library file)
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return NULL; }
    NslmGpu* g = (NslmGpu*) calloc(1, sizeof(NslmGpu));
    unsigned* t = (unsigned*) malloc(65536 * 4);
    t[0] = 0;
    for (int s = 1; s < 65536; ++s) t[s] = lfsr_stream24((uint16_t) s);
    const int ok = cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) == cudaSuccess && cudaMalloc((void**) &g->G, 65536 * 4) == cudaSuccess &&
                   cudaMalloc((void**) &g->tab, (size_t) GSTEP_MAX * 65536 * SG_ENT * 4) == cudaSuccess &&
                   cudaMemcpy(g->G, t, 65536 * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    free(t);
    if (!ok) { snprintf(err, (size_t) errlen, "CUDA setup failed"); nslm_gpu_close(g); return NULL; }
    return g;
}

int nslm_gpu_search(NslmGpu* g, const float* w, int rows, int cols, const float* sh, int bias, const SearchOpts* o,
                    int prune, uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen) {
    if (cols % 8 || rows < 1 || o->n_exp < 1 || o->n_exp > 3) { snprintf(msg, (size_t) msglen, "bad shape or options"); return -1; }
    float* shp = (float*) malloc(sizeof(float) * (size_t) cols);
    for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;   // sqrt(h) = 1: f32-exact no-op
    const int rc = run(g, w, rows, cols, shp, cols, bias, o, prune, 0, seed, nib, err, msg, msglen);
    free(shp);
    return rc;
}

int nslm_gpu_search_block(NslmGpu* g, const float* w8, int rows, const float a[64], int bias, const SearchOpts* o,
                          uint16_t* seed, uint16_t* nib, float* err, char* msg, int msglen) {
    return run(g, w8, rows, 8, a, 64, bias, o, 1, 1, seed, nib, err, msg, msglen);
}

void nslm_gpu_close(NslmGpu* g) {
    if (!g) return;
    cudaFree(g->G);
    cudaFree(g->tab);
    if (g->st) cudaStreamDestroy(g->st);
    free(g);
}

}   // extern "C"
