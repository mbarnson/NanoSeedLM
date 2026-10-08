// nslm/search4.cu - the P = 4 seed search on CUDA (nslm/search4_gpu.h), the counterpart of nslm/search4.metal: every
// expression is nslm/lib_search4.c's, in the same order.  Compiled with --fmad=false (no contraction); square roots
// and divisions are IEEE (__fsqrt_rn, __fdiv_rn): bit-identical results.
// Grid (column groups from a.g0, row tiles of S4_ROWS), S4_TPB threads, S4_BPT rows per thread; the seed table is
// built S4_NCH seeds at a time in shared memory.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include "search4_gpu.h"

#define MAGIC 12582912.0f
#define kR32 (1.0f / 32767.0f)
#define INF_F __int_as_float(0x7f800000)

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

// nslm4_seed_entry: U[32], L[10], 1/diag[4], ok (ent[46])
static __device__ void seed_entry(int s, const float* sh, float* ent) {
    unsigned short st = (unsigned short) s;
    float U[32];
    for (int k = 0; k < 32; ++k) {
        st = lfsr_step_d(st);
        const float u = (float) ((int) st - 32768) * kR32;
        U[k] = u * sh[k / 4];
    }
    float G[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float g = 0.0f;
            for (int c = 0; c < 8; ++c) g = g + U[4 * c + i] * U[4 * c + j];
            G[i][j] = g;
        }
    for (int k = 0; k < 32; ++k) ent[k] = U[k];
    ent[46] = 0.0f;
    const float a00 = G[0][0];
    if (!(a00 > 0.0f)) return;
    const float l00 = __fsqrt_rn(a00);
    const float l10 = __fdiv_rn(G[1][0], l00), l20 = __fdiv_rn(G[2][0], l00), l30 = __fdiv_rn(G[3][0], l00);
    const float a11 = G[1][1] - l10 * l10;
    if (!(a11 > 0.0f)) return;
    const float l11 = __fsqrt_rn(a11);
    const float l21 = __fdiv_rn(G[2][1] - l20 * l10, l11), l31 = __fdiv_rn(G[3][1] - l30 * l10, l11);
    const float a22 = (G[2][2] - l20 * l20) - l21 * l21;
    if (!(a22 > 0.0f)) return;
    const float l22 = __fsqrt_rn(a22);
    const float l32 = __fdiv_rn((G[3][2] - l30 * l20) - l31 * l21, l22);
    const float a33 = ((G[3][3] - l30 * l30) - l31 * l31) - l32 * l32;
    if (!(a33 > 0.0f)) return;
    const float l33 = __fsqrt_rn(a33);
    ent[32] = l00; ent[33] = l10; ent[34] = l11; ent[35] = l20; ent[36] = l21; ent[37] = l22; ent[38] = l30; ent[39] = l31;
    ent[40] = l32; ent[41] = l33;
    ent[42] = __fdiv_rn(1.0f, l00); ent[43] = __fdiv_rn(1.0f, l11); ent[44] = __fdiv_rn(1.0f, l22); ent[45] = __fdiv_rn(1.0f, l33);
    ent[46] = 1.0f;
}

static __device__ __forceinline__ void solve4(const float* L, const float* D, const float* b, float* t) {
    const float y0 = b[0] * D[0];
    const float y1 = (b[1] - L[1] * y0) * D[1];
    const float y2 = ((b[2] - L[3] * y0) - L[4] * y1) * D[2];
    const float y3 = (((b[3] - L[6] * y0) - L[7] * y1) - L[8] * y2) * D[3];
    t[3] = y3 * D[3];
    t[2] = (y2 - L[8] * t[3]) * D[2];
    t[1] = ((y1 - L[4] * t[2]) - L[7] * t[3]) * D[1];
    t[0] = (((y0 - L[1] * t[1]) - L[3] * t[2]) - L[6] * t[3]) * D[0];
}

static __device__ __forceinline__ float cand_err(const float* L, const float* q, const float* b, float wn, float sc) {
    const float qb = ((q[0] * b[0] + q[1] * b[1]) + q[2] * b[2]) + q[3] * b[3];
    const float z0 = ((L[0] * q[0] + L[1] * q[1]) + L[3] * q[2]) + L[6] * q[3];
    const float z1 = (L[2] * q[1] + L[4] * q[2]) + L[7] * q[3];
    const float z2 = L[5] * q[2] + L[8] * q[3];
    const float z3 = L[9] * q[3];
    const float qgq = ((z0 * z0 + z1 * z1) + z2 * z2) + z3 * z3;
    const float rec = (sc * sc) * qgq;
    return rec > 4.0f * wn ? INF_F : (wn - (2.0f * sc) * qb) + rec;
}

// nslm4_decode_block + the weighted error, for (seed, e, q)
static __device__ float decoded_err4(const float* x, const float* sh, int seed, int e, const int* q) {
    unsigned short st = (unsigned short) seed;
    const float sc = kR32 * pow2f(e);
    float er = 0.0f;
    float wv[8];
    for (int c = 0; c < 8; ++c) {
        int isum = 0;
        for (int p = 0; p < 4; ++p) { st = lfsr_step_d(st); isum += ((int) st - 32768) * q[p]; }
        wv[c] = f2bf_f((float) isum * sc);
    }
    for (int c = 0; c < 8; ++c) {
        const float d = x[c] - sh[c] * wv[c];
        er = er + d * d;
    }
    return er;
}

__global__ void __launch_bounds__(S4_TPB) k_seed_search4(Search4Args a, const float* W, const float* SH, unsigned short* seed_out,
                                                         unsigned short* coef_out, unsigned char* ecode_out, float* err_out) {
    __shared__ float tab[S4_NCH * S4_ENT];
    const int tid = threadIdx.x;
    const int g = a.g0 + (int) blockIdx.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15;
    float sh[8];
    for (int c = 0; c < 8; ++c) sh[c] = SH[g * 8 + c];
    float x[S4_BPT][8], wn[S4_BPT], best[S4_BPT];
    int bs[S4_BPT], be[S4_BPT], row[S4_BPT];
    for (int j = 0; j < S4_BPT; ++j) {
        row[j] = (int) blockIdx.y * S4_ROWS + j * S4_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        for (int c = 0; c < 8; ++c) x[j][c] = W[(size_t) r * a.cols + g * 8 + c] * sh[c];
        float n = 0.0f;
        for (int c = 0; c < 8; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INF_F; bs[j] = 1; be[j] = lo;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += S4_NCH) {
        __syncthreads();
        if (tid < S4_NCH && s0 + tid <= a.n_seeds) seed_entry(s0 + tid, sh, tab + tid * S4_ENT);
        __syncthreads();
        const int nk = min(S4_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            const float* U = tab + k * S4_ENT;
            if (U[46] == 0.0f) continue;
            const float* L = U + 32, *D = U + 42;
            for (int j = 0; j < S4_BPT; ++j) {
                float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int c = 0; c < 8; ++c)
                    for (int p = 0; p < 4; ++p) b[p] = b[p] + U[4 * c + p] * x[j][c];
                float t[4];
                solve4(L, D, b, t);
                const float m = fmaxf(fmaxf(fabsf(t[0]), fabsf(t[1])), fmaxf(fabsf(t[2]), fabsf(t[3])));
                int e0 = flog2f(m) - 2;
                e0 = e0 < lo ? lo : (e0 > hi ? hi : e0);
                for (int ci = 0; ci < a.n_exp; ++ci) {
                    int e = e0 + a.exp_delta[ci];
                    e = e < lo ? lo : (e > hi ? hi : e);
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[4];
                    for (int p = 0; p < 4; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    const float er = cand_err(L, q, b, wn[j], sc);
                    if (er < best[j]) { best[j] = er; bs[j] = s0 + k; be[j] = e; }
                }
            }
        }
    }
    // refit: rebuild the winner's entry in registers
    __syncthreads();
    for (int j = 0; j < S4_BPT; ++j) {
        unsigned short st = (unsigned short) bs[j];
        float U[32];
        for (int k = 0; k < 32; ++k) { st = lfsr_step_d(st); U[k] = ((float) ((int) st - 32768) * kR32) * sh[k / 4]; }
        float G[4][4];
        for (int i = 0; i < 4; ++i)
            for (int jj = 0; jj < 4; ++jj) {
                float gg = 0.0f;
                for (int c = 0; c < 8; ++c) gg = gg + U[4 * c + i] * U[4 * c + jj];
                G[i][jj] = gg;
            }
        const float l00 = __fsqrt_rn(G[0][0]);
        const float l10 = __fdiv_rn(G[1][0], l00), l20 = __fdiv_rn(G[2][0], l00), l30 = __fdiv_rn(G[3][0], l00);
        const float l11 = __fsqrt_rn(G[1][1] - l10 * l10);
        const float l21 = __fdiv_rn(G[2][1] - l20 * l10, l11), l31 = __fdiv_rn(G[3][1] - l30 * l10, l11);
        const float l22 = __fsqrt_rn((G[2][2] - l20 * l20) - l21 * l21);
        const float l32 = __fdiv_rn((G[3][2] - l30 * l20) - l31 * l21, l22);
        const float l33 = __fsqrt_rn(((G[3][3] - l30 * l30) - l31 * l31) - l32 * l32);
        const float L[10] = {l00, l10, l11, l20, l21, l22, l30, l31, l32, l33};
        const float D[4] = {__fdiv_rn(1.0f, l00), __fdiv_rn(1.0f, l11), __fdiv_rn(1.0f, l22), __fdiv_rn(1.0f, l33)};
        float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = 0; c < 8; ++c)
            for (int p = 0; p < 4; ++p) b[p] = b[p] + U[4 * c + p] * x[j][c];
        float t[4];
        solve4(L, D, b, t);
        const int e = be[j];
        const float inv = pow2f(-e);
        int q[4], bq[4];
        for (int p = 0; p < 4; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
        float bde = decoded_err4(x[j], sh, bs[j], e, q);
        if (a.refit)
            for (int d = 0; d < 81; ++d) {
                int c4[4], dd = d;
                bool okq = true;
                for (int p = 0; p < 4; ++p) { c4[p] = q[p] + dd % 3 - 1; dd /= 3; okq = okq && c4[p] >= -8 && c4[p] <= 7; }
                if (!okq) continue;
                const float ee = decoded_err4(x[j], sh, bs[j], e, c4);
                if (ee < bde) { bde = ee; for (int p = 0; p < 4; ++p) bq[p] = c4[p]; }
            }
        if (row[j] >= a.rows) continue;
        const size_t k = (size_t) row[j] * (size_t) ng + (size_t) g;
        seed_out[k] = (unsigned short) bs[j];
        coef_out[k] = (unsigned short) ((bq[0] & 15) | ((bq[1] & 15) << 4) | ((bq[2] & 15) << 8) | ((bq[3] & 15) << 12));
        ecode_out[k] = (unsigned char) (e - a.bias);
        err_out[k] = bde;
    }
}

// ---- the host side (search4_gpu.h) ---------------------------------------------------------------------------------

struct Nslm4Gpu {
    cudaStream_t st;
};

extern "C" {

Nslm4Gpu* nslm4_gpu_open(const char* library, char* err, int errlen) {
    (void) library;   // the kernel is linked in (Metal loads it from a library file)
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) { snprintf(err, (size_t) errlen, "no CUDA device"); return NULL; }
    Nslm4Gpu* g = (Nslm4Gpu*) calloc(1, sizeof(Nslm4Gpu));
    if (cudaStreamCreateWithFlags(&g->st, cudaStreamNonBlocking) != cudaSuccess) { snprintf(err, (size_t) errlen, "CUDA stream"); free(g); return NULL; }
    return g;
}

int nslm4_gpu_search(Nslm4Gpu* g, const float* w, int rows, int cols, const float* sh, int bias, const Search4Opts* o,
                     uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err, char* msg, int msglen) {
    if (cols % 8 || rows < 1 || o->n_exp < 1 || o->n_exp > 3) { snprintf(msg, (size_t) msglen, "bad shape or options"); return -1; }
    const int ng = cols / 8;
    const size_t nb = (size_t) rows * ng;
    float *W = NULL, *SH = NULL, *Ro = NULL;
    unsigned short *So = NULL, *Co = NULL;
    unsigned char* Eo = NULL;
    float* shp = (float*) malloc(sizeof(float) * (size_t) cols);
    for (int c = 0; c < cols; ++c) shp[c] = sh ? sh[c] : 1.0f;
    int rc = -1;
    if (cudaMalloc((void**) &W, sizeof(float) * (size_t) rows * cols) || cudaMalloc((void**) &SH, sizeof(float) * (size_t) cols) ||
        cudaMalloc((void**) &So, 2 * nb) || cudaMalloc((void**) &Co, 2 * nb) || cudaMalloc((void**) &Eo, nb) || cudaMalloc((void**) &Ro, 4 * nb)) {
        snprintf(msg, (size_t) msglen, "out of GPU memory");
        goto done;
    }
    cudaMemcpyAsync(W, w, sizeof(float) * (size_t) rows * cols, cudaMemcpyHostToDevice, g->st);
    cudaMemcpyAsync(SH, shp, sizeof(float) * (size_t) cols, cudaMemcpyHostToDevice, g->st);
    {
        // column groups per launch: about 5e9 block-seeds, so no launch runs long (display watchdogs)
        int gstep = (int) (5e9 / ((double) rows * o->n_seeds));
        if (gstep < 1) gstep = 1;
        if (gstep > ng) gstep = ng;
        for (int g0 = 0; g0 < ng; g0 += gstep) {
            const int gn = ng - g0 < gstep ? ng - g0 : gstep;
            Search4Args a = {rows, cols, g0, bias, o->n_seeds, o->n_exp, o->refit, {o->exp_delta[0], o->exp_delta[1], o->exp_delta[2]}};
            k_seed_search4<<<dim3((unsigned) gn, (unsigned) ((rows + S4_ROWS - 1) / S4_ROWS)), S4_TPB, 0, g->st>>>(a, W, SH, So, Co, Eo, Ro);
        }
    }
    cudaMemcpyAsync(seed, So, 2 * nb, cudaMemcpyDeviceToHost, g->st);
    cudaMemcpyAsync(coef, Co, 2 * nb, cudaMemcpyDeviceToHost, g->st);
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

void nslm4_gpu_close(Nslm4Gpu* g) {
    if (!g) return;
    cudaStreamDestroy(g->st);
    free(g);
}

}   // extern "C"
