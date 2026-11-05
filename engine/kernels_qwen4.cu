// engine/kernels_qwen4.cu - CUDA kernels of qwen4 (engine/kernels_qwen4.h), the counterparts of nslm/qwen4_ops.h.
// f32 values and sums.
#include <math.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <stdint.h>

#include "kernels_qwen4.h"

#define FULL 0xFFFFFFFFu

static __device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
// The sum of v over the block (blockDim.x a multiple of 32), the same value in every thread
static __device__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[w] = v;
    __syncthreads();
    float s = 0;
    for (int k = 0; k < nw; ++k) s += red[k];
    return s;
}
static __device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }
static __device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

// ---- Gated DeltaNet ---------------------------------------------------------------------------------------------------

// Thread (channel, token): cv = silu(causal conv of the last K inputs); inputs before the chunk come from conv.
// One row (gridDim.y = 1): the thread also keeps its channel's last K - 1 inputs, so k_q4_gdn_conv_keep is not needed.
__global__ void k_q4_gdn_conv(int cd, int K, const float* w, const float* qkv, int ld, float* conv, float* cv) {   // qkv rows ld apart
    const int ch = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (ch >= cd) return;
    const float x = qkv[(size_t) t * ld + ch];
    float acc = w[(size_t) ch * K + K - 1] * x;
    for (int k = 0, u = t - (K - 1); k < K - 1; ++k, ++u) acc += w[(size_t) ch * K + k] * (u >= 0 ? qkv[(size_t) u * ld + ch] : conv[(size_t) (K - 1 + u) * cd + ch]);
    cv[(size_t) t * cd + ch] = acc * sigm(acc);
    if (gridDim.y == 1 && K > 1) {
        for (int k = 0; k + 1 < K - 1; ++k) conv[(size_t) k * cd + ch] = conv[(size_t) (k + 1) * cd + ch];
        conv[(size_t) (K - 2) * cd + ch] = x;
    }
}
// conv keeps the last K - 1 inputs; kconv (not NULL) also the state after each row t < T - 1, at kconv[t]
__global__ void k_q4_gdn_conv_keep(int cd, int K, const float* qkv, int ld, float* conv, float* kconv, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= cd) return;
    float old[7];
    for (int k = 0; k < K - 1; ++k) old[k] = conv[(size_t) k * cd + ch];
    for (int t = kconv ? 0 : T - 1; t < T; ++t) {
        float* dst = t == T - 1 ? conv : kconv + (size_t) t * (K - 1) * cd;
        for (int k = 0, u = t + 2 - K; k < K - 1; ++k, ++u) dst[(size_t) k * cd + ch] = u >= 0 ? qkv[(size_t) u * ld + ch] : old[K - 1 + u];
    }
}

// Block (token, key head), DK threads: the head's q and k L2-normalised in place (q also by dk^-0.5), once a token
__global__ void k_q4_gdn_qknorm(int cd, int kd, float* cv) {
    __shared__ float red[32];
    float* qv = cv + (size_t) blockIdx.x * cd + (size_t) blockIdx.y * blockDim.x, *kv = qv + kd;
    const float q = qv[threadIdx.x], k = kv[threadIdx.x];
    const float sq = rsqrtf(block_sum(q * q, red) + 1e-6f) * rsqrtf((float) blockDim.x), sk = rsqrtf(block_sum(k * k, red) + 1e-6f);
    qv[threadIdx.x] = q * sq;
    kv[threadIdx.x] = k * sk;
}

typedef struct {
    int cd, kd, vd, hv, rep, dv, abld;   // abld: the rows of a and b are abld apart
} GdnArgs;

// Block = 32 value rows of head hv; SPLIT threads hold a row's state column S[hv][.][i] as float4 runs of rows
// ((m SPLIT + p) 4, +4). Stages of GDN_TS tokens come by cp.async, the next during this one. y = S q, ungated.
#define GDN_TS 16
static __device__ __forceinline__ float4 mad4(float4 a, float4 b, float4 c) {
    return make_float4(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y), fmaf(a.z, b.z, c.z), fmaf(a.w, b.w, c.w));
}
template <int D4> struct GdnStage {   // a stage's q and k of head kh, v of the block's 32 rows, raw a and b of head hv
    float4 q[GDN_TS][D4], k[GDN_TS][D4];
    float4 v[GDN_TS][8];
    float a[GDN_TS], b[GDN_TS];   // raw, then (each stage, by GDN_TS threads) the decay g and beta
};
template <int D4>
static __device__ __forceinline__ void gdn_fetch(const GdnArgs& a, const float* cv, const float* aa, const float* bb, int hv, int kh, int t0,
                                                 int nt, GdnStage<D4>* st) {
    for (int x = threadIdx.x; x < nt * D4; x += blockDim.x) {
        const float* c = cv + (size_t) (t0 + x / D4) * a.cd + 4 * (x % D4);
        __pipeline_memcpy_async(&st->q[x / D4][x % D4], c + kh * 4 * D4, 16);
        __pipeline_memcpy_async(&st->k[x / D4][x % D4], c + a.kd + kh * 4 * D4, 16);
    }
    for (int x = threadIdx.x; x < nt * 8; x += blockDim.x)
        __pipeline_memcpy_async(&st->v[x / 8][x % 8], cv + (size_t) (t0 + x / 8) * a.cd + 2 * a.kd + (size_t) hv * a.dv + blockIdx.y * 32 + 4 * (x % 8), 16);
    if (threadIdx.x < nt) {
        __pipeline_memcpy_async(&st->a[threadIdx.x], aa + (size_t) (t0 + threadIdx.x) * a.abld + hv, 4);
        __pipeline_memcpy_async(&st->b[threadIdx.x], bb + (size_t) (t0 + threadIdx.x) * a.abld + hv, 4);
    }
    __pipeline_commit();
}
template <int DK, int SPLIT>
__global__ void __launch_bounds__(32 * SPLIT) k_q4_gdn_step(GdnArgs a, const float* cv, const float* aa, const float* bb,
                                                            const float* A_log, const float* dt_bias, float* S, float* kS, float* y, int T) {
    constexpr int DM = DK / SPLIT / 4, D4 = DK / 4;
    const int hv = blockIdx.x, p = threadIdx.x % SPLIT, r = threadIdx.x / SPLIT, i = blockIdx.y * 32 + r, kh = hv / a.rep;
    __shared__ __align__(16) GdnStage<D4> st[2];
    float4 s[DM];
    float* Sh = S + (size_t) hv * DK * a.dv + i;
#pragma unroll
    for (int m = 0; m < DM; ++m) {
        float* c = Sh + (size_t) (m * SPLIT + p) * 4 * a.dv;
        s[m] = make_float4(c[0], c[a.dv], c[2 * a.dv], c[3 * a.dv]);
    }
    const float A = expf(A_log[hv]), dtb = dt_bias[hv];
    if (T > 0) gdn_fetch<D4>(a, cv, aa, bb, hv, kh, 0, min(GDN_TS, T), &st[0]);
    for (int t0 = 0, cur = 0; t0 < T; t0 += GDN_TS, cur ^= 1) {
        const int nt = min(GDN_TS, T - t0);
        __pipeline_wait_prior(0);
        __syncthreads();   // this stage is in, and every thread is done with the other buffer
        if (t0 + GDN_TS < T) gdn_fetch<D4>(a, cv, aa, bb, hv, kh, t0 + GDN_TS, min(GDN_TS, T - t0 - GDN_TS), &st[cur ^ 1]);
        GdnStage<D4>* g4 = &st[cur];
        if (threadIdx.x < nt) {   // the stage's g and beta, off the tokens' chain
            const float ap = g4->a[threadIdx.x] + dtb, sp = ap > 20.0f ? ap : log1pf(expf(ap));
            g4->a[threadIdx.x] = expf(-A * sp);
            g4->b[threadIdx.x] = sigm(g4->b[threadIdx.x]);
        }
        __syncthreads();
        for (int u = 0; u < nt; ++u) {
            const float g = g4->a[u], beta = g4->b[u];
            float4 kv4 = make_float4(0, 0, 0, 0), kw4 = kv4, y4 = kv4, z4 = kv4;   // two chains each
#pragma unroll
            for (int m = 0; m < DM; m += 2) {
                kv4 = mad4(s[m], g4->k[u][m * SPLIT + p], kv4);
                if (m + 1 < DM) kw4 = mad4(s[m + 1], g4->k[u][(m + 1) * SPLIT + p], kw4);
            }
            float kv = ((kv4.x + kw4.x) + (kv4.y + kw4.y)) + ((kv4.z + kw4.z) + (kv4.w + kw4.w));
            if (SPLIT == 2) kv += __shfl_xor_sync(FULL, kv, 1);
            const float delta = (((const float*) g4->v[u])[r] - g * kv) * beta;
#pragma unroll
            for (int m = 0; m < DM; ++m) {
                const float4 kk = g4->k[u][m * SPLIT + p];
                s[m] = make_float4(g * s[m].x + delta * kk.x, g * s[m].y + delta * kk.y, g * s[m].z + delta * kk.z, g * s[m].w + delta * kk.w);
                if (m & 1) z4 = mad4(s[m], g4->q[u][m * SPLIT + p], z4);
                else y4 = mad4(s[m], g4->q[u][m * SPLIT + p], y4);
            }
            float yq = ((y4.x + z4.x) + (y4.y + z4.y)) + ((y4.z + z4.z) + (y4.w + z4.w));
            if (SPLIT == 2) yq += __shfl_xor_sync(FULL, yq, 1);
            if (p == 0) y[(size_t) (t0 + u) * a.vd + (size_t) hv * a.dv + i] = yq;
            if (kS && t0 + u + 1 < T)   // the state after this row, at kS[t0 + u]
#pragma unroll
                for (int m = 0; m < DM; ++m) {
                    float* c = kS + (size_t) (t0 + u) * a.hv * DK * a.dv + (Sh - S) + (size_t) (m * SPLIT + p) * 4 * a.dv;
                    c[0] = s[m].x; c[a.dv] = s[m].y; c[2 * a.dv] = s[m].z; c[3 * a.dv] = s[m].w;
                }
        }
    }
#pragma unroll
    for (int m = 0; m < DM; ++m) {
        float* c = Sh + (size_t) (m * SPLIT + p) * 4 * a.dv;
        c[0] = s[m].x; c[a.dv] = s[m].y; c[2 * a.dv] = s[m].z; c[3 * a.dv] = s[m].w;
    }
}

// Block (head, token), thread = value row: RMSNorm over the head (plain weight), gated by sigmoid(z)
__global__ void k_q4_gdn_norm(int hv, int dv, float eps, const float* norm_w, const float* z, int zld, float* y) {   // z rows zld apart
    __shared__ float red[32];
    const size_t o = ((size_t) blockIdx.y * hv + blockIdx.x) * dv + threadIdx.x;
    const float v = y[o], r = rsqrtf(block_sum(v * v, red) / dv + eps);
    y[o] = v * r * norm_w[threadIdx.x] * sigm(z[(size_t) blockIdx.y * zld + (size_t) blockIdx.x * dv + threadIdx.x]);
}

static int gdn_run(cudaStream_t s, const Qwen4Cfg* c, Q4GdnDev w, const float* qkv, int qld, const float* z, int zld, const float* a,
                   const float* b, int abld, float* conv, float* S, float* kconv, float* kS, float* cv, float* y, int T) {
    const int kd = c->lin_hk * c->lin_dk, vd = c->lin_hv * c->lin_dv, cd = 2 * kd + vd;
    if (c->lin_conv < 1 || c->lin_conv > 8 || c->lin_dv % 32 || c->lin_dv > 1024) return -1;
    k_q4_gdn_conv<<<dim3((unsigned) (cd + 255) / 256, (unsigned) T), 256, 0, s>>>(cd, c->lin_conv, w.conv_w, qkv, qld, conv, cv);
    if (T > 1) k_q4_gdn_conv_keep<<<(unsigned) (cd + 255) / 256, 256, 0, s>>>(cd, c->lin_conv, qkv, qld, conv, kconv, T);
    k_q4_gdn_qknorm<<<dim3((unsigned) T, (unsigned) c->lin_hk), (unsigned) c->lin_dk, 0, s>>>(cd, kd, cv);
    const GdnArgs ga = {cd, kd, vd, c->lin_hv, c->lin_hv / c->lin_hk, c->lin_dv, abld};
    const dim3 grid((unsigned) c->lin_hv, (unsigned) c->lin_dv / 32);
    switch (c->lin_dk) {
    case 32: k_q4_gdn_step<32, 1><<<grid, 32, 0, s>>>(ga, cv, a, b, w.A_log, w.dt_bias, S, kS, y, T); break;
    case 64: k_q4_gdn_step<64, 1><<<grid, 32, 0, s>>>(ga, cv, a, b, w.A_log, w.dt_bias, S, kS, y, T); break;
    case 128: k_q4_gdn_step<128, 2><<<grid, 64, 0, s>>>(ga, cv, a, b, w.A_log, w.dt_bias, S, kS, y, T); break;
    default: return -1;
    }
    k_q4_gdn_norm<<<dim3((unsigned) c->lin_hv, (unsigned) T), (unsigned) c->lin_dv, 0, s>>>(c->lin_hv, c->lin_dv, c->eps, w.norm_w, z, zld, y);
    return 0;
}
int kq4_gdn(cudaStream_t s, const Qwen4Cfg* c, Q4GdnDev w, const float* qkv, const float* z, const float* a, const float* b,
            float* conv, float* S, float* cv, float* y, int T) {
    const int vd = c->lin_hv * c->lin_dv;
    return gdn_run(s, c, w, qkv, 2 * c->lin_hk * c->lin_dk + vd, z, vd, a, b, c->lin_hv, conv, S, NULL, NULL, cv, y, T);
}
int kq4_gdn_ld(cudaStream_t s, const Qwen4Cfg* c, Q4GdnDev w, const float* qkv, const float* z, const float* a, const float* b, int ld,
               float* conv, float* S, float* kconv, float* kS, float* cv, float* y, int T) {
    return gdn_run(s, c, w, qkv, ld, z, ld, a, b, ld, conv, S, kconv, kS, cv, y, T);
}

// ---- RMSNorm (1 + w) and the hyper-connection pointwise steps --------------------------------------------------------

// A thread's values of a group in registers (groups of up to Q4_ROW_MAX): the loads go out together, not in a chain
#define NR (Q4_ROW_MAX / 256)
__global__ void __launch_bounds__(256) k_q4_norm1p(const float* x, int len, int g, const float* w, float eps, float* y) {
    __shared__ float red[32];
    const int groups = len / g, t = blockIdx.x / groups, a = (blockIdx.x % groups) * g;
    const float* xi = x + (size_t) t * len + a;
    float v[NR], wv[NR], ss = 0;
#pragma unroll
    for (int j = 0; j < NR; ++j) {
        const int k = threadIdx.x + j * 256;
        v[j] = k < g ? xi[k] : 0.0f;
        wv[j] = k < g ? w[a + k] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < NR; ++j) ss += v[j] * v[j];
    const float r = rsqrtf(block_sum(ss, red) / g + eps);
#pragma unroll
    for (int j = 0; j < NR; ++j)
        if (threadIdx.x + j * 256 < g) y[(size_t) t * len + a + threadIdx.x + j * 256] = v[j] * r * (1.0f + wv[j]);
}
// Groups longer than Q4_ROW_MAX (the MTP head's norm of all streams): the values read twice
__global__ void __launch_bounds__(256) k_q4_norm1p_long(const float* x, int len, int g, const float* w, float eps, float* y) {
    __shared__ float red[32];
    const int groups = len / g, t = blockIdx.x / groups, a = (blockIdx.x % groups) * g;
    const float* xi = x + (size_t) t * len + a;
    float ss = 0;
    for (int k = threadIdx.x; k < g; k += 256) ss += xi[k] * xi[k];
    const float r = rsqrtf(block_sum(ss, red) / g + eps);
    for (int k = threadIdx.x; k < g; k += 256) y[(size_t) t * len + a + k] = xi[k] * r * (1.0f + w[a + k]);
}
void kq4_norm1p(cudaStream_t s, const float* x, int len, int g, const float* w, float eps, float* y, int T) {
    if (g > Q4_ROW_MAX) k_q4_norm1p_long<<<(unsigned) (T * (len / g)), 256, 0, s>>>(x, len, g, w, eps, y);
    else k_q4_norm1p<<<(unsigned) (T * (len / g)), 256, 0, s>>>(x, len, g, w, eps, y);
}

// Row t of lo [rows][ld]: the act on its first rank values (in place), the gate of the rest into inj [rows][ld - rank]
__global__ void k_q4_hc_lo(float inv_hc, float* lo, int rank, int ld, float* inj, int n) {   // n = rows * ld
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / ld, j = i % ld;
    const float x = lo[i] * inv_hc;
    if (j < rank) lo[i] = x * sigm(x);
    else inj[(size_t) t * (ld - rank) + j - rank] = 2.0f * sigm(x);
}
__global__ void k_q4_hc_mix(int hc, int d, const float* normed, const float* up, float* mixed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= d) return;
    const size_t o = (size_t) t * hc * d + i;
    float m = 0;
    for (int s = 0; s < hc; ++s) m += sigm(up[o + (size_t) s * d]) * normed[o + (size_t) s * d];
    mixed[(size_t) t * d + i] = m / hc;
}
__global__ void k_q4_hc_add(int hc, int d, float* H, const float* branch, const float* inj) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= d) return;
    const float b = branch[(size_t) t * d + i];
    for (int s = 0; s < hc; ++s) H[((size_t) t * hc + s) * d + i] += b * inj[t * hc + s];
}
void kq4_hc_lo(cudaStream_t s, int hc, float* lo, int rows, int rank, int ld, float* inj) {
    k_q4_hc_lo<<<(unsigned) (rows * ld + 255) / 256, 256, 0, s>>>(1.0f / hc, lo, rank, ld, inj, rows * ld);
}
void kq4_hc_act(cudaStream_t s, int hc, float* lo, int n) { kq4_hc_lo(s, hc, lo, 1, n, n, NULL); }
void kq4_hc_gate(cudaStream_t s, int hc, float* inj, int n) { kq4_hc_lo(s, hc, inj, n / hc, 0, hc, inj); }
void kq4_hc_mix(cudaStream_t s, int hc, int d, const float* normed, const float* up, float* mixed, int T) {
    k_q4_hc_mix<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(hc, d, normed, up, mixed);
}
// k_q4_hc_add, then k_q4_norm1p of each stream into normed, in one pass: block (row, stream), the same threads and order
__global__ void __launch_bounds__(256) k_q4_hc_add_norm(int hc, int d, float* H, const float* br, const float* inj, const float* w,
                                                        float eps, float* normed) {
    __shared__ float red[32];
    const int t = blockIdx.x / hc, s = blockIdx.x % hc;
    float* h = H + ((size_t) t * hc + s) * d;
    const float* b = br + (size_t) t * d;
    const float g = inj[t * hc + s];
    float v[NR], bv[NR], wv[NR], ss = 0;
#pragma unroll
    for (int j = 0; j < NR; ++j) {
        const int i = threadIdx.x + j * 256;
        v[j] = i < d ? h[i] : 0.0f;
        bv[j] = i < d ? b[i] : 0.0f;
        wv[j] = i < d ? w[s * d + i] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < NR; ++j) {
        v[j] += bv[j] * g;
        ss += v[j] * v[j];
    }
    const float r = rsqrtf(block_sum(ss, red) / d + eps);
#pragma unroll
    for (int j = 0; j < NR; ++j) {
        const int i = threadIdx.x + j * 256;
        if (i < d) { h[i] = v[j]; normed[((size_t) t * hc + s) * d + i] = v[j] * r * (1.0f + wv[j]); }
    }
}
void kq4_hc_add_norm(cudaStream_t s, int hc, int d, float* H, const float* branch, const float* inj, const float* w, float eps, float* normed, int T) {
    k_q4_hc_add_norm<<<(unsigned) (T * hc), 256, 0, s>>>(hc, d, H, branch, inj, w, eps, normed);
}
void kq4_hc_add(cudaStream_t s, int hc, int d, float* H, const float* branch, const float* inj, int T) {
    k_q4_hc_add<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(hc, d, H, branch, inj);
}

// ---- embedding streams, head norm + RoPE, MoE --------------------------------------------------------------------------

__global__ void k_q4_tile(int hc, int d, const float* x, float* H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= d) return;
    for (int s = 0; s < hc; ++s) H[((size_t) t * hc + s) * d + i] = x[(size_t) t * d + i];
}
void kq4_tile(cudaStream_t s, int hc, int d, const float* x, float* H, int T) {
    k_q4_tile<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(hc, d, x, H);
}

// One block: RMSNorm (1 + w) of xh[hd], then RoPE of its first rot values at pos (f32 angles from a double inverse
// frequency, cos and sin in double: as the C ops).  red[32] and v[hd] are the block's shared scratch.
static __device__ void norm_rope(float* xh, int hd, const float* w, float eps, int rot, const float* inv, int pos, float* red, float* v) {
    float ss = 0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += xh[i] * xh[i];
    const float r = rsqrtf(block_sum(ss, red) / hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) v[i] = xh[i] * r * (1.0f + w[i]);
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = v[i];
        if (i < rot) {
            const float a = (float) pos * inv[i % (rot / 2)];
            double sn, cs;
            sincos((double) a, &sn, &cs);
            o = i < rot / 2 ? (float) (v[i] * cs - v[i + rot / 2] * sn) : (float) (v[i] * cs + v[i - rot / 2] * sn);
        }
        xh[i] = o;
    }
}
// Block (head, row) at position pos0 (+ *dpos) + row
__global__ void __launch_bounds__(128) k_q4_head_norm_rope(float* x, int rs, int hs, int hd, const float* w, float eps, int rot,
                                                           const float* inv, int pos0, const int32_t* dpos) {
    __shared__ float red[32], v[1024];
    norm_rope(x + (size_t) blockIdx.y * rs + (size_t) blockIdx.x * hs, hd, w, eps, rot, inv, pos0 + (dpos ? *dpos : 0) + (int) blockIdx.y, red, v);
}
void kq4_rope_inv(const Qwen4Cfg* c, float* inv) {
    for (int f = 0; f < c->rot_dim / 2; ++f) inv[f] = (float) pow((double) c->rope_theta, -2.0 * f / c->rot_dim);
}
static float* rope_tmp(const Qwen4Cfg* c) {   // the wrappers' table (the kernel tests); freed by cudaFree, which syncs
    float h[512], *d = NULL;
    kq4_rope_inv(c, h);
    cudaMalloc(&d, sizeof h);
    cudaMemcpy(d, h, sizeof(float) * (size_t) (c->rot_dim / 2), cudaMemcpyHostToDevice);
    return d;
}
void kq4_head_norm_rope_inv(cudaStream_t s, const Qwen4Cfg* c, float* x, int rs, int nh, int hs, int hd, const float* w,
                            const float* inv, int pos0, const int32_t* dpos, int T) {
    k_q4_head_norm_rope<<<dim3((unsigned) nh, (unsigned) T), 128, 0, s>>>(x, rs, hs, hd, w, c->eps, c->rot_dim, inv, pos0, dpos);
}
void kq4_head_norm_rope(cudaStream_t s, const Qwen4Cfg* c, float* x, int rs, int nh, int hs, int hd, const float* w, int pos0,
                        const int32_t* dpos, int T) {
    float* inv = rope_tmp(c);
    kq4_head_norm_rope_inv(s, c, x, rs, nh, hs, hd, w, inv, pos0, dpos, T);
    cudaFree(inv);
}
// dst[(pos + t) * w + i] = src[t * ss + i], pos = pos0 (+ *dpos)
__global__ void k_q4_store(float* dst, int w, const float* src, int ss, int pos0, const int32_t* dpos) {
    const int t = (int) blockIdx.y, i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < w) dst[(size_t) (pos0 + (dpos ? *dpos : 0) + t) * w + i] = src[(size_t) t * ss + i];
}
void kq4_store(cudaStream_t s, float* dst, int w, const float* src, int ss, int pos0, const int32_t* dpos, int T) {
    k_q4_store<<<dim3((unsigned) (w + 255) / 256, (unsigned) T), 256, 0, s>>>(dst, w, src, ss, pos0, dpos);
}

// Row t: one warp, a lane's logits in registers (E at most 32 RT_PER). k rounds of a warp arg-max (ties: the higher
// index); the weights are the softmax over all E, renormalised over the k picked.
#define RT_PER 32
__global__ void k_q4_route(int E, int k, const float* logits, int ld, int32_t* ind, float* wt) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* l = logits + (size_t) t * ld;
    float v[RT_PER], mx = -INFINITY, top = 0, w[32];
#pragma unroll
    for (int j = 0; j < RT_PER; ++j) {
        v[j] = lane + 32 * j < E ? l[lane + 32 * j] : -INFINITY;
        mx = fmaxf(mx, v[j]);
    }
    mx = warp_max(mx);
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = -1;
#pragma unroll
        for (int j = 0; j < RT_PER; ++j)
            if (lane + 32 * j < E && (v[j] > bv || (v[j] == bv && lane + 32 * j > bi))) { bv = v[j]; bi = lane + 32 * j; }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(FULL, bv, o);
            const int oi = __shfl_xor_sync(FULL, bi, o);
            if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
        }
#pragma unroll
        for (int j = 0; j < RT_PER; ++j) if (lane + 32 * j == bi) v[j] = -INFINITY;   // taken
        if (lane == 0) ind[(size_t) t * k + r] = bi;
        w[r] = expf(bv - mx);
        top += w[r];
    }
    if (lane == 0) for (int r = 0; r < k; ++r) wt[(size_t) t * k + r] = w[r] / top;
}
void kq4_route_ld(cudaStream_t s, const Qwen4Cfg* c, const float* logits, int ld, int32_t* ind, float* wt, int T) {
    k_q4_route<<<(unsigned) T, 32, 0, s>>>(c->n_exp, c->top_k, logits, ld, ind, wt);
}
void kq4_route(cudaStream_t s, const Qwen4Cfg* c, const float* logits, int32_t* ind, float* wt, int T) {
    kq4_route_ld(s, c, logits, c->n_exp, ind, wt, T);
}

__global__ void __launch_bounds__(256) k_q4_tail_skip(int32_t* ind, float* wt, const int32_t* slot, int P, int k, int R, int renorm) {
    __shared__ int skip[256];
    const int p = threadIdx.x;
    int sk = p < P && p % k >= R && slot[ind[p]] < 0;
    for (int q = 0; sk && q < P; ++q) sk = !(q % k < R && ind[q] == ind[p]);   // a higher rank brings it anyway
    skip[p] = sk;
    __syncthreads();
    if (sk) ind[p] = ind[p - p % k];
    if (!renorm) { if (sk) wt[p] = 0.0f; return; }
    if (p >= P / k) return;   // renorm: thread p scales row p's kept weights to the row's sum
    float all = 0, kept = 0;
    for (int j = 0; j < k; ++j) { all += wt[p * k + j]; kept += skip[p * k + j] ? 0 : wt[p * k + j]; }
    for (int j = 0; j < k; ++j) wt[p * k + j] = skip[p * k + j] ? 0.0f : wt[p * k + j] * (all / kept);
}
void kq4_tail_skip(cudaStream_t s, int32_t* ind, float* wt, const int32_t* slot, int T, int k, int R, int renorm) {
    if (R > 0 && R < k && T * k <= 256) k_q4_tail_skip<<<1, 256, 0, s>>>(ind, wt, slot, T * k, k, R, renorm);
}

__global__ void k_q4_signal(uint32_t* seq, volatile uint32_t* flag) {
    const uint32_t v = ++*seq;
    __threadfence_system();
    *flag = v;
}
__global__ void k_q4_wait(const uint32_t* seq, volatile const uint32_t* ack) {
    const uint32_t v = *seq;
    while (*ack < v) __nanosleep(100);
}
void kq4_signal(cudaStream_t s, uint32_t* seq, volatile uint32_t* flag) { k_q4_signal<<<1, 1, 0, s>>>(seq, flag); }
void kq4_wait(cudaStream_t s, const uint32_t* seq, volatile const uint32_t* ack) { k_q4_wait<<<1, 1, 0, s>>>(seq, ack); }

__global__ void k_q4_moe_combine(int d, int k, const float* D, const float* wt, const float* sh, const float* sg, int sgld, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= d) return;
    float a = sigm(sg[(size_t) t * sgld]) * sh[(size_t) t * d + i];
    for (int j = 0; j < k; ++j) a += wt[t * k + j] * D[((size_t) t * k + j) * d + i];
    out[(size_t) t * d + i] = a;
}
void kq4_moe_combine(cudaStream_t s, int d, int k, const float* D, const float* wt, const float* sh, const float* sg, int sgld, float* out, int T) {
    k_q4_moe_combine<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(d, k, D, wt, sh, sg, sgld, out);
}

// ---- PLE ---------------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(256) k_q4_ple_gate(int hc, int d, const float* keys, const float* qs, const float* vals, float* gated) {
    __shared__ float red[32];
    const int sidx = blockIdx.x, t = blockIdx.y;
    const size_t o = ((size_t) t * hc + sidx) * d;
    float dot = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) dot += keys[o + i] * qs[o + i];
    const float g = block_sum(dot, red) / sqrtf((float) d);
    const float gs = sigm(copysignf(sqrtf(fmaxf(fabsf(g), 1e-6f)), g) * (g != 0.0f));   // a signed square root
    for (int i = threadIdx.x; i < d; i += blockDim.x) gated[o + i] = gs * vals[(size_t) t * d + i];
}
void kq4_ple_gate(cudaStream_t s, const Qwen4Cfg* c, const float* keys, const float* qs, const float* vals, float* gated, int T) {
    k_q4_ple_gate<<<dim3((unsigned) c->hc, (unsigned) T), 256, 0, s>>>(c->hc, c->d, keys, qs, vals, gated);
}

// One thread per channel, tokens in order; the last (K - 1) * dil inputs in registers (at most 32)
__global__ void k_q4_ple_conv(int HD, int K, int dil, const float* w, const float* cin, const float* gated, float* hist, float* khist, float* H, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= HD) return;
    const int L = (K - 1) * dil;
    float st[32];
    for (int j = 0; j < L; ++j) st[j] = hist[(size_t) j * HD + ch];
    for (int t = 0; t < T; ++t) {
        const float x = cin[(size_t) t * HD + ch];
        float acc = w[(size_t) ch * K + K - 1] * x;
        for (int k = 0; k < K - 1; ++k) acc += w[(size_t) ch * K + k] * st[L - (K - 1 - k) * dil];
        H[(size_t) t * HD + ch] += gated[(size_t) t * HD + ch] + acc * sigm(acc);
        for (int j = 0; j + 1 < L; ++j) st[j] = st[j + 1];
        if (L > 0) st[L - 1] = x;
        for (int j = 0; khist && t + 1 < T && j < L; ++j) khist[((size_t) t * L + j) * HD + ch] = st[j];
    }
    for (int j = 0; j < L; ++j) hist[(size_t) j * HD + ch] = st[j];
}
void kq4_ple_conv_keep(cudaStream_t s, const Qwen4Cfg* c, const float* w, const float* cin, const float* gated, float* hist, float* khist,
                       float* H, int T) {
    const int HD = c->hc * c->d;
    k_q4_ple_conv<<<(unsigned) (HD + 255) / 256, 256, 0, s>>>(HD, c->ple_conv, c->ngram, w, cin, gated, hist, khist, H, T);
}
void kq4_ple_conv(cudaStream_t s, const Qwen4Cfg* c, const float* w, const float* cin, const float* gated, float* hist, float* H, int T) {
    kq4_ple_conv_keep(s, c, w, cin, gated, hist, NULL, H, T);
}

// ---- QSA ---------------------------------------------------------------------------------------------------------------

// Block b: key block j = pos / R + b if rows [pos, pos + T) complete it: the mean of its R raw keys, then norm + RoPE at
// position j * R
__global__ void __launch_bounds__(128) k_q4_qsa_blocks(int D, int R, const float* raw, const float* w, float eps, int rot, const float* inv,
                                                       int pos0, const int32_t* dpos, int T, float* blk) {
    __shared__ float red[32], v[1024];
    const int pos = pos0 + (dpos ? *dpos : 0), j = pos / R + (int) blockIdx.x;
    if (j >= (pos + T) / R) return;
    float* bj = blk + (size_t) j * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float m = 0;
        for (int u = 0; u < R; ++u) m += raw[((size_t) j * R + u) * D + i];
        bj[i] = m / R;
    }
    __syncthreads();
    norm_rope(bj, D, w, eps, rot, inv, j * R, red, v);
}
void kq4_qsa_blocks_inv(cudaStream_t s, const Qwen4Cfg* c, const float* raw, const float* w, const float* inv, int pos0,
                        const int32_t* dpos, int T, float* blk) {
    k_q4_qsa_blocks<<<(unsigned) (T / c->ix_ratio + 1), 128, 0, s>>>(c->ix_dim, c->ix_ratio, raw, w, c->eps, c->rot_dim, inv, pos0, dpos, T, blk);
}
void kq4_qsa_blocks(cudaStream_t s, const Qwen4Cfg* c, const float* raw, const float* w, int pos0, const int32_t* dpos, int T, float* blk) {
    float* inv = rope_tmp(c);
    kq4_qsa_blocks_inv(s, c, raw, w, inv, pos0, dpos, T, blk);
    cudaFree(inv);
}
__global__ void k_q4_qsa_rows(int R, int k, int pos0, const int32_t* dpos, int32_t* done, int32_t* nsel, int T) {
    const int t = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= T) return;
    const int d = (pos0 + (dpos ? *dpos : 0) + t + 1) / R;
    done[t] = d;
    nsel[t] = d > k ? k : 0;
}
void kq4_qsa_rows(cudaStream_t s, const Qwen4Cfg* c, int pos0, const int32_t* dpos, int32_t* done, int32_t* nsel, int T) {
    k_q4_qsa_rows<<<(unsigned) (T + 255) / 256, 256, 0, s>>>(c->ix_ratio, c->ix_budget / c->ix_ratio, pos0, dpos, done, nsel, T);
}

// Decode rows: a warp per (row, block), its lanes over the dims
__global__ void __launch_bounds__(256) k_q4_qsa_scores1(int nh, int D, const float* iq, int rs, const float* blk, const int32_t* done,
                                                        int maxdone, float* sc) {
    const int t = blockIdx.y, j = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (j >= done[t]) return;
    float s = 0;
    for (int h = 0; h < nh; ++h) {
        float d = 0;
        for (int i = lane; i < D; i += 32) d += iq[(size_t) t * rs + h * D + i] * blk[(size_t) j * D + i];
        s += fmaxf(warp_sum(d), 0.0f);
    }
    if (lane == 0) sc[(size_t) t * maxdone + j] = s;
}
// Prompt rows: a block of QS_R rows x 32 key blocks; thread (row r, block jj): the sum over heads of relu(q_h . blk_j). The rows' q and
// the blocks' keys are in shared memory, the keys' rows one float apart in banks (no conflicts).
#define QS_R 8
__global__ void __launch_bounds__(256) k_q4_qsa_scores(int nh, int D, const float* iq, int rs, const float* blk, const int32_t* done,
                                                       int maxdone, float* sc, int T) {
    extern __shared__ float qss[];   // q [QS_R][nh D], keys [32][D + 1]
    float* qs = qss, *ks = qss + QS_R * nh * D;
    const int t0 = blockIdx.y * QS_R, j0 = blockIdx.x * 32, r = threadIdx.x >> 5, jj = threadIdx.x & 31;
    for (int x = threadIdx.x; x < QS_R * nh * D; x += blockDim.x)
        qs[x] = t0 + x / (nh * D) < T ? iq[(size_t) (t0 + x / (nh * D)) * rs + x % (nh * D)] : 0.0f;
    for (int x = threadIdx.x; x < 32 * D; x += blockDim.x) ks[x / D * (D + 1) + x % D] = j0 + x / D < maxdone ? blk[(size_t) (j0 + x / D) * D + x % D] : 0.0f;
    __syncthreads();
    const int t = t0 + r, j = j0 + jj;
    if (t >= T || j >= done[t]) return;
    float s = 0;
    for (int h = 0; h < nh; ++h) {
        const float* q = qs + (r * nh + h) * D, *k = ks + jj * (D + 1);
        float d = 0;
        for (int i = 0; i < D; ++i) d += q[i] * k[i];
        s += fmaxf(d, 0.0f);
    }
    sc[(size_t) t * maxdone + j] = s;
}
// Block per row: the k-th largest score by a radix select on its bits (scores are >= 0), then every score above it in
// block order (attention sums the keys in this order, so it must not depend on timing) and, of the equal ones, the highest
__global__ void __launch_bounds__(256) k_q4_qsa_topk(const float* sc, const int32_t* done, int maxdone, int k, int32_t* sel) {
    __shared__ int hist[256], nout, need, wcnt[8];
    __shared__ uint32_t pre;
    const int t = blockIdx.x, n = done[t], lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (n <= k) return;
    const uint32_t* u = (const uint32_t*) (sc + (size_t) t * maxdone);   // scores >= 0: their bits sort as the floats
    uint32_t mask = 0;
    if (threadIdx.x == 0) { pre = 0; need = k; }
    for (int sh = 24; sh >= 0; sh -= 8) {   // the k-th largest score, 8 bits a pass: count each digit under the prefix
        hist[threadIdx.x] = 0;
        __syncthreads();
        const uint32_t p = pre;
        for (int j = threadIdx.x; j < n; j += blockDim.x) if ((u[j] & mask) == p) atomicAdd(&hist[(u[j] >> sh) & 255], 1);
        __syncthreads();
        if (threadIdx.x == 0) {
            int cum = 0, d = 255;
            while (cum + hist[d] < need) cum += hist[d--];
            pre = p | (uint32_t) d << sh;
            need -= cum;
        }
        mask |= 255u << sh;
        __syncthreads();
    }
    const uint32_t p = pre;
    int32_t* out = sel + (size_t) t * k;
    if (threadIdx.x == 0) nout = 0;
    __syncthreads();
    // The scores above p in index order, then those equal to p from the highest index: a block-wide prefix of the matches
    for (int pass = 0; pass < 2; ++pass)
        for (int c0 = 0; c0 < n && nout < k; c0 += 256) {
            const int j = pass ? n - 1 - c0 - threadIdx.x : c0 + threadIdx.x;
            const int f = j >= 0 && j < n && (pass ? u[j] == p : u[j] > p);
            const uint32_t m = __ballot_sync(FULL, f);
            if (lane == 0) wcnt[w] = __popc(m);
            __syncthreads();
            int before = nout;
            for (int x = 0; x < w; ++x) before += wcnt[x];
            before += __popc(m & ((1u << lane) - 1u));
            if (f && before < k) out[before] = j;
            __syncthreads();
            if (threadIdx.x == 0) for (int x = 0; x < 8; ++x) nout += wcnt[x];
            __syncthreads();
        }
}
void kq4_qsa_select(cudaStream_t s, const Qwen4Cfg* c, const float* iq, int rs, const float* blk, const int32_t* done, int maxdone,
                    float* sc, int32_t* sel, int T) {
    if (maxdone <= 0) return;
    const size_t sm = sizeof(float) * ((size_t) QS_R * c->ix_heads * c->ix_dim + 32 * ((size_t) c->ix_dim + 1));   // 32 KB here
    if (T <= 8) k_q4_qsa_scores1<<<dim3((unsigned) (maxdone + 7) / 8, (unsigned) T), 256, 0, s>>>(c->ix_heads, c->ix_dim, iq, rs, blk, done, maxdone, sc);
    else k_q4_qsa_scores<<<dim3((unsigned) (maxdone + 31) / 32, (unsigned) (T + QS_R - 1) / QS_R), 256, sm, s>>>(c->ix_heads, c->ix_dim, iq, rs, blk,
                                                                                                         done, maxdone, sc, T);
    k_q4_qsa_topk<<<(unsigned) T, 256, 0, s>>>(sc, done, maxdone, c->ix_budget / c->ix_ratio, sel);
}

// ---- attention ---------------------------------------------------------------------------------------------------------

typedef struct {
    int nh, nkv, hd, R, topk, pos0, qld;   // qld: the rows of q are qld apart
    float scale;
} AttArgs;

// Prompt rows: block (KV group, row), a warp per query head of the group. The row's keys come in tiles of AT_KT; each
// key's K and V is read once into shared memory for all heads of the group. Each head takes one online-softmax step a tile
#define AT_KT 16
__global__ void k_q4_attend_g(AttArgs a, const float* q, const float* K, const float* V, const int32_t* sel, const int32_t* nsel,
                              const int32_t* dpos, float* o) {
    extern __shared__ float kvs[];   // K [AT_KT][hd], then V [AT_KT][hd]
    __shared__ int keys[AT_KT];
    const int g = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, dl = a.hd / 32, h = g * (a.nh / a.nkv) + w;
    const int pos = a.pos0 + (dpos ? *dpos : 0) + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    const float* qh = q + (size_t) t * a.qld + (size_t) h * 2 * a.hd;
    float* Vs = kvs + AT_KT * a.hd;
    float qv[8], acc[8], m = -INFINITY, l = 0;
    for (int i = 0; i < dl; ++i) { qv[i] = qh[lane + 32 * i]; acc[i] = 0; }
    for (int k0 = 0; k0 < n; k0 += AT_KT) {
        const int nk = min(AT_KT, n - k0);
        __syncthreads();   // the previous tile is read
        if (threadIdx.x < nk) {
            const int k = k0 + threadIdx.x;
            keys[threadIdx.x] = k < ns * a.R ? sel[(size_t) t * a.topk + k / a.R] * a.R + k % a.R : tail + (k - ns * a.R);
        }
        __syncthreads();
        const int h4 = a.hd / 4;   // 16-byte loads: fewer load rounds a tile
#pragma unroll 4
        for (int x = threadIdx.x; x < nk * h4; x += blockDim.x) {
            const size_t at = (((size_t) keys[x / h4] * a.nkv + g) * a.hd) / 4 + x % h4;
            ((float4*) kvs)[x] = ((const float4*) K)[at];
            ((float4*) Vs)[x] = ((const float4*) V)[at];
        }
        __syncthreads();
        float d[AT_KT], mt = m, ps = 0;   // the tile's scores first (independent), then one softmax step for the tile
#pragma unroll
        for (int kk = 0; kk < AT_KT; ++kk) {
            float x = 0;
            for (int i = 0; kk < nk && i < dl; ++i) x += qv[i] * kvs[kk * a.hd + lane + 32 * i];
            d[kk] = x;
        }
#pragma unroll
        for (int kk = 0; kk < AT_KT; ++kk) { d[kk] = kk < nk ? warp_sum(d[kk]) * a.scale : -INFINITY; mt = fmaxf(mt, d[kk]); }
        const float cf = expf(m - mt);
        for (int i = 0; i < dl; ++i) acc[i] *= cf;
#pragma unroll
        for (int kk = 0; kk < AT_KT; ++kk) {
            if (kk >= nk) break;
            const float p = expf(d[kk] - mt);
            ps += p;
            for (int i = 0; i < dl; ++i) acc[i] += p * Vs[kk * a.hd + lane + 32 * i];
        }
        l = l * cf + ps;
        m = mt;
    }
    for (int i = 0; i < dl; ++i) o[((size_t) t * a.nh + h) * a.hd + lane + 32 * i] = acc[i] / l * sigm(qh[a.hd + lane + 32 * i]);
}

// Prompt rows on tensor cores: block (KV group, row), the group's heads as matrix rows. S = Q K^T (wmma), O = P V
// (mma.sync), TF32 in three parts; one pass, an online softmax rescales O by row. K, then V, tiles by cp.async.
#define TC_KT 16
#define TC_PS 20   // P's row stride: the mma.sync A loads fall in different banks
template <typename F> static __device__ __forceinline__ void tf32_split(const F& f, F& hi, F& lo) {
    for (int i = 0; i < f.num_elements; ++i) {
        hi.x[i] = nvcuda::wmma::__float_to_tf32(f.x[i]);
        lo.x[i] = nvcuda::wmma::__float_to_tf32(f.x[i] - hi.x[i]);
    }
}
template <typename A, typename B, typename C> static __device__ __forceinline__ void mma3s(C& c, const A& ah, const A& al, const B& b) {
    B bh, bl;
    tf32_split(b, bh, bl);
    nvcuda::wmma::mma_sync(c, ah, bh, c);
    nvcuda::wmma::mma_sync(c, ah, bl, c);
    nvcuda::wmma::mma_sync(c, al, bh, c);
}
// The tile of TC_KT keys (key < 0: zeros) of K or V into the shared tile, by cp.async; the caller commits and waits
static __device__ __forceinline__ void tc_fetch(const AttArgs& a, const int* keys, const float* KV, int g, int ld, float* tile) {
    for (int x = threadIdx.x; x < TC_KT * a.hd / 4; x += blockDim.x) {
        const int kk = x / (a.hd / 4), c4 = x % (a.hd / 4), key = keys[kk];
        const float* src = KV + (key >= 0 ? ((size_t) key * a.nkv + g) * a.hd + 4 * c4 : 0);
        __pipeline_memcpy_async(tile + kk * ld + 4 * c4, src, 16, key >= 0 ? 0 : 16);
    }
    __pipeline_commit();
}
// mma.sync m16n8k8 TF32. Lane (g = lane / 4, t = lane % 4) holds a = {A[g][t], A[g+8][t], A[g][t+4], A[g+8][t+4]},
// b = {B[t][g], B[t+4][g]}, c = {C[g][2t], C[g][2t+1], C[g+8][2t], C[g+8][2t+1]}
static __device__ __forceinline__ uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}
static __device__ __forceinline__ void mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int N> static __device__ __forceinline__ void tf32_parts(const float* x, uint32_t* hi, uint32_t* lo) {
#pragma unroll
    for (int i = 0; i < N; ++i) { hi[i] = to_tf32(x[i]); lo[i] = to_tf32(x[i] - __uint_as_float(hi[i])); }
}
static size_t tc_smem(int hd) { return sizeof(float) * ((size_t) TC_KT * (hd + 8) + 4 * 256 + 16 * TC_PS); }
__global__ void __launch_bounds__(128, 4) k_q4_attend_tc(AttArgs a, const float* q, const float* K, const float* V, const int32_t* sel,
                                                         const int32_t* nsel, const int32_t* dpos, float* o) {
    using namespace nvcuda;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> FragA;
    extern __shared__ __align__(32) float tcs[];   // the Q, then K or V, tile [16][ld], 4 partial S, P [16][TC_PS]
    __shared__ int keys[TC_KT];
    __shared__ float mx[16], sum[16], alpha[16];
    const int hd = a.hd, ld = hd + 8, G = a.nh / a.nkv, g = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, tid = threadIdx.x, ds = hd / 4;
    const int gq = (tid & 31) >> 2, tq = tid & 3;   // the lane's mma.sync rows gq, gq + 8 and columns 2 tq, 2 tq + 1
    const int pos = a.pos0 + (dpos ? *dpos : 0) + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    float* Ks = tcs, *Sp = Ks + TC_KT * ld, *Ps = Sp + 4 * 256;
    const float* qg = q + (size_t) t * a.qld + (size_t) g * G * 2 * hd;   // head h of the group: qg + h * 2 hd
    for (int x = tid; x < 16 * hd; x += blockDim.x) Ks[x / hd * ld + x % hd] = x / hd < G ? qg[(size_t) (x / hd) * 2 * hd + x % hd] : 0.0f;
    if (tid < 16) { mx[tid] = -INFINITY; sum[tid] = 0.0f; }
    __syncthreads();
    FragA qh[8], ql[8];   // the warp's dims [w ds, (w + 1) ds), 8 a fragment
#pragma unroll
    for (int j = 0; j < 8; ++j)
        if (j < ds / 8) {
            FragA f;
            wmma::load_matrix_sync(f, Ks + w * ds + 8 * j, ld);
            tf32_split(f, qh[j], ql[j]);
        }
    float oc[8][4];   // the warp's O columns [w ds, (w + 1) ds): 8 n-tiles of 8, in the mma.sync c layout
#pragma unroll
    for (int j = 0; j < 8; ++j) oc[j][0] = oc[j][1] = oc[j][2] = oc[j][3] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += TC_KT) {
        const int nk = min(TC_KT, n - k0);
        __syncthreads();   // the previous tile (or Q) is read
        if (tid < TC_KT) {
            const int k = k0 + tid;
            keys[tid] = tid >= nk ? -1 : k < ns * a.R ? sel[(size_t) t * a.topk + k / a.R] * a.R + k % a.R : tail + (k - ns * a.R);
        }
        __syncthreads();
        tc_fetch(a, keys, K, g, ld, Ks);
        __pipeline_wait_prior(0);
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> sacc;   // warp w: the dims [w ds, (w + 1) ds) of S
        wmma::fill_fragment(sacc, 0.0f);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (j < ds / 8) {
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> kb;
                wmma::load_matrix_sync(kb, Ks + w * ds + 8 * j, ld);   // K^T: the tile [key][dim] read column-major
                mma3s(sacc, qh[j], ql[j], kb);
            }
        wmma::store_matrix_sync(Sp + w * 256, sacc, 16, wmma::mem_row_major);
        __syncthreads();   // S is done with the K tile: V takes it, while the softmax runs
        tc_fetch(a, keys, V, g, ld, Ks);
        {   // row r: 8 threads of 2 keys each. The new maximum, the scale of the row's old O and sum, its probabilities
            const int r = tid >> 3, h = tid & 7;
            float sc[2], mt = -INFINITY;
            for (int u = 0; u < 2; ++u) {
                const int kk = 2 * h + u, x = r * 16 + kk;
                sc[u] = kk < nk && r < G ? (Sp[x] + Sp[256 + x] + Sp[512 + x] + Sp[768 + x]) * a.scale : -INFINITY;
                mt = fmaxf(mt, sc[u]);
            }
            for (int sh = 1; sh < 8; sh <<= 1) mt = fmaxf(mt, __shfl_xor_sync(FULL, mt, sh));
            const float m0 = mx[r], m1 = fmaxf(m0, mt);
            float ps = 0;
            for (int u = 0; u < 2; ++u) {
                const float p = sc[u] == -INFINITY ? 0.0f : expf(sc[u] - m1);
                Ps[r * TC_PS + 2 * h + u] = p;
                ps += p;
            }
            for (int sh = 1; sh < 8; sh <<= 1) ps += __shfl_xor_sync(FULL, ps, sh);
            if (!h) {
                const float al = m0 == -INFINITY ? 0.0f : expf(m0 - m1);
                alpha[r] = al;
                sum[r] = sum[r] * al + ps;
                mx[r] = m1;
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();
        const float a0 = alpha[gq], a1 = alpha[gq + 8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { oc[j][0] *= a0; oc[j][1] *= a0; oc[j][2] *= a1; oc[j][3] *= a1; }
#pragma unroll
        for (int k8 = 0; k8 < TC_KT; k8 += 8) {
            const float pv[4] = {Ps[gq * TC_PS + k8 + tq], Ps[(gq + 8) * TC_PS + k8 + tq], Ps[gq * TC_PS + k8 + tq + 4], Ps[(gq + 8) * TC_PS + k8 + tq + 4]};
            uint32_t ph[4], pl[4];
            tf32_parts<4>(pv, ph, pl);
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (j < ds / 8) {
                    const float* vb = Ks + (size_t) (k8 + tq) * ld + w * ds + 8 * j + gq;
                    const float bv[2] = {vb[0], vb[4 * ld]};
                    uint32_t bh[2], bl[2];
                    tf32_parts<2>(bv, bh, bl);
                    mma_tf32(oc[j], ph, bh);
                    mma_tf32(oc[j], ph, bl);
                    mma_tf32(oc[j], pl, bh);
                }
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int u = 0; u < 4; ++u) {   // O / sum, gated: rows gq (u < 2) and gq + 8, columns 2 tq + u % 2
            const int h = gq + 8 * (u >> 1), i = w * ds + 8 * j + 2 * tq + (u & 1);
            if (j < ds / 8 && h < G) o[((size_t) t * a.nh + (size_t) g * G + h) * hd + i] = oc[j][u] / sum[h] * sigm(qg[(size_t) h * 2 * hd + hd + i]);
        }
}

static int attend_tc(const Qwen4Cfg* c) {   // the tensor-core kernel fits, and may take its shared memory
    static int set;
    if (!set) set = cudaFuncSetAttribute(k_q4_attend_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) tc_smem(256)) == cudaSuccess;
    return set && c->n_head / c->n_kv <= 16 && c->head_dim % 64 == 0;
}
void kq4_attend_info(const Qwen4Cfg* c, int* regs, int* blocks) {   // the prompt-row kernel
    const void* f = attend_tc(c) ? (const void*) k_q4_attend_tc : (const void*) k_q4_attend_g;
    const int nt = attend_tc(c) ? 128 : 32 * (c->n_head / c->n_kv);
    const size_t sm = attend_tc(c) ? tc_smem(c->head_dim) : sizeof(float) * 2 * AT_KT * (size_t) c->head_dim;
    cudaFuncAttributes fa;
    *regs = cudaFuncGetAttributes(&fa, f) == cudaSuccess ? fa.numRegs : -1;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, f, nt, sm) != cudaSuccess) *blocks = -1;
}

// Decode rows: block (KV group, split, row), a warp per head of the group. Split sp takes keys [n sp / AD_S, n (sp + 1) / AD_S)
// of the row's n. Their K and V come in tiles of AT_KT through shared memory, once for all heads. part: m, l, acc[hd].
#define AD_S 32
__global__ void k_q4_attend_d(AttArgs a, const float* q, const float* K, const float* V, const int32_t* sel, const int32_t* nsel,
                              const int32_t* dpos, float* part) {
    __shared__ float Ks[AT_KT][256], Vs[AT_KT][256];
    __shared__ int keys[AT_KT];
    const int g = blockIdx.x, sp = blockIdx.y, t = blockIdx.z, w = threadIdx.x >> 5, lane = threadIdx.x & 31, dl = a.hd / 32;
    const int h = g * (a.nh / a.nkv) + w;
    const int pos = a.pos0 + (dpos ? *dpos : 0) + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    const int k0 = (int) ((int64_t) n * sp / AD_S), k1 = (int) ((int64_t) n * (sp + 1) / AD_S);
    const float* qh = q + (size_t) t * a.qld + (size_t) h * 2 * a.hd;
    float qv[8], acc[8], m = -INFINITY, l = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { qv[i] = i < dl ? qh[lane + 32 * i] : 0.0f; acc[i] = 0; }
    for (int b = k0; b < k1; b += AT_KT) {
        const int nk = min(AT_KT, k1 - b);
        __syncthreads();   // the previous tile is read
        if (threadIdx.x < nk) {
            const int k = b + threadIdx.x;
            keys[threadIdx.x] = k < ns * a.R ? sel[(size_t) t * a.topk + k / a.R] * a.R + k % a.R : tail + (k - ns * a.R);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nk * a.hd; i += blockDim.x) {
            const size_t off = ((size_t) keys[i / a.hd] * a.nkv + g) * a.hd + i % a.hd;
            Ks[i / a.hd][i % a.hd] = K[off];
            Vs[i / a.hd][i % a.hd] = V[off];
        }
        __syncthreads();
        float d[AT_KT], mx = m;
#pragma unroll
        for (int j = 0; j < AT_KT; ++j) {
            d[j] = 0;
#pragma unroll
            for (int i = 0; i < 8; ++i) if (i < dl) d[j] += qv[i] * Ks[j][lane + 32 * i];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)   // the tile's dot products reduced together, as warp_sum does each
#pragma unroll
            for (int j = 0; j < AT_KT; ++j) d[j] += __shfl_xor_sync(FULL, d[j], o);
#pragma unroll
        for (int j = 0; j < AT_KT; ++j) if (j < nk) { d[j] *= a.scale; mx = fmaxf(mx, d[j]); }
        const float cf = expf(m - mx);
        l *= cf;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= cf;
#pragma unroll
        for (int j = 0; j < AT_KT; ++j)
            if (j < nk) {
                const float p = expf(d[j] - mx);
                l += p;
#pragma unroll
                for (int i = 0; i < 8; ++i) if (i < dl) acc[i] += p * Vs[j][lane + 32 * i];
            }
        m = mx;
    }
    float* pp = part + (((size_t) t * a.nh + h) * AD_S + sp) * (a.hd + 2);
    if (lane == 0) { pp[0] = m; pp[1] = l; }
    for (int i = 0; i < dl; ++i) pp[2 + lane + 32 * i] = acc[i];
}
// Block (head, row): the AD_S splits of k_q4_attend_d, then the output gate
__global__ void k_q4_attend_dc(AttArgs a, const float* q, const float* part, float* o) {
    __shared__ float wt[AD_S];
    __shared__ float L;
    const int h = blockIdx.x, t = blockIdx.y, st = a.hd + 2;
    const float* p = part + ((size_t) t * a.nh + h) * AD_S * st;
    if (threadIdx.x < 32) {
        const float ms = threadIdx.x < AD_S ? p[threadIdx.x * st] : -INFINITY, M = warp_max(ms);
        const float e = ms == -INFINITY ? 0.0f : expf(ms - M);
        if (threadIdx.x < AD_S) wt[threadIdx.x] = e;
        const float sl = warp_sum(threadIdx.x < AD_S ? p[threadIdx.x * st + 1] * e : 0.0f);
        if (threadIdx.x == 0) L = sl;
    }
    __syncthreads();
    const float* qh = q + (size_t) t * a.qld + (size_t) h * 2 * a.hd;
    for (int i = threadIdx.x; i < a.hd; i += blockDim.x) {
        float v = 0;
        for (int x = 0; x < AD_S; ++x) v += p[x * st + 2 + i] * wt[x];
        o[((size_t) t * a.nh + h) * a.hd + i] = v / L * sigm(qh[a.hd + i]);
    }
}
size_t kq4_attend_part_len(const Qwen4Cfg* c) { return (size_t) 8 * c->n_head * AD_S * (c->head_dim + 2); }
int kq4_attend(cudaStream_t s, const Qwen4Cfg* c, const float* q, const float* K, const float* V, int pos0, const int32_t* dpos,
               const int32_t* sel, const int32_t* nsel, float* o, int T) {
    float* part = NULL;
    if (T <= 8 && cudaMalloc(&part, sizeof(float) * kq4_attend_part_len(c)) != cudaSuccess) return -1;
    const int rc = kq4_attend_part(s, c, q, c->n_head * 2 * c->head_dim, K, V, pos0, dpos, sel, nsel, o, part, T);
    cudaFree(part);
    return rc;
}
int kq4_attend_part(cudaStream_t s, const Qwen4Cfg* c, const float* q, int qld, const float* K, const float* V, int pos0,
                    const int32_t* dpos, const int32_t* sel, const int32_t* nsel, float* o, float* part, int T) {
    if (c->head_dim % 32 || c->head_dim > 256 || c->n_head % c->n_kv || c->n_head / c->n_kv > 32) return -1;
    const AttArgs a = {c->n_head, c->n_kv, c->head_dim, c->ix_ratio, c->ix_budget / c->ix_ratio, pos0, qld, 1.0f / sqrtf((float) c->head_dim)};
    const int G = c->n_head / c->n_kv;
    if (T <= 8) {   // decode rows: keys split over AD_S blocks a KV group, then the splits combined
        k_q4_attend_d<<<dim3((unsigned) c->n_kv, AD_S, (unsigned) T), 32 * (unsigned) G, 0, s>>>(a, q, K, V, sel, nsel, dpos, part);
        k_q4_attend_dc<<<dim3((unsigned) c->n_head, (unsigned) T), 256, 0, s>>>(a, q, part, o);
    } else if (attend_tc(c))   // prompt rows on tensor cores
        k_q4_attend_tc<<<dim3((unsigned) c->n_kv, (unsigned) T), 128, tc_smem(c->head_dim), s>>>(a, q, K, V, sel, nsel, dpos, o);
    else   // prompt rows: the group's heads share each key's reads (12x less K / V traffic at 24 / 2 heads)
        k_q4_attend_g<<<dim3((unsigned) c->n_kv, (unsigned) T), 32 * (unsigned) G, sizeof(float) * 2 * AT_KT * (size_t) c->head_dim, s>>>(
            a, q, K, V, sel, nsel, dpos, o);
    return 0;
}

