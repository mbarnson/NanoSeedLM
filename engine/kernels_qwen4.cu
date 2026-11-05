// engine/kernels_qwen4.cu - CUDA kernels of qwen4 (engine/kernels_qwen4.h), the counterparts of nslm/qwen4_ops.h.
// f32 values and sums.
#include <math.h>
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

// One thread per channel, the T tokens in order: cv = silu(causal conv of the last K inputs); conv keeps the last K - 1.
__global__ void k_q4_gdn_conv(int cd, int K, const float* w, const float* qkv, float* conv, float* cv, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= cd) return;
    float st[7], wk[8];
    for (int k = 0; k < K - 1; ++k) st[k] = conv[(size_t) k * cd + ch];
    for (int k = 0; k < K; ++k) wk[k] = w[(size_t) ch * K + k];
    for (int t = 0; t < T; ++t) {
        const float x = qkv[(size_t) t * cd + ch];
        float acc = wk[K - 1] * x;
        for (int k = 0; k < K - 1; ++k) acc += wk[k] * st[k];
        cv[(size_t) t * cd + ch] = acc * sigm(acc);
        for (int k = 0; k + 1 < K - 1; ++k) st[k] = st[k + 1];
        if (K > 1) st[K - 2] = x;
    }
    for (int k = 0; k < K - 1; ++k) conv[(size_t) k * cd + ch] = st[k];
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
    int cd, kd, vd, hv, rep, dv;
} GdnArgs;

// Block = 32 value rows of head hv (grid hv x dv / 32); SPLIT threads hold one row's state column S[hv][.][i] in registers
// over the T tokens, DK / SPLIT values each.  q and k come normalised (k_q4_gdn_qknorm).  y gets the raw read S q;
// k_q4_gdn_norm normalises and gates it.
template <int DK, int SPLIT>
__global__ void __launch_bounds__(32 * SPLIT) k_q4_gdn_step(GdnArgs a, const float* cv, const float* aa, const float* bb,
                                                            const float* A_log, const float* dt_bias, float* S, float* y, int T) {
    constexpr int DJ = DK / SPLIT;
    const int hv = blockIdx.x, p = threadIdx.x % SPLIT, i = blockIdx.y * 32 + threadIdx.x / SPLIT, kh = hv / a.rep;
    __shared__ float q[DK], k[DK];
    float s[DJ];
    float* St = S + (size_t) hv * DK * a.dv + (size_t) p * DJ * a.dv + i;
#pragma unroll
    for (int j = 0; j < DJ; ++j) s[j] = St[(size_t) j * a.dv];
    const float A = expf(A_log[hv]), dtb = dt_bias[hv];
    for (int t = 0; t < T; ++t) {
        const float* c = cv + (size_t) t * a.cd;
        __syncthreads();   // the previous token's reads of q and k are done
        for (int j = threadIdx.x; j < DK; j += blockDim.x) { q[j] = c[kh * DK + j]; k[j] = c[a.kd + kh * DK + j]; }
        __syncthreads();
        const float ap = aa[(size_t) t * a.hv + hv] + dtb, sp = ap > 20.0f ? ap : log1pf(expf(ap));
        const float g = expf(-A * sp), beta = sigm(bb[(size_t) t * a.hv + hv]);
        const float* kp = k + p * DJ;
        const float* qp = q + p * DJ;
        float kv = 0;
#pragma unroll
        for (int j = 0; j < DJ; ++j) kv += s[j] * kp[j];
        if (SPLIT == 2) kv += __shfl_xor_sync(FULL, kv, 1);
        const float delta = (c[2 * a.kd + hv * a.dv + i] - g * kv) * beta;
        float yq = 0;
#pragma unroll
        for (int j = 0; j < DJ; ++j) { s[j] = g * s[j] + delta * kp[j]; yq += s[j] * qp[j]; }
        if (SPLIT == 2) yq += __shfl_xor_sync(FULL, yq, 1);
        if (p == 0) y[(size_t) t * a.vd + (size_t) hv * a.dv + i] = yq;
    }
#pragma unroll
    for (int j = 0; j < DJ; ++j) St[(size_t) j * a.dv] = s[j];
}

// Block (head, token), thread = value row: RMSNorm over the head (plain weight), gated by sigmoid(z)
__global__ void k_q4_gdn_norm(int hv, int dv, float eps, const float* norm_w, const float* z, float* y) {
    __shared__ float red[32];
    const size_t o = ((size_t) blockIdx.y * hv + blockIdx.x) * dv + threadIdx.x;
    const float v = y[o], r = rsqrtf(block_sum(v * v, red) / dv + eps);
    y[o] = v * r * norm_w[threadIdx.x] * sigm(z[o]);
}

int kq4_gdn(cudaStream_t s, const Qwen4Cfg* c, Q4GdnDev w, const float* qkv, const float* z, const float* a, const float* b,
            float* conv, float* S, float* cv, float* y, int T) {
    const int kd = c->lin_hk * c->lin_dk, vd = c->lin_hv * c->lin_dv, cd = 2 * kd + vd;
    if (c->lin_conv < 1 || c->lin_conv > 8 || c->lin_dv % 32 || c->lin_dv > 1024) return -1;
    k_q4_gdn_conv<<<(unsigned) (cd + 255) / 256, 256, 0, s>>>(cd, c->lin_conv, w.conv_w, qkv, conv, cv, T);
    k_q4_gdn_qknorm<<<dim3((unsigned) T, (unsigned) c->lin_hk), (unsigned) c->lin_dk, 0, s>>>(cd, kd, cv);
    const GdnArgs ga = {cd, kd, vd, c->lin_hv, c->lin_hv / c->lin_hk, c->lin_dv};
    const dim3 grid((unsigned) c->lin_hv, (unsigned) c->lin_dv / 32);
    switch (c->lin_dk) {
    case 32: k_q4_gdn_step<32, 1><<<grid, 32, 0, s>>>(ga, cv, a, b, w.A_log, w.dt_bias, S, y, T); break;
    case 64: k_q4_gdn_step<64, 1><<<grid, 32, 0, s>>>(ga, cv, a, b, w.A_log, w.dt_bias, S, y, T); break;
    case 128: k_q4_gdn_step<128, 2><<<grid, 64, 0, s>>>(ga, cv, a, b, w.A_log, w.dt_bias, S, y, T); break;
    default: return -1;
    }
    k_q4_gdn_norm<<<dim3((unsigned) c->lin_hv, (unsigned) T), (unsigned) c->lin_dv, 0, s>>>(c->lin_hv, c->lin_dv, c->eps, w.norm_w, z, y);
    return 0;
}

// ---- RMSNorm (1 + w) and the hyper-connection pointwise steps --------------------------------------------------------

__global__ void __launch_bounds__(256) k_q4_norm1p(const float* x, int len, int g, const float* w, float eps, float* y) {
    __shared__ float red[32];
    const int groups = len / g, t = blockIdx.x / groups, a = (blockIdx.x % groups) * g;
    const float* xi = x + (size_t) t * len + a;
    float ss = 0;
    for (int k = threadIdx.x; k < g; k += blockDim.x) ss += xi[k] * xi[k];
    const float r = rsqrtf(block_sum(ss, red) / g + eps);
    for (int k = threadIdx.x; k < g; k += blockDim.x) y[(size_t) t * len + a + k] = xi[k] * r * (1.0f + w[a + k]);
}
void kq4_norm1p(cudaStream_t s, const float* x, int len, int g, const float* w, float eps, float* y, int T) {
    k_q4_norm1p<<<(unsigned) (T * (len / g)), 256, 0, s>>>(x, len, g, w, eps, y);
}

__global__ void k_q4_hc_act(float inv_hc, float* lo, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { const float x = lo[i] * inv_hc; lo[i] = x * sigm(x); }
}
__global__ void k_q4_hc_gate(float inv_hc, float* inj, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inj[i] = 2.0f * sigm(inj[i] * inv_hc);
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
void kq4_hc_act(cudaStream_t s, int hc, float* lo, int n) { k_q4_hc_act<<<(unsigned) (n + 255) / 256, 256, 0, s>>>(1.0f / hc, lo, n); }
void kq4_hc_gate(cudaStream_t s, int hc, float* inj, int n) { k_q4_hc_gate<<<(unsigned) (n + 255) / 256, 256, 0, s>>>(1.0f / hc, inj, n); }
void kq4_hc_mix(cudaStream_t s, int hc, int d, const float* normed, const float* up, float* mixed, int T) {
    k_q4_hc_mix<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(hc, d, normed, up, mixed);
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
static __device__ void norm_rope(float* xh, int hd, const float* w, float eps, int rot, double theta, int pos, float* red, float* v) {
    float ss = 0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += xh[i] * xh[i];
    const float r = rsqrtf(block_sum(ss, red) / hd + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) v[i] = xh[i] * r * (1.0f + w[i]);
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = v[i];
        if (i < rot) {
            const int f = i % (rot / 2);
            const float a = (float) pos * (float) pow(theta, -2.0 * f / rot);
            const double cs = cos((double) a), sn = sin((double) a);
            o = i < rot / 2 ? (float) (v[i] * cs - v[i + rot / 2] * sn) : (float) (v[i] * cs + v[i - rot / 2] * sn);
        }
        xh[i] = o;
    }
}
// Block (head, row) at position pos0 (+ *dpos) + row
__global__ void __launch_bounds__(128) k_q4_head_norm_rope(float* x, int rs, int hs, int hd, const float* w, float eps, int rot,
                                                           double theta, int pos0, const int32_t* dpos) {
    __shared__ float red[32], v[1024];
    norm_rope(x + (size_t) blockIdx.y * rs + (size_t) blockIdx.x * hs, hd, w, eps, rot, theta, pos0 + (dpos ? *dpos : 0) + (int) blockIdx.y, red, v);
}
void kq4_head_norm_rope(cudaStream_t s, const Qwen4Cfg* c, float* x, int rs, int nh, int hs, int hd, const float* w, int pos0,
                        const int32_t* dpos, int T) {
    k_q4_head_norm_rope<<<dim3((unsigned) nh, (unsigned) T), 128, 0, s>>>(x, rs, hs, hd, w, c->eps, c->rot_dim, (double) c->rope_theta, pos0, dpos);
}
// dst[(pos + t) * w + i] = src[t * ss + i], pos = pos0 (+ *dpos)
__global__ void k_q4_store(float* dst, int w, const float* src, int ss, int pos0, const int32_t* dpos) {
    const int t = (int) blockIdx.y, i = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (i < w) dst[(size_t) (pos0 + (dpos ? *dpos : 0) + t) * w + i] = src[(size_t) t * ss + i];
}
void kq4_store(cudaStream_t s, float* dst, int w, const float* src, int ss, int pos0, const int32_t* dpos, int T) {
    k_q4_store<<<dim3((unsigned) (w + 255) / 256, (unsigned) T), 256, 0, s>>>(dst, w, src, ss, pos0, dpos);
}

// One warp per row: k rounds of a warp arg max of the logits (ties to the higher expert); weights exp(l - max) renormalised
__global__ void k_q4_route(int E, int k, const float* logits, int32_t* ind, float* wt) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* l = logits + (size_t) t * E;
    float mx = -INFINITY;
    for (int e = lane; e < E; e += 32) mx = fmaxf(mx, l[e]);
    mx = warp_max(mx);
    float top = 0, w[32];
    for (int r = 0; r < k; ++r) {
        float bv = -INFINITY;
        int bi = -1;
        for (int e = lane; e < E; e += 32) {
            bool taken = false;
            for (int q = 0; q < r; ++q) taken |= ind[(size_t) t * k + q] == e;
            if (!taken && (l[e] > bv || (l[e] == bv && e > bi))) { bv = l[e]; bi = e; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(FULL, bv, o);
            const int oi = __shfl_xor_sync(FULL, bi, o);
            if (ov > bv || (ov == bv && oi > bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) ind[(size_t) t * k + r] = bi;
        __syncwarp();
        w[r] = expf(bv - mx);
        top += w[r];
    }
    if (lane == 0) for (int r = 0; r < k; ++r) wt[(size_t) t * k + r] = w[r] / top;
}
void kq4_route(cudaStream_t s, const Qwen4Cfg* c, const float* logits, int32_t* ind, float* wt, int T) {
    k_q4_route<<<(unsigned) T, 32, 0, s>>>(c->n_exp, c->top_k, logits, ind, wt);
}

__global__ void k_q4_moe_combine(int d, int k, const float* D, const float* wt, const float* sh, const float* sg, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (i >= d) return;
    float a = sigm(sg[t]) * sh[(size_t) t * d + i];
    for (int j = 0; j < k; ++j) a += wt[t * k + j] * D[((size_t) t * k + j) * d + i];
    out[(size_t) t * d + i] = a;
}
void kq4_moe_combine(cudaStream_t s, int d, int k, const float* D, const float* wt, const float* sh, const float* sg, float* out, int T) {
    k_q4_moe_combine<<<dim3((unsigned) (d + 255) / 256, (unsigned) T), 256, 0, s>>>(d, k, D, wt, sh, sg, out);
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
__global__ void k_q4_ple_conv(int HD, int K, int dil, const float* w, const float* cin, const float* gated, float* hist, float* H, int T) {
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
    }
    for (int j = 0; j < L; ++j) hist[(size_t) j * HD + ch] = st[j];
}
void kq4_ple_conv(cudaStream_t s, const Qwen4Cfg* c, const float* w, const float* cin, const float* gated, float* hist, float* H, int T) {
    const int HD = c->hc * c->d;
    k_q4_ple_conv<<<(unsigned) (HD + 255) / 256, 256, 0, s>>>(HD, c->ple_conv, c->ngram, w, cin, gated, hist, H, T);
}

// ---- QSA ---------------------------------------------------------------------------------------------------------------

// Block b: key block j = pos / R + b if rows [pos, pos + T) complete it: the mean of its R raw keys, then norm + RoPE at
// position j * R
__global__ void __launch_bounds__(128) k_q4_qsa_blocks(int D, int R, const float* raw, const float* w, float eps, int rot, double theta,
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
    norm_rope(bj, D, w, eps, rot, theta, j * R, red, v);
}
void kq4_qsa_blocks(cudaStream_t s, const Qwen4Cfg* c, const float* raw, const float* w, int pos0, const int32_t* dpos, int T, float* blk) {
    k_q4_qsa_blocks<<<(unsigned) (T / c->ix_ratio + 1), 128, 0, s>>>(c->ix_dim, c->ix_ratio, raw, w, c->eps, c->rot_dim, (double) c->rope_theta,
                                                                 pos0, dpos, T, blk);
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

// Warp per (row, block): sum over heads of relu(q_h . blk_j)
__global__ void __launch_bounds__(256) k_q4_qsa_scores(int nh, int D, const float* iq, int rs, const float* blk, const int32_t* done,
                                                       int maxdone, float* sc) {
    const int t = blockIdx.y, j = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (j >= done[t]) return;
    float s = 0;
    for (int h = 0; h < nh; ++h) {
        float d = 0;
        for (int i = lane; i < D; i += 32) d += iq[(size_t) t * rs + h * D + i] * blk[(size_t) j * D + i];
        d = warp_sum(d);
        s += fmaxf(d, 0.0f);
    }
    if (lane == 0) sc[(size_t) t * maxdone + j] = s;
}
// Block per row: the k-th largest score by a radix select on its bits (scores are >= 0), then every score above it in
// block order (attention sums the keys in this order, so it must not depend on timing) and, of the equal ones, the highest
__global__ void __launch_bounds__(256) k_q4_qsa_topk(const float* sc, const int32_t* done, int maxdone, int k, int32_t* sel) {
    __shared__ float red[32];
    __shared__ int nout, wcnt[8];
    const int t = blockIdx.x, n = done[t];
    if (n <= k) return;
    const uint32_t* u = (const uint32_t*) (sc + (size_t) t * maxdone);
    uint32_t p = 0;
    for (int b = 31; b >= 0; --b) {
        const uint32_t cand = p | (1u << b);
        float cnt = 0;
        for (int j = threadIdx.x; j < n; j += blockDim.x) cnt += u[j] >= cand;
        if (block_sum(cnt, red) >= (float) k) p = cand;
    }
    if (threadIdx.x == 0) nout = 0;
    __syncthreads();
    int32_t* out = sel + (size_t) t * k;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int j0 = 0; j0 < n; j0 += 256) {   // a block-wide prefix of the matches
        const int j = j0 + threadIdx.x, f = j < n && u[j] > p;
        const uint32_t m = __ballot_sync(FULL, f);
        if (lane == 0) wcnt[w] = __popc(m);
        __syncthreads();
        int before = nout;
        for (int x = 0; x < w; ++x) before += wcnt[x];
        if (f) out[before + __popc(m & ((1u << lane) - 1u))] = j;
        __syncthreads();
        if (threadIdx.x == 0) for (int x = 0; x < 8; ++x) nout += wcnt[x];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int j = n - 1, m = nout; j >= 0 && m < k; --j) if (u[j] == p) out[m++] = j;
}
void kq4_qsa_select(cudaStream_t s, const Qwen4Cfg* c, const float* iq, int rs, const float* blk, const int32_t* done, int maxdone,
                    float* sc, int32_t* sel, int T) {
    if (maxdone <= 0) return;
    k_q4_qsa_scores<<<dim3((unsigned) (maxdone + 7) / 8, (unsigned) T), 256, 0, s>>>(c->ix_heads, c->ix_dim, iq, rs, blk, done, maxdone, sc);
    k_q4_qsa_topk<<<(unsigned) T, 256, 0, s>>>(sc, done, maxdone, c->ix_budget / c->ix_ratio, sel);
}

// ---- attention ---------------------------------------------------------------------------------------------------------

typedef struct {
    int nh, nkv, hd, R, topk, pos0;
    float scale;
} AttArgs;

// Block (head, row), 8 warps: each warp takes every 8th visible key with an online softmax (lanes hold hd / 32 dims of q
// and of the output), then the warps merge; the output is gated by sigmoid of the query's gate half
__global__ void __launch_bounds__(256) k_q4_attend(AttArgs a, const float* q, const float* K, const float* V, const int32_t* sel,
                                                   const int32_t* nsel, const int32_t* dpos, float* o) {
    __shared__ float sm[8], sl[8], sacc[8][256];
    const int h = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, dl = a.hd / 32, g = h / (a.nh / a.nkv);
    const int pos = a.pos0 + (dpos ? *dpos : 0) + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    const float* qh = q + ((size_t) t * a.nh + h) * 2 * a.hd;
    float qv[8], acc[8], m = -INFINITY, l = 0;
    for (int i = 0; i < dl; ++i) { qv[i] = qh[lane + 32 * i]; acc[i] = 0; }
    for (int k = w; k < n; k += 8) {
        const int key = k < ns * a.R ? sel[(size_t) t * a.topk + k / a.R] * a.R + k % a.R : tail + (k - ns * a.R);
        const float* kk = K + ((size_t) key * a.nkv + g) * a.hd;
        float d = 0;
        for (int i = 0; i < dl; ++i) d += qv[i] * kk[lane + 32 * i];
        d = warp_sum(d) * a.scale;
        const float mn = fmaxf(m, d), cf = expf(m - mn), p = expf(d - mn);
        const float* vv = V + ((size_t) key * a.nkv + g) * a.hd;
        for (int i = 0; i < dl; ++i) acc[i] = acc[i] * cf + p * vv[lane + 32 * i];
        l = l * cf + p;
        m = mn;
    }
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    for (int i = 0; i < dl; ++i) sacc[w][lane + 32 * i] = acc[i];
    __syncthreads();
    float M = -INFINITY;
    for (int x = 0; x < 8; ++x) M = fmaxf(M, sm[x]);
    float L = 0;
    for (int x = 0; x < 8; ++x) L += sm[x] == -INFINITY ? 0.0f : sl[x] * expf(sm[x] - M);
    for (int i = threadIdx.x; i < a.hd; i += blockDim.x) {
        float v = 0;
        for (int x = 0; x < 8; ++x) if (sm[x] != -INFINITY) v += sacc[x][i] * expf(sm[x] - M);
        o[((size_t) t * a.nh + h) * a.hd + i] = v / L * sigm(qh[a.hd + i]);
    }
}
// Prompt rows: block (KV group, row), a warp per query head of the group. The row's keys come in tiles of AT_KT; each
// key's K and V is read once into shared memory for all heads of the group. Each head takes one online-softmax step a tile
#define AT_KT 16
__global__ void k_q4_attend_g(AttArgs a, const float* q, const float* K, const float* V, const int32_t* sel, const int32_t* nsel,
                              const int32_t* dpos, float* o) {
    extern __shared__ float kvs[];   // K [AT_KT][hd], then V [AT_KT][hd]
    __shared__ int keys[AT_KT];
    const int g = blockIdx.x, t = blockIdx.y, w = threadIdx.x >> 5, lane = threadIdx.x & 31, dl = a.hd / 32, h = g * (a.nh / a.nkv) + w;
    const int pos = a.pos0 + (dpos ? *dpos : 0) + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    const float* qh = q + ((size_t) t * a.nh + h) * 2 * a.hd;
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

int kq4_attend(cudaStream_t s, const Qwen4Cfg* c, const float* q, const float* K, const float* V, int pos0, const int32_t* dpos,
               const int32_t* sel, const int32_t* nsel, float* o, int T) {
    if (c->head_dim % 32 || c->head_dim > 256 || c->n_head % c->n_kv) return -1;
    const AttArgs a = {c->n_head, c->n_kv, c->head_dim, c->ix_ratio, c->ix_budget / c->ix_ratio, pos0, 1.0f / sqrtf((float) c->head_dim)};
    const int G = c->n_head / c->n_kv;
    if (T > 8 && G <= 32)   // prompt rows: the group's heads share each key's reads (12x less K / V traffic at 24 / 2 heads)
        k_q4_attend_g<<<dim3((unsigned) c->n_kv, (unsigned) T), 32 * (unsigned) G, sizeof(float) * 2 * AT_KT * (size_t) c->head_dim, s>>>(
            a, q, K, V, sel, nsel, dpos, o);
    else k_q4_attend<<<dim3((unsigned) c->n_head, (unsigned) T), 256, 0, s>>>(a, q, K, V, sel, nsel, dpos, o);
    return 0;
}

