// engine/kernels_qwen4.cu - CUDA kernels of Qwen3.8 (engine/kernels_qwen4.h), the counterparts of nslm/qwen4_ops.h.
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

typedef struct {
    int cd, kd, vd, hv, rep, dv;
} GdnArgs;

// Block = 32 value rows of head hv (grid hv x dv / 32); SPLIT threads hold one row's state column S[hv][.][i] in registers
// over the T tokens, DK / SPLIT values each.  y gets the raw read S q; k_q4_gdn_norm normalises and gates it.
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
        float qq = 0, kk = 0;
        for (int j = 0; j < DK; ++j) { qq += q[j] * q[j]; kk += k[j] * k[j]; }
        const float sq = rsqrtf(qq + 1e-6f) * rsqrtf((float) DK), sk = rsqrtf(kk + 1e-6f);   // L2 norms; q also dk^-0.5
        const float ap = aa[(size_t) t * a.hv + hv] + dtb, sp = ap > 20.0f ? ap : log1pf(expf(ap));
        const float g = expf(-A * sp), beta = sigm(bb[(size_t) t * a.hv + hv]);
        const float* kp = k + p * DJ;
        const float* qp = q + p * DJ;
        float kv = 0;
#pragma unroll
        for (int j = 0; j < DJ; ++j) kv += s[j] * kp[j];
        if (SPLIT == 2) kv += __shfl_xor_sync(FULL, kv, 1);
        const float delta = (c[2 * a.kd + hv * a.dv + i] - g * kv * sk) * beta;
        float yq = 0;
#pragma unroll
        for (int j = 0; j < DJ; ++j) { s[j] = g * s[j] + delta * kp[j] * sk; yq += s[j] * qp[j]; }
        if (SPLIT == 2) yq += __shfl_xor_sync(FULL, yq, 1);
        if (p == 0) y[(size_t) t * a.vd + (size_t) hv * a.dv + i] = yq * sq;
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
