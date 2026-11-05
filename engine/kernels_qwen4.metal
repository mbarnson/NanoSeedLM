// engine/kernels_qwen4.metal - Metal kernels of qwen4 (out/res/kernels_qwen4.metallib), the counterparts of
// nslm/qwen4_ops.h and of CUDA's engine/kernels_qwen4.cu.  f32 values and sums, compiled without fast math.
// C includes this file for the argument structs only.
#ifndef NSLM_KERNELS_QWEN4_METAL
#define NSLM_KERNELS_QWEN4_METAL

typedef struct { int cd, K, T, keep; } Q4ConvArgs;                   // GDN causal conv: channels, kernel, tokens; keep: below
typedef struct { int cd, kd, vd, hv, rep, dv, T, keep; } Q4GdnArgs;   // GDN step: conv_dim, hk * dk, hv * dv, ...
typedef struct { int hv, dv; float eps; int pad; } Q4GNormArgs;       // GDN output norm
typedef struct { int len, g; float eps; int parts; } Q4NormArgs;      // RMSNorm (1 + w) in groups of g, each written by parts threadgroups
typedef struct { int hc, d, n; float inv_hc; } Q4HcArgs;              // hyper-connection pointwise steps
typedef struct { int rs, hs, hd, rot, pos0, pstep; float eps; int pad; } Q4RopeArgs;   // row t at pos0 + t * pstep
typedef struct { int E, k, pad[2]; } Q4RouteArgs;                     // experts, top k (<= 32)
typedef struct { int hc, d, pad[2]; } Q4PleGateArgs;
typedef struct { int HD, K, dil, T, keep, pad[3]; } Q4PleConvArgs;   // (K - 1) * dil <= 32
// keep (a verify forward's rows, at most Q4E_KEEP_SLOTS + 1): the state after each row t < T - 1 also to snapshot buffer t
#define Q4E_KEEP_SLOTS 3
typedef struct { int D, R, pad[2]; } Q4QsaMeanArgs;                   // block keys: ix_dim, ix_ratio
typedef struct { int nh, D, rs, maxdone, k, pad[3]; } Q4QsaArgs;      // scores and top k
// hd a multiple of 32, <= 256. nsplit: k_q4_attend_g's key splits (its grid's depth). smin > 0: decode rows by their own
// keys n, so that a row's sums do not depend on the other rows of its forward: rows with n >= smin in k_q4_attend_g, in
// min(smax, ceil(n / skeys)) splits (smax 0: ATG_MAX_SPLIT; merged by k_q4_attend_comb), the others in k_q4_attend
typedef struct { int nh, nkv, hd, R, topk, pos0; float scale; int nsplit, smin, skeys, smax, pad; } Q4AttArgs;
#define ATG_MAX_SPLIT 32
typedef struct { int w, ss, pos0, pad; } Q4StoreArgs;                 // cache rows: width, source row stride, first position
typedef struct { int d, k, pad[2]; } Q4CombineArgs;                   // MoE combine: hidden size, top k
#ifndef GDN_SPLIT128
#define GDN_SPLIT128 4   // k_q4_gdn_step, dk 128: adjacent lanes that share a value row's state (dk / GDN_SPLIT128 values each)
#endif
#define Q4_GDN_SPLIT(dk) ((dk) == 128 ? GDN_SPLIT128 : 1)   // threads 32 * Q4_GDN_SPLIT(dk) a threadgroup
#ifndef ATG_NQ
#define ATG_NQ 4   // k_q4_attend_g: simdgroups a head tile (the head dims in ATG_NQ parts); 2 ATG_NQ in all
#endif

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
// The sum of v over the threadgroup (nsg simdgroups of 32), the same value in every thread
static inline float tg_sum(float v, threadgroup float* red, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);   // an earlier tg_sum's reads of red are done
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0;
    for (uint k = 0; k < nsg; ++k) s += red[k];
    return s;
}

// ---- Gated DeltaNet ---------------------------------------------------------------------------------------------------

// One thread per channel, the T tokens in order: cv = silu(causal conv of the last K inputs); conv keeps the last K - 1.
// keep: also the state after each row t < T - 1 at k0, k1, k2 (t)
kernel void k_q4_gdn_conv(constant Q4ConvArgs& a [[buffer(0)]], device const float* w [[buffer(1)]], device const float* qkv [[buffer(2)]],
                          device float* conv [[buffer(3)]], device float* cv [[buffer(4)]], device float* k0 [[buffer(5)]],
                          device float* k1 [[buffer(6)]], device float* k2 [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
    const int ch = (int) gid, cd = a.cd, K = a.K;
    if (ch >= cd) return;
    float st[7], wk[8];
    for (int k = 0; k < K - 1; ++k) st[k] = conv[(ulong) k * cd + ch];
    for (int k = 0; k < K; ++k) wk[k] = w[(ulong) ch * K + k];
    for (int t = 0; t < a.T; ++t) {
        const float x = qkv[(ulong) t * cd + ch];
        float acc = wk[K - 1] * x;
        for (int k = 0; k < K - 1; ++k) acc += wk[k] * st[k];
        cv[(ulong) t * cd + ch] = acc * sigm(acc);
        for (int k = 0; k + 1 < K - 1; ++k) st[k] = st[k + 1];
        if (K > 1) st[K - 2] = x;
        if (a.keep && t + 1 < a.T) {
            device float* ks = t == 0 ? k0 : t == 1 ? k1 : k2;
            for (int k = 0; k < K - 1; ++k) ks[(ulong) k * cd + ch] = st[k];
        }
    }
    for (int k = 0; k < K - 1; ++k) conv[(ulong) k * cd + ch] = st[k];
}

// Threadgroup = 32 value rows of head hv (grid hv x dv / 32); SPLIT threads (adjacent lanes) hold one row's state column
// S[hv][.][i] over the T tokens, DK / SPLIT values each.  y gets the raw read S q; k_q4_gdn_norm normalises and gates it.
// keep: also the state after each row t < T - 1 at k0, k1, k2 (t)
#define GDN_STEP_ARGS                                                                                                      \
    constant Q4GdnArgs& a [[buffer(0)]], device const float* cv [[buffer(1)]], device const float* aa [[buffer(2)]],      \
        device const float* bb [[buffer(3)]], device const float* A_log [[buffer(4)]],                                    \
        device const float* dt_bias [[buffer(5)]], device float* S [[buffer(6)]], device float* y [[buffer(7)]],          \
        device float* k0 [[buffer(8)]], device float* k1 [[buffer(9)]], device float* k2 [[buffer(10)]],                  \
        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]
template <int DK, int SPLIT>
kernel void k_q4_gdn_step(GDN_STEP_ARGS) {
    constexpr int DJ = DK / SPLIT;
    const int hv = (int) tg.x, p = (int) tid % SPLIT, i = (int) tg.y * 32 + (int) tid / SPLIT, kh = hv / a.rep;
    threadgroup float q[DK], k[DK];
    float s[DJ];
    device float* St = S + (ulong) hv * DK * a.dv + (ulong) p * DJ * a.dv + i;
    for (int j = 0; j < DJ; ++j) s[j] = St[(ulong) j * a.dv];
    const float A = exp(A_log[hv]), dtb = dt_bias[hv];
    for (int t = 0; t < a.T; ++t) {
        device const float* c = cv + (ulong) t * a.cd;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous token's reads of q and k are done
        for (int j = (int) tid; j < DK; j += 32 * SPLIT) { q[j] = c[kh * DK + j]; k[j] = c[a.kd + kh * DK + j]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qq = 0, kk = 0;
        for (int j = 0; j < DK; ++j) { qq += q[j] * q[j]; kk += k[j] * k[j]; }
        const float sq = rsqrt(qq + 1e-6f) * rsqrt((float) DK), sk = rsqrt(kk + 1e-6f);   // L2 norms; q also dk^-0.5
        const float ap = aa[(ulong) t * a.hv + hv] + dtb, sp = ap > 20.0f ? ap : log(1.0f + exp(ap));
        const float g = exp(-A * sp), beta = sigm(bb[(ulong) t * a.hv + hv]);
        threadgroup const float* kp = k + p * DJ;
        threadgroup const float* qp = q + p * DJ;
        float kv = 0;
        for (int j = 0; j < DJ; ++j) kv += s[j] * kp[j];
        for (ushort o = 1; o < SPLIT; o <<= 1) kv += simd_shuffle_xor(kv, o);
        const float delta = (c[2 * a.kd + hv * a.dv + i] - g * kv * sk) * beta;
        float yq = 0;
        for (int j = 0; j < DJ; ++j) { s[j] = g * s[j] + delta * kp[j] * sk; yq += s[j] * qp[j]; }
        for (ushort o = 1; o < SPLIT; o <<= 1) yq += simd_shuffle_xor(yq, o);
        if (p == 0) y[(ulong) t * a.vd + (ulong) hv * a.dv + i] = yq * sq;
        if (a.keep && t + 1 < a.T) {
            device float* ks = (t == 0 ? k0 : t == 1 ? k1 : k2) + (St - S);
            for (int j = 0; j < DJ; ++j) ks[(ulong) j * a.dv] = s[j];
        }
    }
    for (int j = 0; j < DJ; ++j) St[(ulong) j * a.dv] = s[j];
}
template [[host_name("k_q4_gdn_step_32")]] kernel void k_q4_gdn_step<32, 1>(GDN_STEP_ARGS);
template [[host_name("k_q4_gdn_step_64")]] kernel void k_q4_gdn_step<64, 1>(GDN_STEP_ARGS);
template [[host_name("k_q4_gdn_step_128")]] kernel void k_q4_gdn_step<128, GDN_SPLIT128>(GDN_STEP_ARGS);

// Threadgroup (head, token), thread = value row (dv threads): RMSNorm over the head (plain weight), gated by sigmoid(z)
kernel void k_q4_gdn_norm(constant Q4GNormArgs& a [[buffer(0)]], device const float* norm_w [[buffer(1)]], device const float* z [[buffer(2)]],
                          device float* y [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const ulong o = ((ulong) tg.y * a.hv + tg.x) * a.dv + tid;
    const float v = y[o], r = rsqrt(tg_sum(v * v, red, lane, sg, (uint) a.dv / 32) / a.dv + a.eps);
    y[o] = v * r * norm_w[tid] * sigm(z[o]);
}

// ---- RMSNorm (1 + w) and the hyper-connection pointwise steps --------------------------------------------------------

// Threadgroup = one group of g values of one token (256 threads)
kernel void k_q4_norm1p(constant Q4NormArgs& a [[buffer(0)]], device const float* x [[buffer(1)]], device const float* w [[buffer(2)]],
                        device float* y [[buffer(3)]], uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    // parts > 1 (not in place): parts threadgroups a group, each takes the whole group's sum (the same order, so the
    // same r) and writes its slice: more threadgroups at one row (hyper-connections: 4 groups)
    const int parts = max(a.parts, 1), gb = (int) b / parts, part = (int) b % parts;
    const int groups = a.len / a.g, t = gb / groups, o = (gb % groups) * a.g;
    device const float* xi = x + (ulong) t * a.len + o;
    float ss = 0;
    for (int k = (int) tid; k < a.g; k += 256) ss += xi[k] * xi[k];
    const float r = rsqrt(tg_sum(ss, red, lane, sg, 8) / a.g + a.eps);
    const int chunk = (a.g + parts - 1) / parts, k1 = min(a.g, (part + 1) * chunk);
    for (int k = part * chunk + (int) tid; k < k1; k += 256) y[(ulong) t * a.len + o + k] = xi[k] * r * (1.0f + w[o + k]);
}

kernel void k_q4_hc_act(constant Q4HcArgs& a [[buffer(0)]], device float* lo [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) { const float x = lo[i] * a.inv_hc; lo[i] = x * sigm(x); }
}
kernel void k_q4_hc_gate(constant Q4HcArgs& a [[buffer(0)]], device float* inj [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < a.n) inj[i] = 2.0f * sigm(inj[i] * a.inv_hc);
}
// grid (d, T): mixed[t][i] = mean_s sigmoid(up[t][s][i]) normed[t][s][i]
kernel void k_q4_hc_mix(constant Q4HcArgs& a [[buffer(0)]], device const float* normed [[buffer(1)]], device const float* up [[buffer(2)]],
                        device float* mixed [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const int i = (int) g.x, t = (int) g.y;
    if (i >= a.d) return;
    const ulong o = (ulong) t * a.hc * a.d + i;
    float m = 0;
    for (int s = 0; s < a.hc; ++s) m += sigm(up[o + (ulong) s * a.d]) * normed[o + (ulong) s * a.d];
    mixed[(ulong) t * a.d + i] = m / a.hc;
}
// grid (d, T): H[t][s][i] += branch[t][i] inj[t][s]
kernel void k_q4_hc_add(constant Q4HcArgs& a [[buffer(0)]], device float* H [[buffer(1)]], device const float* branch [[buffer(2)]],
                        device const float* inj [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const int i = (int) g.x, t = (int) g.y;
    if (i >= a.d) return;
    const float b = branch[(ulong) t * a.d + i];
    for (int s = 0; s < a.hc; ++s) H[((ulong) t * a.hc + s) * a.d + i] += b * inj[t * a.hc + s];
}

// grid (d, T): the embedding x[t] copied into each of the hc streams of H[t] (Q4HcArgs hc, d)
kernel void k_q4_tile(constant Q4HcArgs& a [[buffer(0)]], device const float* x [[buffer(1)]], device float* H [[buffer(2)]],
                      uint2 g [[thread_position_in_grid]]) {
    const int i = (int) g.x, t = (int) g.y;
    if (i >= a.d) return;
    const float v = x[(ulong) t * a.d + i];
    for (int s = 0; s < a.hc; ++s) H[((ulong) t * a.hc + s) * a.d + i] = v;
}
// grid (w, T): dst[(pos0 + t) * w + i] = src[t * ss + i] (the K, V and raw indexer key caches)
kernel void k_q4_store(constant Q4StoreArgs& a [[buffer(0)]], device float* dst [[buffer(1)]], device const float* src [[buffer(2)]],
                       uint2 g [[thread_position_in_grid]]) {
    const int i = (int) g.x, t = (int) g.y;
    if (i < a.w) dst[(ulong) (a.pos0 + t) * a.w + i] = src[(ulong) t * a.ss + i];
}
// grid (d, T): out[t] = sigmoid(sg[t]) sh[t] + sum_j wt[t][j] D[t][j] (the routed experts' rows, token-major pairs)
kernel void k_q4_moe_combine(constant Q4CombineArgs& a [[buffer(0)]], device const float* D [[buffer(1)]], device const float* wt [[buffer(2)]],
                             device const float* sh [[buffer(3)]], device const float* sg [[buffer(4)]], device float* out [[buffer(5)]],
                             uint2 g [[thread_position_in_grid]]) {
    const int i = (int) g.x, t = (int) g.y;
    if (i >= a.d) return;
    float s = sigm(sg[t]) * sh[(ulong) t * a.d + i];
    for (int j = 0; j < a.k; ++j) s += wt[(ulong) t * a.k + j] * D[((ulong) t * a.k + j) * a.d + i];
    out[(ulong) t * a.d + i] = s;
}

// ---- head norm + RoPE, the router ----------------------------------------------------------------------------------

// Threadgroup (head, row), 128 threads: RMSNorm (1 + w) of the head, then RoPE of its first rot values at position
// pos0 + row * pstep.  inv[rot / 2]: the inverse frequencies, rounded to f32 by the host as the C ops do; f32 angles.
kernel void k_q4_head_norm_rope(constant Q4RopeArgs& a [[buffer(0)]], device float* x [[buffer(1)]], device const float* w [[buffer(2)]],
                                device const float* inv [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32], v[1024];
    device float* xh = x + (ulong) tg.y * a.rs + (ulong) tg.x * a.hs;
    float ss = 0;
    for (int i = (int) tid; i < a.hd; i += 128) ss += xh[i] * xh[i];
    const float r = rsqrt(tg_sum(ss, red, lane, sg, 4) / a.hd + a.eps);
    for (int i = (int) tid; i < a.hd; i += 128) v[i] = xh[i] * r * (1.0f + w[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int pos = a.pos0 + (int) tg.y * a.pstep, h2 = a.rot / 2;
    for (int i = (int) tid; i < a.hd; i += 128) {
        float o = v[i];
        if (i < a.rot) {
            const float ang = (float) pos * inv[i % h2], cs = cos(ang), sn = sin(ang);
            o = i < h2 ? v[i] * cs - v[i + h2] * sn : v[i] * cs + v[i - h2] * sn;
        }
        xh[i] = o;
    }
}

// Threadgroup per row, 256 threads (E <= 1024: up to 4 logits a thread, in registers), the top k (ties to the higher
// expert) in two phases without a barrier a round: each simdgroup takes its own top k by simdgroup reductions, then
// simdgroup 0 takes the top k of their 8 k candidates. The order is total, so the winners are the round-by-round
// arg max's. Weights exp(l - max) renormalised over the k, summed in round order (the max is round 0's winner).
#define ROUTE_TG 256
static inline bool route_better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia > ib); }
kernel void k_q4_route(constant Q4RouteArgs& a [[buffer(0)]], device const float* logits [[buffer(1)]], device int* ind [[buffer(2)]],
                       device float* wt [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float cv[ROUTE_TG / 32][32];   // each simdgroup's top k, in order
    threadgroup int ci[ROUTE_TG / 32][32];
    device const float* l = logits + (ulong) t * a.E;
    float v[4];
    int nv = 0;
    for (int e = (int) tid; e < a.E && nv < 4; e += ROUTE_TG) v[nv++] = l[e];
    uint taken = 0;   // bit i: logit tid + ROUTE_TG * i is chosen
    for (int r = 0; r < a.k; ++r) {   // phase 1: this simdgroup's top k
        float bv = -INFINITY;
        int bi = -1;
        for (int i = 0; i < nv; ++i) {
            const int e = (int) tid + ROUTE_TG * i;
            if (!((taken >> i) & 1u) && route_better(v[i], e, bv, bi)) { bv = v[i]; bi = e; }
        }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (route_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (bi >= 0 && bi % ROUTE_TG == (int) tid) taken |= 1u << (bi / ROUTE_TG);
        if (lane == 0) { cv[sg][r] = bv; ci[sg][r] = bi; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0) return;
    // phase 2, simdgroup 0: lane s < 8 walks simdgroup s's list (in order), the best head wins a round
    int head = 0;
    const int ns = ROUTE_TG / 32;
    float mx = 0, top = 0, wr = 0;
    int ir = 0;
    for (int r = 0; r < a.k; ++r) {
        float bv = (int) lane < ns && head < a.k ? cv[lane][head] : -INFINITY;
        int bi = (int) lane < ns && head < a.k ? ci[lane][head] : -1;
        const float mv = bv;
        const int mi = bi;
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (route_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (mi == bi && mv == bv && (int) lane < ns) ++head;   // this lane's head won
        if (r == 0) mx = bv;
        const float w = exp(bv - mx);
        top += w;   // the same in every lane
        if ((int) lane == r) { ir = bi; wr = w; }
    }
    if ((int) lane < a.k) { ind[(ulong) t * a.k + lane] = ir; wt[(ulong) t * a.k + lane] = wr / top; }
}

// ---- PLE ---------------------------------------------------------------------------------------------------------------

// Threadgroup (stream, token), 256 threads: gated = sigmoid(signed sqrt(keys . qs / sqrt(d))) vals
kernel void k_q4_ple_gate(constant Q4PleGateArgs& a [[buffer(0)]], device const float* keys [[buffer(1)]], device const float* qs [[buffer(2)]],
                          device const float* vals [[buffer(3)]], device float* gated [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    const ulong o = ((ulong) tg.y * a.hc + tg.x) * a.d;
    float dot = 0;
    for (int i = (int) tid; i < a.d; i += 256) dot += keys[o + i] * qs[o + i];
    const float g = tg_sum(dot, red, lane, sg, 8) / sqrt((float) a.d);
    const float gs = sigm(g == 0.0f ? 0.0f : copysign(sqrt(max(fabs(g), 1e-6f)), g));   // a signed square root
    for (int i = (int) tid; i < a.d; i += 256) gated[o + i] = gs * vals[(ulong) tg.y * a.d + i];
}

// One thread per channel, tokens in order: H += gated + silu(causal conv, dilation dil); the last (K - 1) * dil inputs in
// registers. keep: also the history after each row t < T - 1 at k0, k1, k2 (t)
kernel void k_q4_ple_conv(constant Q4PleConvArgs& a [[buffer(0)]], device const float* w [[buffer(1)]], device const float* cin [[buffer(2)]],
                          device const float* gated [[buffer(3)]], device float* hist [[buffer(4)]], device float* H [[buffer(5)]],
                          device float* k0 [[buffer(6)]], device float* k1 [[buffer(7)]], device float* k2 [[buffer(8)]],
                          uint gid [[thread_position_in_grid]]) {
    const int ch = (int) gid, HD = a.HD, K = a.K;
    if (ch >= HD) return;
    const int L = (K - 1) * a.dil;
    float st[32];
    for (int j = 0; j < L; ++j) st[j] = hist[(ulong) j * HD + ch];
    for (int t = 0; t < a.T; ++t) {
        const float x = cin[(ulong) t * HD + ch];
        float acc = w[(ulong) ch * K + K - 1] * x;
        for (int k = 0; k < K - 1; ++k) acc += w[(ulong) ch * K + k] * st[L - (K - 1 - k) * a.dil];
        H[(ulong) t * HD + ch] += gated[(ulong) t * HD + ch] + acc * sigm(acc);
        for (int j = 0; j + 1 < L; ++j) st[j] = st[j + 1];
        if (L > 0) st[L - 1] = x;
        if (a.keep && t + 1 < a.T) {
            device float* ks = t == 0 ? k0 : t == 1 ? k1 : k2;
            for (int j = 0; j < L; ++j) ks[(ulong) j * HD + ch] = st[j];
        }
    }
    for (int j = 0; j < L; ++j) hist[(ulong) j * HD + ch] = st[j];
}

// ---- QSA ---------------------------------------------------------------------------------------------------------------

// Threadgroup = block j: blk[j] = the mean of raw keys j * R .. j * R + R - 1 (k_q4_head_norm_rope then norms and rotates it)
kernel void k_q4_qsa_mean(constant Q4QsaMeanArgs& a [[buffer(0)]], device const float* raw [[buffer(1)]], device float* blk [[buffer(2)]],
                          uint j [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    for (int i = (int) tid; i < a.D; i += 128) {
        float m = 0;
        for (int u = 0; u < a.R; ++u) m += raw[((ulong) j * a.R + u) * a.D + i];
        blk[(ulong) j * a.D + i] = m / a.R;
    }
}

// Simdgroup per (row, block), 8 blocks a threadgroup: sum over heads of relu(q_h . blk_j)
kernel void k_q4_qsa_scores(constant Q4QsaArgs& a [[buffer(0)]], device const float* iq [[buffer(1)]], device const float* blk [[buffer(2)]],
                            device const int* done [[buffer(3)]], device float* sc [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    const int t = (int) tg.y, j = (int) tg.x * 8 + (int) sg;
    if (j >= done[t]) return;
    float s = 0;
    for (int h = 0; h < a.nh; ++h) {
        float d = 0;
        for (int i = (int) lane; i < a.D; i += 32) d += iq[(ulong) t * a.rs + h * a.D + i] * blk[(ulong) j * a.D + i];
        s += max(simd_sum(d), 0.0f);
    }
    if (lane == 0) sc[(ulong) t * a.maxdone + j] = s;
}
// Threadgroup per row (256 threads): the k-th largest score by a radix select on its bits (scores are >= 0), then every
// score above it in block order (attention sums the keys in this order, so it must not depend on timing: a
// threadgroup-wide prefix of the matches, as CUDA's ballot prefix) and, of the equal ones, those of the highest blocks.
// Rows with done <= k are left as they are.
kernel void k_q4_qsa_topk(constant Q4QsaArgs& a [[buffer(0)]], device const float* sc [[buffer(1)]], device const int* done [[buffer(2)]],
                          device int* sel [[buffer(3)]], uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[32];
    threadgroup uint wcnt[8];
    const int n = done[t];
    if (n <= a.k) return;
    device const uint* u = (device const uint*) (sc + (ulong) t * a.maxdone);
    uint p = 0;
    for (int b = 31; b >= 0; --b) {
        const uint cand = p | (1u << b);
        float cnt = 0;
        for (int j = (int) tid; j < n; j += 256) cnt += u[j] >= cand ? 1.0f : 0.0f;
        if (tg_sum(cnt, red, lane, sg, 8) >= (float) a.k) p = cand;
    }
    device int* out = sel + (ulong) t * a.k;
    uint nout = 0;   // the same in every thread
    for (int j0 = 0; j0 < n; j0 += 256) {
        const int j = j0 + (int) tid;
        const uint f = j < n && u[j] > p ? 1u : 0u;
        const uint before_lane = simd_prefix_exclusive_sum(f), in_sg = simd_sum(f);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous chunk's reads of wcnt are done
        if (lane == 0) wcnt[sg] = in_sg;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = nout, total = 0;
        for (uint x = 0; x < 8; ++x) { before += x < sg ? wcnt[x] : 0u; total += wcnt[x]; }
        if (f) out[before + before_lane] = j;
        nout += total;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0)
        for (int j = n - 1, m = (int) nout; j >= 0 && m < a.k; --j)
            if (u[j] == p) out[m++] = j;
}

// ---- attention ---------------------------------------------------------------------------------------------------------

// Threadgroup (head, row), 8 simdgroups: each takes every 8th visible key with an online softmax (lanes hold hd / 32
// dims of q and of the output), then the simdgroups merge; the output is gated by sigmoid of the query's gate half
kernel void k_q4_attend(constant Q4AttArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const float* K [[buffer(2)]],
                        device const float* V [[buffer(3)]], device const int* sel [[buffer(4)]], device const int* nsel [[buffer(5)]],
                        device float* o [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[8], sl[8], sacc[8][256];
    const int h = (int) tg.x, t = (int) tg.y, w = (int) sg, dl = a.hd / 32, g = h / (a.nh / a.nkv);
    const int pos = a.pos0 + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    if (a.smin && n >= a.smin) return;   // k_q4_attend_g's row
    device const float* qh = q + ((ulong) t * a.nh + h) * 2 * a.hd;
    float qv[8], acc[8], m = -INFINITY, l = 0;
    for (int i = 0; i < dl; ++i) { qv[i] = qh[lane + 32 * i]; acc[i] = 0; }
    for (int k = w; k < n; k += 8) {
        const int key = k < ns * a.R ? sel[(ulong) t * a.topk + k / a.R] * a.R + k % a.R : tail + (k - ns * a.R);
        device const float* kk = K + ((ulong) key * a.nkv + g) * a.hd;
        float d = 0;
        for (int i = 0; i < dl; ++i) d += qv[i] * kk[lane + 32 * i];
        d = simd_sum(d) * a.scale;
        const float mn = max(m, d), cf = exp(m - mn), p = exp(d - mn);
        device const float* vv = V + ((ulong) key * a.nkv + g) * a.hd;
        for (int i = 0; i < dl; ++i) acc[i] = acc[i] * cf + p * vv[lane + 32 * i];
        l = l * cf + p;
        m = mn;
    }
    if (lane == 0) { sm[w] = m; sl[w] = l; }
    for (int i = 0; i < dl; ++i) sacc[w][lane + 32 * i] = acc[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float M = -INFINITY;
    for (int x = 0; x < 8; ++x) M = max(M, sm[x]);
    float L = 0;
    for (int x = 0; x < 8; ++x) L += sm[x] == -INFINITY ? 0.0f : sl[x] * exp(sm[x] - M);
    for (int i = (int) tid; i < a.hd; i += 256) {
        float v = 0;
        for (int x = 0; x < 8; ++x) if (sm[x] != -INFINITY) v += sacc[x][i] * exp(sm[x] - M);
        o[((ulong) t * a.nh + h) * a.hd + i] = v / L * sigm(qh[a.hd + i]);
    }
}

// Prompt rows: threadgroup (KV group g, row t), 4 simdgroups: 2 head tiles (heads 8 ht .. 8 ht + 7 of the group's G <= 16;
// rows past G are padding: q needs 16 heads of slack after the last row) x ATG_NQ parts of the head dims.  The keys in tiles
// of 8, staged in threadgroup memory (the QSA blocks of R keys need not be adjacent), so one load serves all the group's
// heads.  Each half takes its dims' part of the scores Q K^T (f32 simdgroup matrices) and 16 of the output's dim tiles;
// the online softmax per head row (lanes 0..7 of half 0) sees the sum of both parts.  O = diag(rescale) O + P V.  The
// same result as k_q4_attend up to the order of the sums.  hd a multiple of 8 ATG_NQ, <= 256.
#define ATG_K 8
kernel void k_q4_attend_g(constant Q4AttArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const float* K [[buffer(2)]],
                          device const float* V [[buffer(3)]], device const int* sel [[buffer(4)]], device const int* nsel [[buffer(5)]],
                          device float* o [[buffer(6)]], device float* part [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sK[ATG_K * 256], sV[ATG_K * 256];
    threadgroup float sS[2][ATG_NQ][64], sP[2][64], sD[2][64], sL[2][8], sM[2][8];
    threadgroup int sR[2];
    const int g = (int) tg.x, t = (int) tg.y, G = a.nh / a.nkv, hd = a.hd, kvw = a.nkv * hd;
    const int ht = (int) sg / ATG_NQ, dh = (int) sg % ATG_NQ, nj = hd / (8 * ATG_NQ), j0 = dh * nj;   // dim tiles j0 .. j0 + nj - 1
    const int pos = a.pos0 + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
    // split tg.z of nsp: keys [kb, ke), whole tiles of ATG_K (decode rows: the splits' partials go to part, then
    // k_q4_attend_comb merges them; one split writes the output itself)
    const int nsp = a.smin ? min(a.smax > 0 ? a.smax : ATG_MAX_SPLIT, (n + a.skeys - 1) / a.skeys) : a.nsplit > 1 ? a.nsplit : 1;
    if ((a.smin && n < a.smin) || (int) tg.z >= nsp) return;   // k_q4_attend's row, or a split past this row's
    const int per = ((n + nsp - 1) / nsp + ATG_K - 1) / ATG_K * ATG_K;
    const int kb = (int) tg.z * per, ke = min(n, kb + per);
    const bool active = ht * 8 < G;
    device const float* qb = q + ((ulong) t * a.nh + (ulong) g * G + ht * 8) * 2 * hd;   // 8 heads, stride 2 hd
    simdgroup_float8x8 O[32 / ATG_NQ];   // the loops run to a constant so that O[] stays in registers
    for (int j = 0; j < 32 / ATG_NQ; ++j) O[j] = simdgroup_float8x8(0);
    float m = -INFINITY, l = 0;   // half 0, lanes 0..7: head row `lane` of the tile
    for (int k0 = kb; k0 < ke; k0 += ATG_K) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous tile's reads are done
        for (int x = (int) tid; x < ATG_K * hd; x += 64 * ATG_NQ) {
            const int r = x / hd, i = x % hd, k = k0 + r;
            float kv = 0.0f, vv = 0.0f;
            if (k < ke) {
                const int key = k < ns * a.R ? sel[(ulong) t * a.topk + k / a.R] * a.R + k % a.R : tail + (k - ns * a.R);
                kv = K[(ulong) key * kvw + (ulong) g * hd + i];
                vv = V[(ulong) key * kvw + (ulong) g * hd + i];
            }
            sK[x] = kv;
            sV[x] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            simdgroup_float8x8 S = simdgroup_float8x8(0);
            for (int j = 0; j < 32 / ATG_NQ; ++j) {
                if (j >= nj) break;
                simdgroup_float8x8 qm, km;
                simdgroup_load(qm, qb + (j0 + j) * 8, (ulong) 2 * hd);                    // 8 heads x 8 dims
                simdgroup_load(km, sK + (j0 + j) * 8, (ulong) hd, ulong2(0, 0), true);    // transposed: 8 dims x 8 keys
                simdgroup_multiply_accumulate(S, qm, km, S);
            }
            simdgroup_store(S, sS[ht][dh], 8);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && dh == 0) {
            float cf = 1.0f;
            if (lane < 8) {
                float sc[8], mx = m;
                for (int c = 0; c < 8; ++c) {
                    float sum = 0.0f;
                    for (int u = 0; u < ATG_NQ; ++u) sum += sS[ht][u][lane * 8 + c];
                    sc[c] = k0 + c < ke ? sum * a.scale : -INFINITY;
                    mx = max(mx, sc[c]);
                }
                cf = exp(m - mx);   // 0 on the first tile (m = -inf; key k0 is visible, so mx is finite)
                float ps = 0.0f;
                for (int c = 0; c < 8; ++c) { const float p = exp(sc[c] - mx); sP[ht][lane * 8 + c] = p; ps += p; }
                for (int c = 0; c < 8; ++c) sD[ht][lane * 8 + c] = c == (int) lane ? cf : 0.0f;
                l = l * cf + ps;
                m = mx;
            }
            const bool rescale = simd_any(lane < 8 && cf != 1.0f);
            if (lane == 0) sR[ht] = rescale ? 1 : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            const bool rescale = sR[ht] != 0;
            simdgroup_float8x8 P, D;
            simdgroup_load(P, sP[ht], 8);
            if (rescale) simdgroup_load(D, sD[ht], 8);
            for (int j = 0; j < 32 / ATG_NQ; ++j) {
                if (j >= nj) break;
                simdgroup_float8x8 vm;
                simdgroup_load(vm, sV + (j0 + j) * 8, (ulong) hd);   // 8 keys x 8 dims
                if (rescale) simdgroup_multiply(O[j], D, O[j]);
                simdgroup_multiply_accumulate(O[j], P, vm, O[j]);
            }
        }
    }
    // the output / l, gated by sigmoid of the query's gate half: through threadgroup memory, only the group's heads
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* sO = ht == 0 ? sK : sV;
    if (active) {
        for (int j = 0; j < 32 / ATG_NQ; ++j) if (j < nj) simdgroup_store(O[j], sO + (j0 + j) * 8, (ulong) hd);
        if (dh == 0 && lane < 8) { sL[ht][lane] = l; sM[ht][lane] = m; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int h = 0; active && h < 8 && ht * 8 + h < G; ++h) {
        const ulong head = (ulong) t * a.nh + (ulong) g * G + ht * 8 + h;
        if (nsp == 1) {
            for (int i = j0 * 8 + (int) lane; i < (j0 + nj) * 8; i += 32)
                o[head * hd + i] = sO[h * hd + i] / sL[ht][h] * sigm(qb[(ulong) h * 2 * hd + hd + i]);
        } else {   // this split's partial: O unnormalised, then its max and sum (m = -inf: no keys)
            device float* pb = part + (head * a.nsplit + tg.z) * (hd + 2);
            for (int i = j0 * 8 + (int) lane; i < (j0 + nj) * 8; i += 32) pb[i] = sO[h * hd + i];
            if (dh == 0 && lane == 0) { pb[hd] = sM[ht][h]; pb[hd + 1] = sL[ht][h]; }
        }
    }
}

// k_q4_attend_g's partials of decode rows merged (its nsplit, or with smin each row's own count): threadgroup (head h,
// row t), 256 threads; the output / L, gated by sigmoid of the query's gate half
kernel void k_q4_attend_comb(constant Q4AttArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const float* part [[buffer(2)]],
                             device float* o [[buffer(3)]], device const int* nsel [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const int h = (int) tg.x, t = (int) tg.y, hd = a.hd, st = hd + 2;
    int nsp = a.nsplit;
    if (a.smin) {   // this row's splits; none (k_q4_attend's row) or one (written by k_q4_attend_g): nothing to merge
        const int pos = a.pos0 + t, ns = nsel[t], tail = ns ? (pos + 1) / a.R * a.R : 0, n = ns * a.R + pos + 1 - tail;
        nsp = n < a.smin ? 0 : min(a.smax > 0 ? a.smax : ATG_MAX_SPLIT, (n + a.skeys - 1) / a.skeys);
        if (nsp < 2) return;
    }
    const ulong head = (ulong) t * a.nh + h;
    device const float* pb = part + head * a.nsplit * st;
    float M = -INFINITY;
    for (int s = 0; s < nsp; ++s) M = max(M, pb[s * st + hd]);
    float L = 0.0f;
    for (int s = 0; s < nsp; ++s) { const float ms = pb[s * st + hd]; if (ms != -INFINITY) L += pb[s * st + hd + 1] * exp(ms - M); }
    device const float* qh = q + head * 2 * hd;
    for (int i = (int) tid; i < hd; i += 256) {
        float v = 0.0f;
        for (int s = 0; s < nsp; ++s) { const float ms = pb[s * st + hd]; if (ms != -INFINITY) v += pb[s * st + i] * exp(ms - M); }
        o[head * hd + i] = v / L * sigm(qh[hd + i]);
    }
}

#endif   // __METAL_VERSION__
#endif   // NSLM_KERNELS_QWEN4_METAL
