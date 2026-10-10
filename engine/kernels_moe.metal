// engine/kernels_moe.metal - Metal kernels of the MoVA engine (engine/mova_gpu.m); host code #includes it for the
// argument structs and constants.  Activations are f32 buffers of BF16-rounded values, rounded wherever the MLX
// reference implementation produces a BF16 tensor.  Every kernel has a C reference in tests/test_mova_kernels.c.
#ifndef NSLM_KERNELS_MOE_METAL
#define NSLM_KERNELS_MOE_METAL
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#endif

// Weight formats (per tensor): BF16; MLX affine Q8 / Q4, group 64, dequantized to BF16 as mx.dequantize does;
// SEED4 / SEED4P4: LFSR seeds, expanded from the per-seed stream table.
enum { MF_BF16 = 0, MF_SEED4 = 1, MF_Q8 = 2, MF_Q4 = 3, MF_SEED4P4 = 4, MF_SEED6P8 = 5 };   // = NS_* (nslm/model_st.h)
#define MV_MAXT 8          // dense matvec: tokens per weight-row load
#define MV_ROWS 8          // simdgroups per matvec threadgroup
#ifndef MV_NR_SEED
#define MV_NR_SEED 1       // rows per simdgroup (sharing each x load), SEED4
#endif
#ifndef MV_NR_AFF
#define MV_NR_AFF 1        // rows per simdgroup, BF16 / Q8 / Q4
#endif
#ifndef MV_SEED_BPL
#define MV_SEED_BPL 2      // seed matvec (k_mv_gu): adjacent blocks per lane per step (2: 4-byte stream loads)
#endif
#ifndef MV_SEED_U
#define MV_SEED_U 1        // seed matvec, T > 1: interleaved blocks per lane per step
#endif
#ifndef MV_Q4_BPL
#define MV_Q4_BPL 2        // Q4 matvec: consecutive 8-weight blocks per lane per step (lanes step by 32 * MV_Q4_BPL)
#endif
#define MV_NR(fmt) ((fmt) == 1 ? MV_NR_SEED : MV_NR_AFF)
#define MV_NR_MAX (MV_NR_SEED > MV_NR_AFF ? MV_NR_SEED : MV_NR_AFF)
#define MV_RPT(fmt) (MV_ROWS * MV_NR(fmt))   // rows per threadgroup
#define ATT_SG 4           // simdgroups per attention threadgroup (the 4 query heads of one KV head)
#define ATT_HD 128
#ifndef ATT_KU
#define ATT_KU 4           // decode attention: keys per step
#endif
#ifndef ATTF_RS
#define ATTF_RS 1          // prefill attention: 8-row blocks per threadgroup
#endif
#ifndef ATTF_BK
#define ATTF_BK 16         // prefill attention: keys per tile
#endif
#ifndef ATTF_DIAG_F32
#define ATTF_DIAG_F32 0    // diagnostics only (with ATTF_BF 0): q * scale and P unrounded, f32 fragments throughout
#endif
#ifndef ATTF_BF
#define ATTF_BF 1          // prefill attention: BF16 operand fragments (exact products, f32 sums); 0 = f32 fragments
#endif
#define ATTF_G 4           // prefill attention: query heads per KV head (MoVA's 32 / 8; checked at load)

typedef struct {
    int K, R;              // input width, rows per slice
    int P;                 // output vectors: dense = tokens (<= MV_MAXT), gather = (token, expert) pairs
    int xdiv;              // gather: pair p reads input row p / xdiv
    int xs, ys;            // input / output row strides (floats)
    int ebias0;            // unused (seed biases are per slice, in a buffer)
    int add;               // 1: y = bf16(y + bf16(acc)) (residual add), else y = bf16(acc)
} MvArgs;

typedef struct {
    int d, n, top_k;       // input width, experts, selected
    int hw;                // router partition width (d / 2)
    float scale;           // router_scaling_factor
} RouterArgs;

typedef struct {
    int n_head, n_kv, chunk, n_splits;   // chunk: target keys per split; each row splits its own 0..pos into
                                         // min(n_splits, ceil((pos + 1) / chunk)) parts (independent of T)
    float scale;
} AttnArgs;

typedef struct { int pos, kv0; int pad[2]; } RowInfo;   // kv0: the KV cache row of the sequence slot's position 0
// MLA (a TransMLA conversion, nslm/mova_cfg.h): latent attention over one shared latent c (rank r) and one 128-dim RoPE
// key per position.  chunk / n_splits as AttnArgs (n_splits 1: one pass that writes the output, the prefill path).
#define MLA_KU 4           // latent attention: keys per step (staged in threadgroup memory for all heads)
#define MLA_MAXL 32        // latent dims per lane (r <= 1024)
#define MLPF_R 4           // Metal k_mla_prefill: simdgroups of 8 rows each
#define MLPF_BK 16         // keys per tile
#define MLA_FP4_LEAD 256   // MLA FP4 caches: the latent's leading dims kept in FP8 (a multiple of 32)
#define MLAF_Q 16          // Metal k_mla_attn: query heads per threadgroup
#define MLAF_K 64          // keys per tile
#define MLAF_SG 16         // simdgroups: 2 x MLAF_K / 8 (a key block, both head blocks, half the dims), one 8-dim output column
#define MLAF_DC 128        // dims per staged chunk
#define MLAF_KLD (MLAF_DC + 2)               // staged keys [key][dim] row stride (BF16; padded against bank conflicts)
#define MLAF_QLD (MLAF_Q + 2)                // staged queries [dim][head] row stride
#define MLAF_TH (32 * MLAF_SG / MLAF_Q)      // softmax: threads per head (<= 32: one simdgroup)
#define MLAF_OC (MLA_MAXL * 32 / MLAF_DC)    // latent chunks (r <= 1024)
typedef struct {
    int n_head, r, chunk, n_splits;
    float scale;
    int pad[3];
} MlaArgs;
typedef struct {           // MLA prompt attention, expanded (k_mla_prefill)
    int n_head, kb0, kb1;  // keys [kb0, kb1) of this pass (the expanded keys and values hold rows kb0 ..)
    int first, last;       // the pass starts the softmax state / finishes it into the gated output
    float scale;
    int T, pad;
} MlpArgs;
typedef struct {           // per-head maps: y[t][h][o] = W_h[o] . x[t * xs + h * hs ..]
    int H, O, I, xs, hs, gate;
    int pad[2];
} HmvArgs;

#define MM_BM 32           // GEMM: weight rows per threadgroup
#define MM_BN 32           // GEMM: tokens (or pairs) per threadgroup
#define MM_BK 32
typedef struct {
    int K, R;              // input width, rows per slice
    int T;                 // dense: tokens; grouped: unused
    int xs, ys;            // input / output row strides
    int xdiv;              // grouped: pair p reads input row p / xdiv
    int add;               // dense: 1 residual add; 2 (Metal) the gate: y = bf16(bf16(acc) * bf16(softplus_ln2(g))), g
                           // at buffer 10 in y's layout
    int pad;
} MmArgs;
typedef struct { int slice, start, count, pad; } MmTile;   // grouped GEMM: one expert's run of pairs in perm[]

#ifdef __METAL_VERSION__

constant short FC_FMT [[function_constant(0)]];
constant short FC_T [[function_constant(1)]];
// MLA quantized caches: the RoPE key FP8; the latent FP8 (KV_FP8) or its first MLA_FP4_LEAD dims FP8, the rest FP4
#define KLH ATT_HD
#define LLH(r) (FC_FMT == KV_FP8 ? (r) : min(MLA_FP4_LEAD, (r)))
enum { KV_BF16 = 0, KV_Q8 = 1, KV_FP8 = 2, KV_FP4 = 3 };   // = ENG_KV_* (engine/engine_api.h); FP8 / FP4: nslm/kvq.h

static inline float bfr(float x) {   // round to BF16 (nearest even), as f32
    uint u = as_type<uint>(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}
static inline float bf(ushort h) { return as_type<float>((uint) h << 16); }
static inline ushort tobf(float x) {
    uint u = as_type<uint>(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (ushort) (u >> 16);
}
// ---- the low-bit KV cache codecs (nslm/kvq.h, the spec: same codes bit for bit) ----
static inline float e4m3_dec(uint c) {
    const uint e = (c >> 3) & 15u, m = c & 7u;
    const float v = e ? as_type<float>(((e + 120u) << 23) | (m << 20)) : (float) m * 0x1p-9f;
    return c & 0x80u ? -v : v;
}
static inline float e8m0_dec(uint s) { return as_type<float>(s << 23); }   // 2^(s - 127), s in 1 .. 254
constant float E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
static inline float e2m1_dec(uint n) { return E2M1[n & 15u]; }
static inline uint e4m3_enc_abs(float a) {
    if (!(a < 448.0f)) return 0x7Eu;
    if (a < 0x1p-6f) return (uint) rint(a * 512.0f);
    int e;
    const float f = frexp(a, e);
    int E = e - 1, m = (int) rint((2 * f - 1) * 8);
    if (m == 8) { ++E; m = 0; }
    return min((uint) (((E + 7) << 3) | m), 0x7Eu);
}
static inline uint e2m1_enc(float x, float s) {   // midpoints 0.25 0.75 1.25 1.75 2.5 3.5 5 (times s: exact); ties even
    const float a = fabs(x);
    uint k = (uint) (a > 0.25f * s) + (uint) (a > 0.75f * s) + (uint) (a > 1.25f * s) + (uint) (a > 1.75f * s) +
             (uint) (a > 2.5f * s) + (uint) (a > 3.5f * s) + (uint) (a > 5.0f * s);
    k += (uint) (a == 0.75f * s || a == 1.75f * s || a == 3.5f * s);   // a tie above an odd code: the even one
    return k && x < 0 ? k | 8u : k;
}
static inline uint fp8_scale(float amax) {
    if (!(amax > 0)) return 127u;
    int k;
    frexp(amax, k);
    const int e = clamp(amax <= ldexp(448.0f, k - 9) ? k - 9 : k - 8, -126, 127);
    return (uint) (e + 127);
}
static inline uint fp4_scale(float amax) {
    if (!(amax > 0)) return 0u;
    const int c0 = (int) e4m3_enc_abs(amax / 6);
    int best = -1;
    float bd = 0;
    for (int c = max(c0 - 1, 0); c <= min(c0 + 1, 0x7E); ++c) {
        const float d = fabs(6 * e4m3_dec((uint) c) - amax);
        if (best < 0 || d < bd || (d == bd && !(c & 1))) { best = c; bd = d; }
    }
    return (uint) best;
}
// Quantized cache rows (nslm/kvq.h): the first lh values FP8 (E8M0 scale per 32), the rest FP4 (E4M3 scale per 16):
// lh + (len - lh) / 2 code bytes a row, lh / 32 + (len - lh) / 16 scales.  fmt KV_BF16: plain BF16 rows.
static inline ulong kv_rowb(int lh, int len) { return (ulong) (lh + (len - lh) / 2); }
static inline ulong kv_srowb(int lh, int len) { return (ulong) (lh / 32 + (len - lh) / 16); }
// Value d of cache row `row` (len values)
static inline float kv_get(short fmt, int lh, device const uchar* C, device const uchar* S, ulong row, int len, int d) {
    if (fmt == KV_BF16) return bf(((device const ushort*) C)[row * len + d]);
    C += row * kv_rowb(lh, len);
    S += row * kv_srowb(lh, len);
    if (d < lh) return e4m3_dec(C[d]) * e8m0_dec(S[d / 32]);
    const int e = d - lh;
    return e2m1_dec(C[lh + e / 2] >> (4 * (e & 1))) * e4m3_dec(S[lh / 32 + e / 16]);
}
// Values d .. d + 7 (d a multiple of 8) of a cache row as BF16 pairs (exact: a code times its scale).
static inline uint4 kv_get8(short fmt, int lh, device const uchar* C, device const uchar* S, ulong row, int len, int d) {
    if (fmt == KV_BF16) return *(device const uint4*) (C + (row * len + d) * 2);
    C += row * kv_rowb(lh, len);
    S += row * kv_srowb(lh, len);
    float v[8];
    if (d < lh) {
        const uint2 w = *(device const uint2*) (C + d);
        const float s = e8m0_dec(S[d / 32]);
        for (int i = 0; i < 4; ++i) { v[i] = e4m3_dec(w.x >> (8 * i)) * s; v[4 + i] = e4m3_dec(w.y >> (8 * i)) * s; }
    } else {
        const int e = d - lh;
        const uint w = *(device const uint*) (C + lh + e / 2);
        const float s = e4m3_dec(S[lh / 32 + e / 16]);
        for (int i = 0; i < 8; ++i) v[i] = e2m1_dec(w >> (4 * i)) * s;
    }
    uint4 o;   // exact in BF16: the top halves
    for (int i = 0; i < 4; ++i) o[i] = as_type<uint>(v[2 * i]) >> 16 | (as_type<uint>(v[2 * i + 1]) & 0xFFFF0000u);
    return o;
}
static inline float silu_bf(float g) {   // nn.silu on a BF16 tensor: g * sigmoid(g), each op rounded to BF16
    const float s = bfr(1.0f / (1.0f + exp(-g)));
    return bfr(g * s);
}

// ---- embedding -------------------------------------------------------------------------------------------------------
// x[t][:] = row ids[t] of the table (BF16, Q8 dequantized to BF16, or SEED4P4 decoded to BF16: W seeds, S coefficient
// words, B the int32 bias, G the 32-bit stream table, EN exponent codes; tile_weights' arithmetic for one weight)
kernel void k_embed(constant int& d [[buffer(0)]], device const ushort* W [[buffer(1)]], device const uint* QW [[buffer(2)]],
                    device const ushort* S [[buffer(3)]], device const ushort* B [[buffer(4)]],
                    device const int* ids [[buffer(5)]], device float* x [[buffer(6)]], device const uint* G [[buffer(7)]],
                    device const uchar* EN [[buffer(8)]], uint2 g [[thread_position_in_grid]]) {
    const uint c = g.x, t = g.y;
    if ((int) c >= d) return;
    const ulong row = (ulong) ids[t];
    float v;
    if (FC_FMT == MF_SEED4P4) {
        const ulong k = row * (ulong) (d / 8) + c / 8;
        const uint s = W[k], gs = G[s], lo = s | (gs << 16), i = c % 8;
        const int ci = (int) S[k];
        const float q0 = float((ci << 28) >> 28), q1 = float((ci << 24) >> 28), q2 = float((ci << 20) >> 28), q3 = float((ci << 16) >> 28);
        float cs[4];
        for (uint p = 0; p < 4; ++p) {
            const uint kk = 4 * i + p + 1;
            cs[p] = as_type<float>(0x4B000000u | (kk <= 16 ? extract_bits(lo, kk, 16u) : extract_bits(gs, kk - 16, 16u))) - 8421376.0f;
        }
        const float sc = as_type<float>(uint(((device const int*) B)[0] + (int) ((EN[k >> 1] >> ((k & 1) * 4)) & 15) + 127) << 23) * (1.0f / 32767.0f);
        v = bfr(((((cs[0] * q0) + (cs[1] * q1)) + (cs[2] * q2)) + (cs[3] * q3)) * sc);
    } else if (FC_FMT == MF_SEED6P8) {
        const ulong k = row * (ulong) (d / 8) + c / 8;
        const uint s = W[k], gs = G[s], lo = s | (gs << 16), s2 = gs >> 16, g2 = G[s2], lo2 = s2 | (g2 << 16), i = c % 8;
        const uint cw = ((device const uint*) S)[k];
        float is = 0.0f;
        for (uint p = 0; p < 8; ++p) {
            const uint kk = 8 * i + p + 1;   // 1 .. 64
            const uint bits = kk <= 16 ? extract_bits(lo, kk, 16u) : kk <= 32 ? extract_bits(gs, kk - 16, 16u)
                            : kk <= 48 ? extract_bits(lo2, kk - 32, 16u) : extract_bits(g2, kk - 48, 16u);
            is = is + (as_type<float>(0x4B000000u | bits) - 8421376.0f) * float((int) (cw << (28 - 4 * p)) >> 28);
        }
        const float sc = as_type<float>(uint(((device const int*) B)[0] + (int) ((EN[k >> 1] >> ((k & 1) * 4)) & 15) + 127) << 23) * (1.0f / 32767.0f);
        v = bfr(is * sc);
    } else if (FC_FMT == MF_Q8) {
        const uint w = QW[(row * (ulong) d + c) / 4];
        const float q = (float) ((w >> (8 * (c % 4))) & 255u);
        const ulong gi = (row * (ulong) d + c) / 64;
        v = bfr(bf(S[gi]) * q + bf(B[gi]));
    } else v = bf(W[row * (ulong) d + c]);
    x[(ulong) t * d + c] = v;
}

// ---- grouped RMSNorm (2 groups): y = bf16(w * (x * rsqrt(mean_group(x^2) + eps))) -----------------------------------------
kernel void k_gnorm(constant int& d [[buffer(0)]], constant float& eps [[buffer(1)]], device const float* x [[buffer(2)]],
                    device const ushort* w [[buffer(3)]], device float* y [[buffer(4)]],
                    uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2][8];
    device const float* xr = x + (ulong) t * d;
    const int hw = d / 2;
    float s0 = 0, s1 = 0;
    for (int c = (int) tid; c < d; c += 256) {
        const float v = xr[c];
        if (c < hw) s0 += v * v; else s1 += v * v;
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (lane == 0) { part[0][sg] = s0; part[1][sg] = s1; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a = 0, b = 0;
    for (int i = 0; i < 8; ++i) { a += part[0][i]; b += part[1][i]; }
    const float r0 = rsqrt(a / (float) hw + eps), r1 = rsqrt(b / (float) hw + eps);
    for (int c = (int) tid; c < d; c += 256) y[(ulong) t * d + c] = bfr(bf(w[c]) * (xr[c] * (c < hw ? r0 : r1)));
}

// ---- matvec: one simdgroup per output row, lanes over the 8-weight blocks of the row ------------------------------------
// The 8 weights of block j of row r (slice s) as f32, for the BF16 / Q8 / Q4 formats (dequantized to BF16).
static inline void wblock(short fmt, device const uchar* W, device const ushort* S, device const ushort* B, ulong row, int K,
                          int j, thread float (&w)[8]) {
    if (fmt == MF_BF16) {
        const uint4 v = *(device const uint4*) ((device const ushort*) W + row * (ulong) K + (ulong) j * 8);
        const uint u[4] = {v.x, v.y, v.z, v.w};
        for (int i = 0; i < 4; ++i) { w[2 * i] = as_type<float>(u[i] << 16); w[2 * i + 1] = as_type<float>(u[i] & 0xFFFF0000u); }
    } else {
        const ulong gi = (row * (ulong) K + (ulong) j * 8) / 64;
        const float s = bf(S[gi]), b = bf(B[gi]);
        if (fmt == MF_Q8) {
            const uint2 q = *(device const uint2*) ((device const uint*) W + (row * (ulong) K) / 4 + (ulong) j * 2);
            for (int i = 0; i < 4; ++i) { w[i] = bfr(s * (float) ((q.x >> (8 * i)) & 255u) + b); w[4 + i] = bfr(s * (float) ((q.y >> (8 * i)) & 255u) + b); }
        } else {
            const uint q = ((device const uint*) W)[(row * (ulong) K) / 8 + (ulong) j];
            for (int i = 0; i < 8; ++i) w[i] = bfr(s * (float) ((q >> (4 * i)) & 15u) + b);
        }
    }
}

// SEED4 block j of row r: returns 2^e times (sum_p q_p (x . V_p) + (sum_p q_p) xm), xm = -32768 sum(x); the caller
// applies R32 per row.  States from the per-seed stream table G (lfsr_stream24).
static inline void seed_states(uint s, uint g, thread float (&v)[24]) {
    // v_k = 1 + V_k / 2^16, exact: the 16-bit state inserted under the exponent of 1.0 (a shift and an insert; no
    // subtract: the 1 is folded into the block's input-sum term by seed_dot_xm).  g = G[s], the stream table entry.
    const uint lo = s | (g << 16);
    for (short k = 1; k <= 16; ++k) v[k - 1] = as_type<float>(insert_bits(0x3F800000u, lo >> k, 7u, 16u));
    for (short k = 17; k <= 24; ++k) v[k - 1] = as_type<float>(insert_bits(0x3F800000u, g >> (k - 16), 7u, 16u));
}
// One block's contribution without R32: 2^e (sum_p q_p sum_c x_c V_cp - 32768 (sum_p q_p) sum_c x_c).  With
// v = 1 + V / 2^16 that is 2^(e+16) (sum_p q_p d_p + (sum_p q_p) xm) for d_p = x . v_p and xm = -1.5 sum(x); the
// rounding error stays of the order of the direct form's (the 1 and the 32768 cancel against the same sum of x).
static inline float seed_dot_xm(thread const float (&v)[24], int nb, int ebias, float4 x0, float4 x1, float xm) {
    // the signed 4-bit coefficients exactly, without a convert: (u ^ 8) - 8 is u's two's-complement value
    const uint nbu = (uint) nb;
    const float q0 = as_type<float>(0x4B000000u | (extract_bits(nbu, 4u, 4u) ^ 8u)) - 8388616.0f;
    const float q1 = as_type<float>(0x4B000000u | (extract_bits(nbu, 8u, 4u) ^ 8u)) - 8388616.0f;
    const float q2 = as_type<float>(0x4B000000u | (extract_bits(nbu, 12u, 4u) ^ 8u)) - 8388616.0f;
    const float e2 = as_type<float>(uint(ebias + (nb & 15) + 127 + 16) << 23);
    const float d0 = dot(x0, float4(v[0], v[3], v[6], v[9])) + dot(x1, float4(v[12], v[15], v[18], v[21]));
    const float d1 = dot(x0, float4(v[1], v[4], v[7], v[10])) + dot(x1, float4(v[13], v[16], v[19], v[22]));
    const float d2 = dot(x0, float4(v[2], v[5], v[8], v[11])) + dot(x1, float4(v[14], v[17], v[20], v[23]));
    return e2 * (((q0 * d0 + q1 * d1) + q2 * d2) + ((q0 + q1) + q2) * xm);
}

// SEED4P4, 4.5-bit P = 4 blocks (nslm/search4.h): the 32 states as 1 + V / 2^16 (two ops each), from the 32-bit stream
// table entry g = G32[s] (lfsr_stream32): state k = bits k .. k + 15 of the stream s | g << 16.
static inline void seed4_states(uint s, uint g, thread float (&v)[32]) {
    const uint lo = s | (g << 16);
    for (short k = 1; k <= 16; ++k) v[k - 1] = as_type<float>(insert_bits(0x3F800000u, lo >> k, 7u, 16u));
    for (short k = 17; k <= 32; ++k) v[k - 1] = as_type<float>(insert_bits(0x3F800000u, g >> (k - 16), 7u, 16u));
}
// One block without R32: 2^(e+16) (sum_p q_p d_p + (sum_p q_p) xm), d_p = x . v_p, xm = -1.5 sum(x) (as seed_dot_xm).
static inline float seed4_dot(thread const float (&v)[32], uint cw, int e, float4 x0, float4 x1, float xm) {
    const float q0 = as_type<float>(0x4B000000u | (extract_bits(cw, 0u, 4u) ^ 8u)) - 8388616.0f;
    const float q1 = as_type<float>(0x4B000000u | (extract_bits(cw, 4u, 4u) ^ 8u)) - 8388616.0f;
    const float q2 = as_type<float>(0x4B000000u | (extract_bits(cw, 8u, 4u) ^ 8u)) - 8388616.0f;
    const float q3 = as_type<float>(0x4B000000u | (extract_bits(cw, 12u, 4u) ^ 8u)) - 8388616.0f;
    const float e2 = as_type<float>(uint(e + 127 + 16) << 23);
    const float d0 = dot(x0, float4(v[0], v[4], v[8], v[12])) + dot(x1, float4(v[16], v[20], v[24], v[28]));
    const float d1 = dot(x0, float4(v[1], v[5], v[9], v[13])) + dot(x1, float4(v[17], v[21], v[25], v[29]));
    const float d2 = dot(x0, float4(v[2], v[6], v[10], v[14])) + dot(x1, float4(v[18], v[22], v[26], v[30]));
    const float d3 = dot(x0, float4(v[3], v[7], v[11], v[15])) + dot(x1, float4(v[19], v[23], v[27], v[31]));
    return e2 * ((((q0 * d0 + q1 * d1) + q2 * d2) + q3 * d3) + (((q0 + q1) + q2) + q3) * xm);
}
static inline int seed4_ecode(device const uchar* EN, ulong k) { return (EN[k >> 1] >> ((k & 1) * 4)) & 15; }
// SEED6P8, 6.5-bit P = 8 blocks (nslm/searchp.h): 64 states; states 33..64 are states 1..32 of the seed s2 = state 32
// = G[s] >> 16, so two stream-table words give them all.  seed8_dot as seed4_dot with 8 coefficients (32-bit word).
static inline void seed8_states(uint s, device const uint* G, thread float (&v)[64]) {
    const uint g = G[s], s2 = g >> 16, g2 = G[s2], lo = s | (g << 16), lo2 = s2 | (g2 << 16);
    for (short k = 1; k <= 16; ++k) v[k - 1] = as_type<float>(insert_bits(0x3F800000u, lo >> k, 7u, 16u));
    for (short k = 17; k <= 32; ++k) v[k - 1] = as_type<float>(insert_bits(0x3F800000u, g >> (k - 16), 7u, 16u));
    for (short k = 1; k <= 16; ++k) v[31 + k] = as_type<float>(insert_bits(0x3F800000u, lo2 >> k, 7u, 16u));
    for (short k = 17; k <= 32; ++k) v[31 + k] = as_type<float>(insert_bits(0x3F800000u, g2 >> (k - 16), 7u, 16u));
}
static inline float seed8_dot(thread const float (&v)[64], uint cw, int e, float4 x0, float4 x1, float xm) {
    const float e2 = as_type<float>(uint(e + 127 + 16) << 23);
    float r = 0.0f, qs = 0.0f;
    for (short p = 0; p < 8; ++p) {
        const float q = as_type<float>(0x4B000000u | (extract_bits(cw, uint(4 * p), 4u) ^ 8u)) - 8388616.0f;
        const float d = dot(x0, float4(v[p], v[8 + p], v[16 + p], v[24 + p])) + dot(x1, float4(v[32 + p], v[40 + p], v[48 + p], v[56 + p]));
        r += q * d;
        qs += q;
    }
    return e2 * (r + qs * xm);
}

// One lane's share of row `row` . x for one token (T = 1), every format: the k_mv arithmetic (seed4 two-op states,
// Q4 in MLX's qmv form, BF16 / Q8 dequantized as mx.dequantize).  The caller sums the lanes.
static inline float mv_lane(short fmt, device const uchar* W, device const ushort* S, device const ushort* B, device const uint* G,
                            int eb, ulong row, int K, device const float* xb, uint lane, device const uchar* EN) {
    const int nbk = K / 8;
    float acc = 0;
    if (fmt == MF_SEED6P8) {
        device const ushort* seeds = (device const ushort*) W;
        device const uint* cws = (device const uint*) S;
        for (int j = (int) lane; j < nbk; j += 32) {
            const ulong k = row * (ulong) nbk + j;
            float v[64];
            seed8_states(seeds[k], G, v);
            device const float4* xr = (device const float4*) (xb + (ulong) j * 8);
            const float4 x0 = xr[0], x1 = xr[1];
            acc += seed8_dot(v, cws[k], eb + seed4_ecode(EN, k), x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
        }
        return acc * (1.0f / 32767.0f);
    }
    if (fmt == MF_SEED4P4) {
        device const ushort* seeds = (device const ushort*) W;
        if (MV_SEED_BPL == 2 && nbk % 64 == 0) {   // two adjacent blocks per lane: 4-byte seed / coefficient loads, one
                                                    // exponent byte (k even: both nibbles)
            device const uint* s2 = (device const uint*) (seeds + row * (ulong) nbk);
            device const uint* c2 = (device const uint*) (S + row * (ulong) nbk);
            for (int jq = (int) lane; jq < nbk / 2; jq += 32) {
                const ulong k = row * (ulong) nbk + 2 * jq;
                const uint ss = s2[jq], cc = c2[jq], eb2 = EN[k >> 1];
                for (int u = 0; u < 2; ++u) {
                    const uint s = u ? ss >> 16 : ss & 0xFFFFu;
                    float v[32];
                    seed4_states(s, G[s], v);
                    device const float4* xr = (device const float4*) (xb + (ulong) (2 * jq + u) * 8);
                    const float4 x0 = xr[0], x1 = xr[1];
                    acc += seed4_dot(v, u ? cc >> 16 : cc & 0xFFFFu, eb + (int) ((eb2 >> (4 * u)) & 15u), x0, x1,
                                     -1.5f * dot(x0 + x1, float4(1.0f)));
                }
            }
            return acc * (1.0f / 32767.0f);
        }
        for (int j = (int) lane; j < nbk; j += 32) {
            const ulong k = row * (ulong) nbk + j;
            const uint s = seeds[k];
            float v[32];
            seed4_states(s, G[s], v);
            device const float4* xr = (device const float4*) (xb + (ulong) j * 8);
            const float4 x0 = xr[0], x1 = xr[1];
            acc += seed4_dot(v, (uint) S[k], eb + seed4_ecode(EN, k), x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
        }
        return acc * (1.0f / 32767.0f);
    }
    if (fmt == MF_SEED4) {
        device const ushort* seeds = (device const ushort*) W;
        if (MV_SEED_BPL > 1 && nbk % (32 * MV_SEED_BPL) == 0) {   // adjacent blocks per lane per step (one wide load per stream)
                                                                   // when every lane gets the same number of steps
            constexpr int Q = MV_SEED_BPL;
            device const ushort* sr = seeds + row * (ulong) nbk;
            device const ushort* nr = S + row * (ulong) nbk;
            for (int jq = (int) lane; jq < nbk / Q; jq += 32) {
                ushort sv[Q], nv[Q];
                if (Q == 4) {
                    const uint2 a2 = *(device const uint2*) (sr + jq * Q), b2 = *(device const uint2*) (nr + jq * Q);
                    sv[0] = (ushort) a2.x; sv[1] = (ushort) (a2.x >> 16); sv[Q - 2] = (ushort) a2.y; sv[Q - 1] = (ushort) (a2.y >> 16);
                    nv[0] = (ushort) b2.x; nv[1] = (ushort) (b2.x >> 16); nv[Q - 2] = (ushort) b2.y; nv[Q - 1] = (ushort) (b2.y >> 16);
                } else {
                    const uint a1 = *(device const uint*) (sr + jq * Q), b1 = *(device const uint*) (nr + jq * Q);
                    sv[0] = (ushort) a1; sv[Q - 1] = (ushort) (a1 >> 16);
                    nv[0] = (ushort) b1; nv[Q - 1] = (ushort) (b1 >> 16);
                }
                for (int k = 0; k < Q; ++k) {
                    const uint s = sv[k];
                    float v[24];
                    seed_states(s, G[s], v);
                    device const float4* xr = (device const float4*) (xb + (ulong) (jq * Q + k) * 8);
                    const float4 x0 = xr[0], x1 = xr[1];
                    acc += seed_dot_xm(v, (int) nv[k], eb, x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
                }
            }
            return acc * (1.0f / 32767.0f);
        }
        for (int j = (int) lane; j < nbk; j += 32) {
            const uint s = seeds[row * (ulong) nbk + j];
            float v[24];
            seed_states(s, G[s], v);
            device const float4* xr = (device const float4*) (xb + (ulong) j * 8);
            const float4 x0 = xr[0], x1 = xr[1];
            acc += seed_dot_xm(v, (int) S[row * (ulong) nbk + j], eb, x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
        }
        return acc * (1.0f / 32767.0f);
    }
    if (fmt == MF_Q4) {
        for (int j0 = (int) lane * MV_Q4_BPL; j0 < nbk; j0 += 32 * MV_Q4_BPL) {
            const ulong gi = (row * (ulong) K + (ulong) j0 * 8) / 64;
            const float s = bf(S[gi]), b = bf(B[gi]);
            device const uint* qp = (device const uint*) W + (row * (ulong) K) / 8 + (ulong) j0;
            for (int k = 0; k < MV_Q4_BPL; ++k) {
                const uint qw = qp[k];
                const float4 q0 = float4(as_type<float>(0x4B000000u | extract_bits(qw, 0u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 4u, 4u)),
                                         as_type<float>(0x4B000000u | extract_bits(qw, 8u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 12u, 4u))) - 8388608.0f;
                const float4 q1 = float4(as_type<float>(0x4B000000u | extract_bits(qw, 16u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 20u, 4u)),
                                         as_type<float>(0x4B000000u | extract_bits(qw, 24u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 28u, 4u))) - 8388608.0f;
                device const float4* xr = (device const float4*) (xb + (ulong) (j0 + k) * 8);
                const float4 x0 = xr[0], x1 = xr[1];
                acc += s * (dot(q0, x0) + dot(q1, x1)) + b * dot(x0 + x1, float4(1.0f));
            }
        }
        return acc;
    }
    for (int j = (int) lane; j < nbk; j += 32) {
        float w[8];
        wblock(fmt, W, S, B, row, K, j, w);
        device const float4* xr = (device const float4*) (xb + (ulong) j * 8);
        acc += dot(float4(w[0], w[1], w[2], w[3]), xr[0]) + dot(float4(w[4], w[5], w[6], w[7]), xr[1]);
    }
    return acc;
}

// Dense: y[t][r] for t < FC_T tokens sharing each weight-row load (and each seed-state expansion).
// Gather: grid y = pair p; row r of slice sel[p]; input row p / xdiv.
// Each simdgroup takes MV_NR(fmt) consecutive rows; the lanes walk the 8-weight blocks, each x block loaded once for
// all of them.  Buffers: 0 args, 1 W (codes / BF16 / seeds), 2 scales (Q8/Q4) or nibble / coefficient words (seeds),
// 3 biases (Q8/Q4) or int32 slice biases (seeds), 4 x, 5 y, 6 sel (gather), 7 G (seeds), 8 exponent nibbles (SEED4P4).
kernel void k_mv(constant MvArgs& a [[buffer(0)]], device const uchar* W [[buffer(1)]], device const ushort* S [[buffer(2)]],
                 device const ushort* B [[buffer(3)]], device const float* x [[buffer(4)]], device float* y [[buffer(5)]],
                 device const int* sel [[buffer(6)]], device const uint* G [[buffer(7)]], device const uchar* EN [[buffer(8)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                 uint sgi [[simdgroup_index_in_threadgroup]]) {
    const int NR = FC_FMT == MF_SEED4 ? MV_NR_SEED : MV_NR_AFF;   // folded when the pipeline is specialised
    const int r0 = (int) (tg.x * MV_ROWS + sgi) * NR;
    if (r0 >= a.R) return;
    const bool gather = FC_T == 0;
    const int nt = gather ? 1 : FC_T;
    const int p0 = gather ? (int) tg.y : 0;
    const int slice = gather ? sel[p0] : 0;
    const int nbk = a.K / 8;
    ulong row[MV_NR_MAX];
    for (int i = 0; i < NR; ++i) row[i] = (ulong) slice * a.R + min(r0 + i, a.R - 1);
    float acc[MV_NR_MAX][MV_MAXT];
    for (int i = 0; i < NR; ++i) for (int t = 0; t < MV_MAXT; ++t) acc[i][t] = 0;
    device const float* xb = x + (ulong) (gather ? p0 / a.xdiv : 0) * a.xs;
    if (nt == 1 && NR == 1) {   // one token (decode, gather): the shared per-lane arithmetic
        const int ebs = FC_FMT == MF_SEED4 || FC_FMT == MF_SEED4P4 || FC_FMT == MF_SEED6P8 ? ((device const int*) B)[slice] : 0;
        const float s = simd_sum(mv_lane(FC_FMT, W, S, B, G, ebs, row[0], a.K, xb, lane, EN));
        if (lane == 0) {
            device float* yo = y + (ulong) p0 * a.ys + r0;
            *yo = a.add ? bfr(*yo + bfr(s)) : bfr(s);
        }
        return;
    }
    if (FC_FMT == MF_SEED4) {
        device const ushort* seeds = (device const ushort*) W;
        const int eb = ((device const int*) B)[slice];
        // MV_SEED_U interleaved blocks (j, j + 32, ...) per lane per step: their seed, nibble and table loads in flight
        for (int j0 = (int) lane; j0 < nbk; j0 += 32 * MV_SEED_U) {
            for (int i = 0; i < NR; ++i) {
                uint sv[MV_SEED_U], gv[MV_SEED_U];
                int nbv[MV_SEED_U];
                for (int u = 0; u < MV_SEED_U; ++u) {
                    const int j = min(j0 + 32 * u, nbk - 1);
                    sv[u] = seeds[row[i] * (ulong) nbk + j];
                    nbv[u] = (int) S[row[i] * (ulong) nbk + j];
                }
                for (int u = 0; u < MV_SEED_U; ++u) gv[u] = G[sv[u]];
                for (int u = 0; u < MV_SEED_U; ++u) {
                    const int j = j0 + 32 * u;
                    if (j >= nbk) break;
                    float v[24];
                    seed_states(sv[u], gv[u], v);
                    for (int t = 0; t < nt; ++t) {
                        device const float4* xr = (device const float4*) (xb + (ulong) t * a.xs + (ulong) j * 8);
                        const float4 x0 = xr[0], x1 = xr[1];
                        acc[i][t] += seed_dot_xm(v, nbv[u], eb, x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
                    }
                }
            }
        }
        for (int i = 0; i < NR; ++i) for (int t = 0; t < nt; ++t) acc[i][t] *= (1.0f / 32767.0f);
    } else if (FC_FMT == MF_SEED6P8) {   // T > 1: one state expansion per block for every token
        device const ushort* seeds = (device const ushort*) W;
        device const uint* cws = (device const uint*) S;
        const int eb = ((device const int*) B)[slice];
        for (int j = (int) lane; j < nbk; j += 32)
            for (int i = 0; i < NR; ++i) {
                const ulong k = row[i] * (ulong) nbk + j;
                float v[64];
                seed8_states(seeds[k], G, v);
                const int e = eb + seed4_ecode(EN, k);
                for (int t = 0; t < nt; ++t) {
                    device const float4* xr = (device const float4*) (xb + (ulong) t * a.xs + (ulong) j * 8);
                    const float4 x0 = xr[0], x1 = xr[1];
                    acc[i][t] += seed8_dot(v, cws[k], e, x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
                }
            }
        for (int i = 0; i < NR; ++i) for (int t = 0; t < nt; ++t) acc[i][t] *= (1.0f / 32767.0f);
    } else if (FC_FMT == MF_SEED4P4) {   // T > 1: one state expansion per block for every token
        device const ushort* seeds = (device const ushort*) W;
        const int eb = ((device const int*) B)[slice];
        for (int j = (int) lane; j < nbk; j += 32)
            for (int i = 0; i < NR; ++i) {
                const ulong k = row[i] * (ulong) nbk + j;
                const uint s = seeds[k];
                float v[32];
                seed4_states(s, G[s], v);
                const int e = eb + seed4_ecode(EN, k);
                for (int t = 0; t < nt; ++t) {
                    device const float4* xr = (device const float4*) (xb + (ulong) t * a.xs + (ulong) j * 8);
                    const float4 x0 = xr[0], x1 = xr[1];
                    acc[i][t] += seed4_dot(v, (uint) S[k], e, x0, x1, -1.5f * dot(x0 + x1, float4(1.0f)));
                }
            }
        for (int i = 0; i < NR; ++i) for (int t = 0; t < nt; ++t) acc[i][t] *= (1.0f / 32767.0f);
    } else if (FC_FMT == MF_Q4) {
        // MLX's qmv form: per block, s * sum(q x) + b * sum(x) in f32; q exact via or/subtract, no convert.
        // MV_Q4_BPL consecutive blocks per lane per step (one wider load; the same group's scale and bias)
        for (int j0 = (int) lane * MV_Q4_BPL; j0 < nbk; j0 += 32 * MV_Q4_BPL) {
            for (int i = 0; i < NR; ++i) {
                const ulong gi = (row[i] * (ulong) a.K + (ulong) j0 * 8) / 64;
                const float s = bf(S[gi]), b = bf(B[gi]);
                device const uint* qp = (device const uint*) W + (row[i] * (ulong) a.K) / 8 + (ulong) j0;
                uint qws[MV_Q4_BPL];
                if (MV_Q4_BPL == 2) { const uint2 v = *(device const uint2*) qp; qws[0] = v.x; qws[MV_Q4_BPL - 1] = v.y; }
                else for (int k = 0; k < MV_Q4_BPL; ++k) qws[k] = qp[k];
                for (int k = 0; k < MV_Q4_BPL; ++k) {
                    const uint qw = qws[k];
                    const float4 q0 = float4(as_type<float>(0x4B000000u | extract_bits(qw, 0u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 4u, 4u)),
                                             as_type<float>(0x4B000000u | extract_bits(qw, 8u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 12u, 4u))) - 8388608.0f;
                    const float4 q1 = float4(as_type<float>(0x4B000000u | extract_bits(qw, 16u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 20u, 4u)),
                                             as_type<float>(0x4B000000u | extract_bits(qw, 24u, 4u)), as_type<float>(0x4B000000u | extract_bits(qw, 28u, 4u))) - 8388608.0f;
                    for (int t = 0; t < nt; ++t) {
                        device const float4* xr = (device const float4*) (xb + (ulong) t * a.xs + (ulong) (j0 + k) * 8);
                        const float4 x0 = xr[0], x1 = xr[1];
                        acc[i][t] += s * (dot(q0, x0) + dot(q1, x1)) + b * dot(x0 + x1, float4(1.0f));
                    }
                }
            }
        }
    } else {
        for (int j = (int) lane; j < nbk; j += 32) {
            float4 x0[MV_MAXT], x1[MV_MAXT];
            for (int t = 0; t < nt; ++t) {
                device const float4* xr = (device const float4*) (xb + (ulong) t * a.xs + (ulong) j * 8);
                x0[t] = xr[0];
                x1[t] = xr[1];
            }
            for (int i = 0; i < NR; ++i) {
                float w[8];
                wblock(FC_FMT, W, S, B, row[i], a.K, j, w);
                const float4 w0 = float4(w[0], w[1], w[2], w[3]), w1 = float4(w[4], w[5], w[6], w[7]);
                for (int t = 0; t < nt; ++t) acc[i][t] += dot(w0, x0[t]) + dot(w1, x1[t]);
            }
        }
    }
    for (int i = 0; i < NR; ++i) {
        if (r0 + i >= a.R) break;
        for (int t = 0; t < nt; ++t) {
            const float s = simd_sum(acc[i][t]);
            if (lane == 0) {
                device float* yo = y + (ulong) (gather ? p0 : t) * a.ys + r0 + i;
                *yo = a.add ? bfr(*yo + bfr(s)) : bfr(s);
            }
        }
    }
}


// Routed experts' gate and up for one (token, expert) pair per grid row, and the SwiGLU, in one dispatch:
// a[p][r] = bf16(silu(bf16(gate . x)) * bf16(up . x)) - what k_mv (gate), k_mv (up) and k_swiglu compute in three.
// Buffers: 0 args, 1-3 gate W / S / B, 4 x, 5 a, 6 sel, 7 G, 8-10 up W / S / B, 11-12 gate / up exponent nibbles
// (SEED4P4).  One row per simdgroup (MV_NR 1).
kernel void k_mv_gu(constant MvArgs& a [[buffer(0)]], device const uchar* W [[buffer(1)]], device const ushort* S [[buffer(2)]],
                    device const ushort* B [[buffer(3)]], device const float* x [[buffer(4)]], device float* y [[buffer(5)]],
                    device const int* sel [[buffer(6)]], device const uint* G [[buffer(7)]], device const uchar* Wu [[buffer(8)]],
                    device const ushort* Su [[buffer(9)]], device const ushort* Bu [[buffer(10)]],
                    device const uchar* EN [[buffer(11)]], device const uchar* ENu [[buffer(12)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                    uint sgi [[simdgroup_index_in_threadgroup]]) {
    static_assert(MV_NR_SEED == 1 && MV_NR_AFF == 1, "k_mv_gu: one row per simdgroup");
    const int r = (int) (tg.x * MV_ROWS + sgi);
    if (r >= a.R) return;
    const int p0 = (int) tg.y, slice = sel[p0];
    const ulong row = (ulong) slice * a.R + r;
    device const float* xb = x + (ulong) (p0 / a.xdiv) * a.xs;
    const bool sd = FC_FMT == MF_SEED4 || FC_FMT == MF_SEED4P4;
    const int ebg = sd ? ((device const int*) B)[slice] : 0;
    const int ebu = sd ? ((device const int*) Bu)[slice] : 0;
    const float g = simd_sum(mv_lane(FC_FMT, W, S, B, G, ebg, row, a.K, xb, lane, EN));
    const float u = simd_sum(mv_lane(FC_FMT, Wu, Su, Bu, G, ebu, row, a.K, xb, lane, ENu));
    if (lane == 0) y[(ulong) p0 * a.ys + r] = bfr(silu_bf(bfr(g)) * bfr(u));
}


// ---- GEMM (prefill): Y[n][r] = sum_k W[r][k] X[n][k] for a 32-row x 32-token tile; f32 simdgroup matrices (exact
// products of the BF16-valued weights and activations, f32 accumulation).  The weight tile is dequantized per format
// into threadgroup memory: each thread dequantizes 8 consecutive weights (one block) per K step.
// Dense: tile (row block, token block).  Grouped: tile (row block, entry of the tile table); the entry's pairs are
// perm[start .. start + count), each reading input row pair / xdiv and writing output row pair.
// SEED4P4 block j of row `row` (K columns): centred states, isum * fl(R32 2^e)
static inline void seed4p4_block(device const uchar* W, device const ushort* S, device const uint* G, ulong row, int K, int j,
                                 int ebias, device const uchar* EN, thread float (&w)[8]) {
    const int nbk = K / 8;
    const ulong k = row * (ulong) nbk + j;
    const uint s = ((device const ushort*) W)[k], cw = S[k];
    const uint g = G[s], lo = s | (g << 16);
    float c[32];
    for (short kk = 1; kk <= 16; ++kk) c[kk - 1] = as_type<float>(0x4B000000u | extract_bits(lo, (uint) kk, 16u)) - 8421376.0f;
    for (short kk = 17; kk <= 32; ++kk) c[kk - 1] = as_type<float>(0x4B000000u | extract_bits(g, (uint) (kk - 16), 16u)) - 8421376.0f;
    const int ci = (int) cw;
    const float q0 = float((ci << 28) >> 28), q1 = float((ci << 24) >> 28), q2 = float((ci << 20) >> 28), q3 = float((ci << 16) >> 28);
    const float sc = as_type<float>(uint(ebias + seed4_ecode(EN, k) + 127) << 23) * (1.0f / 32767.0f);
    for (short i = 0; i < 8; ++i) w[i] = ((((c[4 * i] * q0) + (c[4 * i + 1] * q1)) + (c[4 * i + 2] * q2)) + (c[4 * i + 3] * q3)) * sc;
}
// SEED6P8 block j of row `row`: centred states, isum * fl(R32 2^e); states 33..64 from s2 = state 32
static inline void seed6p8_block(device const uchar* W, device const ushort* S, device const uint* G, ulong row, int K, int j,
                                 int ebias, device const uchar* EN, thread float (&w)[8]) {
    const int nbk = K / 8;
    const ulong k = row * (ulong) nbk + j;
    const uint s = ((device const ushort*) W)[k], cw = ((device const uint*) S)[k];
    const uint g = G[s], lo = s | (g << 16), s2 = g >> 16, g2 = G[s2], lo2 = s2 | (g2 << 16);
    float c[64];
    for (short kk = 1; kk <= 16; ++kk) c[kk - 1] = as_type<float>(0x4B000000u | extract_bits(lo, (uint) kk, 16u)) - 8421376.0f;
    for (short kk = 17; kk <= 32; ++kk) c[kk - 1] = as_type<float>(0x4B000000u | extract_bits(g, (uint) (kk - 16), 16u)) - 8421376.0f;
    for (short kk = 1; kk <= 16; ++kk) c[31 + kk] = as_type<float>(0x4B000000u | extract_bits(lo2, (uint) kk, 16u)) - 8421376.0f;
    for (short kk = 17; kk <= 32; ++kk) c[31 + kk] = as_type<float>(0x4B000000u | extract_bits(g2, (uint) (kk - 16), 16u)) - 8421376.0f;
    float q[8];
    for (short p = 0; p < 8; ++p) q[p] = float((int) (cw << (28 - 4 * p)) >> 28);
    const float sc = as_type<float>(uint(ebias + seed4_ecode(EN, k) + 127) << 23) * (1.0f / 32767.0f);
    for (short i = 0; i < 8; ++i) {
        float is = 0.0f;
        for (short p = 0; p < 8; ++p) is = is + c[8 * i + p] * q[p];   // exact: |partial sums| < 2^21
        w[i] = is * sc;
    }
}
static inline void tile_weights(short fmt, device const uchar* W, device const ushort* S, device const ushort* B, device const uint* G,
                                int slice, ulong row, int K, int j, int ebias, threadgroup float* dst, device const uchar* EN) {
    float w[8];
    if (fmt == MF_SEED6P8) seed6p8_block(W, S, G, row, K, j, ebias, EN, w);
    else if (fmt == MF_SEED4P4) seed4p4_block(W, S, G, row, K, j, ebias, EN, w);
    else if (fmt == MF_SEED4) {
        const int nbk = K / 8;
        const ushort s = ((device const ushort*) W)[row * (ulong) nbk + j];
        const int nb = (int) S[row * (ulong) nbk + j];
        const uint g = G[s], lo = (uint) s | (g << 16);
        float c[24];
        for (short k = 1; k <= 16; ++k) c[k - 1] = as_type<float>(0x4B000000u | extract_bits(lo, (uint) k, 16u)) - 8421376.0f;
        for (short k = 17; k <= 24; ++k) c[k - 1] = as_type<float>(0x4B000000u | extract_bits(g, (uint) (k - 16), 16u)) - 8421376.0f;
        const float q0 = float((nb << 24) >> 28), q1 = float((nb << 20) >> 28), q2 = float((nb << 16) >> 28);
        const float sc = as_type<float>(uint(ebias + (nb & 15) + 127) << 23) * (1.0f / 32767.0f);
        for (short i = 0; i < 8; ++i) w[i] = (((c[3 * i] * q0) + (c[3 * i + 1] * q1)) + (c[3 * i + 2] * q2)) * sc;
    } else wblock(fmt, W, S, B, row, K, j, w);
    for (short i = 0; i < 8; ++i) dst[i] = w[i];
}

static inline float softplus_ln2_bf(float g) {   // bf16(softplus(g, beta = ln 2)), in f32 as k_attn_reduce
    const float gx = g * 0.69314718055994531f;
    return bfr((max(gx, 0.0f) + log(1.0f + exp(-fabs(gx)))) / 0.69314718055994531f);
}
kernel void k_mm(constant MmArgs& a [[buffer(0)]], device const uchar* W [[buffer(1)]], device const ushort* S [[buffer(2)]],
                 device const ushort* B [[buffer(3)]], device const float* X [[buffer(4)]], device float* Y [[buffer(5)]],
                 device const int* perm [[buffer(6)]], device const uint* G [[buffer(7)]], device const MmTile* tiles [[buffer(8)]],
                 device const uchar* EN [[buffer(9)]], device const float* Gt [[buffer(10)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Wt[MM_BM * MM_BK];   // [row][k]
    threadgroup float Xt[MM_BK * MM_BN];   // [k][n]
    threadgroup float Ot[MM_BN * MM_BM];   // [n][row] for the store
    const bool grouped = FC_T == 0;
    const int r0 = (int) tg.x * MM_BM;
    int slice = 0, start = 0, count;
    if (grouped) { const MmTile tl = tiles[tg.y]; slice = tl.slice; start = tl.start; count = tl.count; }
    else { start = (int) tg.y * MM_BN; count = min(MM_BN, a.T - start); }
    const int ebias = FC_FMT == MF_SEED4 || FC_FMT == MF_SEED4P4 || FC_FMT == MF_SEED6P8 ? ((device const int*) B)[slice] : 0;
    simdgroup_float8x8 acc[2][2];
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    const int sr = (int) (sgi / 2) * 16, sn = (int) (sgi % 2) * 16;   // this simdgroup's 16 x 16 sub-tile
    // the input row of each of the tile's 32 columns (token or pair)
    for (int k0 = 0; k0 < a.K; k0 += MM_BK) {
        if (FC_T == 2 && FC_FMT == MF_SEED4P4) {   // SEED4P4 stored transposed [K][R]: a thread's 8 rows are one block
            const int kk = (int) tid % 32, rb = ((int) tid / 32) * 8;
            if (r0 + rb < a.R) {
                float w[8];
                seed4p4_block(W, S, G, (ulong) (k0 + kk), a.R, (r0 + rb) / 8, ebias, EN, w);
                for (int i = 0; i < 8; ++i) Wt[(rb + i) * MM_BK + kk] = w[i];
            } else for (int i = 0; i < 8; ++i) Wt[(rb + i) * MM_BK + kk] = 0;
        } else if (FC_T == 2 && FC_FMT != MF_BF16) {   // Q8 / Q4 weights stored transposed [K][R] (R a multiple of 64): a thread's
            const int kk = (int) tid % 32, rb = ((int) tid / 32) * 8;   // 8 rows are consecutive codes of one group
            const ulong e0 = (ulong) (k0 + kk) * a.R + r0 + rb;
            if (r0 + rb < a.R) {
                const float sc = bf(S[e0 / 64]), bi = bf(B[e0 / 64]);
                const uint2 q = FC_FMT == MF_Q8 ? *(device const uint2*) (W + e0) : uint2(*(device const uint*) (W + e0 / 2), 0u);
                for (int i = 0; i < 8; ++i) {
                    const uint c = FC_FMT == MF_Q8 ? (i < 4 ? q.x >> (8 * i) : q.y >> (8 * (i - 4))) & 255u : (q.x >> (4 * i)) & 15u;
                    Wt[(rb + i) * MM_BK + kk] = bfr(sc * (float) c + bi);
                }
            } else for (int i = 0; i < 8; ++i) Wt[(rb + i) * MM_BK + kk] = 0;
        } else if (FC_T == 2) {   // BF16 weights stored transposed [K][R] (R a multiple of 8): one k and 8 rows per thread
            const int kk = (int) tid % 32, rb = ((int) tid / 32) * 8;
            if (r0 + rb < a.R) {
                const uint4 w = *(device const uint4*) ((device const ushort*) W + (ulong) (k0 + kk) * a.R + r0 + rb);
                const uint u[4] = {w.x, w.y, w.z, w.w};
                for (int i = 0; i < 4; ++i) {
                    Wt[(rb + 2 * i) * MM_BK + kk] = as_type<float>(u[i] << 16);
                    Wt[(rb + 2 * i + 1) * MM_BK + kk] = as_type<float>(u[i] & 0xFFFF0000u);
                }
            } else for (int i = 0; i < 8; ++i) Wt[(rb + i) * MM_BK + kk] = 0;
        } else {   // weights: 32 rows x 32 k = 128 blocks, one per thread
            const int rr = (int) tid / 4, jb = (int) tid % 4;
            const int row = r0 + rr;
            if (row < a.R) tile_weights(FC_FMT, W, S, B, G, slice, (ulong) slice * a.R + row, a.K, k0 / 8 + jb, ebias, Wt + rr * MM_BK + jb * 8, EN);
            else for (int i = 0; i < 8; ++i) Wt[rr * MM_BK + jb * 8 + i] = 0;
        }
        {   // inputs: 32 columns x 32 k, 8 per thread
            const int n = (int) tid / 4, kk = ((int) tid % 4) * 8;
            if (n < count) {
                const int src = grouped ? perm[start + n] / a.xdiv : start + n;
                device const float* xr = X + (ulong) src * a.xs + k0 + kk;
                for (int i = 0; i < 8; ++i) Xt[(kk + i) * MM_BN + n] = xr[i];
            } else for (int i = 0; i < 8; ++i) Xt[(kk + i) * MM_BN + n] = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kb = 0; kb < MM_BK; kb += 8) {
            simdgroup_float8x8 wa[2], xb[2];
            for (int i = 0; i < 2; ++i) simdgroup_load(wa[i], Wt + (sr + 8 * i) * MM_BK + kb, MM_BK);
            for (int j = 0; j < 2; ++j) simdgroup_load(xb[j], Xt + kb * MM_BN + sn + 8 * j, MM_BN);
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], wa[i], xb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // acc[i][j] holds rows sr+8i.., columns sn+8j..: store transposed into Ot[n][row]
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_store(acc[i][j], Ot + (sn + 8 * j) * MM_BM + sr + 8 * i, MM_BM, ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = (int) tid; idx < MM_BN * MM_BM; idx += 128) {
        const int n = idx / MM_BM, rr = idx % MM_BM, row = r0 + rr;
        if (n >= count || row >= a.R) continue;
        const int dst = grouped ? perm[start + n] : start + n;
        device float* yo = Y + (ulong) dst * a.ys + row;
        const float s = Ot[n * MM_BM + rr];
        *yo = a.add == 2 ? bfr(bfr(s) * softplus_ln2_bf(Gt[(ulong) dst * a.ys + row])) : a.add ? bfr(*yo + bfr(s)) : bfr(s);
    }
}

// ---- routers: logits with the release's two-partition BF16 contract, sigmoid, top-k by sigmoid + bias,
// weights = sigmoid / sum * scale.  k_router_logits: one simdgroup per (expert, row); router weights BF16 [n][d].
// score[t][e] = sigmoid, sel[t][e] = sigmoid + bias (selout).  k_router_topk: one thread per row.
kernel void k_router_logits(constant RouterArgs& a [[buffer(0)]], device const ushort* W [[buffer(1)]], device const ushort* bias [[buffer(2)]],
                            device const float* x [[buffer(3)]], device float* score [[buffer(4)]], device float* selout [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                            uint sgi [[simdgroup_index_in_threadgroup]]) {
    const int e = (int) (tg.x * 8 + sgi), t = (int) tg.y;
    if (e >= a.n) return;
    device const float* xr = x + (ulong) t * a.d;
    device const ushort* wr = W + (ulong) e * a.d;
    float s0 = 0, s1 = 0;
    for (int c = (int) lane * 8; c < a.hw; c += 256) {
        const uint4 w = *(device const uint4*) (wr + c);
        const float4 x0 = *(device const float4*) (xr + c), x1 = *(device const float4*) (xr + c + 4);
        s0 += dot(float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u), as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u)), x0)
            + dot(float4(as_type<float>(w.z << 16), as_type<float>(w.z & 0xFFFF0000u), as_type<float>(w.w << 16), as_type<float>(w.w & 0xFFFF0000u)), x1);
    }
    for (int c = a.hw + (int) lane * 8; c < a.d; c += 256) {
        const uint4 w = *(device const uint4*) (wr + c);
        const float4 x0 = *(device const float4*) (xr + c), x1 = *(device const float4*) (xr + c + 4);
        s1 += dot(float4(as_type<float>(w.x << 16), as_type<float>(w.x & 0xFFFF0000u), as_type<float>(w.y << 16), as_type<float>(w.y & 0xFFFF0000u)), x0)
            + dot(float4(as_type<float>(w.z << 16), as_type<float>(w.z & 0xFFFF0000u), as_type<float>(w.w << 16), as_type<float>(w.w & 0xFFFF0000u)), x1);
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (lane == 0) {
        const float sg = 1.0f / (1.0f + exp(-(bfr(s0) + bfr(s1))));
        score[t * a.n + e] = sg;
        selout[t * a.n + e] = sg + bf(bias[e]);
    }
}
// One simdgroup per token (threadgroups = T, 32 threads): k rounds of a simdgroup arg max over sel, ties to the lowest
// expert id (the scalar reference's strict >), chosen in descending order; weights = score / sum(chosen scores) x scale.
kernel void k_router_topk(constant RouterArgs& a [[buffer(0)]], device const float* score [[buffer(1)]], device const float* sel [[buffer(2)]],
                          device int* inds [[buffer(3)]], device float* wts [[buffer(4)]], constant int& T [[buffer(5)]],
                          uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    if ((int) t >= T) return;
    device const float* sc = score + (ulong) t * a.n;
    device const float* se = sel + (ulong) t * a.n;
    float v[4];   // lane holds experts lane, lane + 32, lane + 64, lane + 96 (n <= 128)
    for (int j = 0; j < 4; ++j) {
        const int e = (int) lane + 32 * j;
        v[j] = e < a.n ? se[e] : -INFINITY;
    }
    int chosen[16];
    float sum = 0;
    for (int k = 0; k < a.top_k; ++k) {
        float bv = -INFINITY;
        int bi = 1 << 20;
        for (int j = 0; j < 4; ++j) {
            const int e = (int) lane + 32 * j;
            if (e < a.n && (v[j] > bv || (v[j] == bv && e < bi))) { bv = v[j]; bi = e; }
        }
        const float m = simd_max(bv);
        const int best = simd_min(bv == m ? bi : 1 << 20);
        chosen[k] = best;
        sum += sc[best];
        if (best % 32 == (int) lane) v[best / 32] = -INFINITY;
    }
    if (lane == 0)
        for (int k = 0; k < a.top_k; ++k) {
            inds[t * a.top_k + k] = chosen[k];
            wts[t * a.top_k + k] = sc[chosen[k]] / sum * a.scale;
        }
}

// ---- elementwise -----------------------------------------------------------------------------------------------------
// MLP activation: a = bf16(silu(g) * u)  (g, u: [n] BF16-valued; n = rows x width)
kernel void k_swiglu(device const float* g [[buffer(0)]], device const float* u [[buffer(1)]], device float* a [[buffer(2)]],
                     constant int& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if ((int) i < n) a[i] = bfr(silu_bf(g[i]) * u[i]);
}

// MoE combine: y[t][c] = bf16(h[t][c] + bf16(bf16(sum_k bf16(D[t,k][c] * bf16(w[t,k]))) + shared[t][c]))
kernel void k_moe_combine(device const float* D [[buffer(0)]], device const float* w [[buffer(1)]], device const float* shared [[buffer(2)]],
                          device float* x [[buffer(3)]], constant int2& dk [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
    const int d = dk.x, k = dk.y, c = (int) g.x, t = (int) g.y;
    if (c >= d) return;
    float s = 0;
    for (int j = 0; j < k; ++j) s += bfr(D[((ulong) t * k + j) * d + c] * bfr(w[t * k + j]));
    const float routed = bfr(s);
    const ulong i = (ulong) t * d + c;
    x[i] = bfr(x[i] + bfr(routed + shared[i]));
}

// MoVA values: v[t][c] = bf16(sum_k bf16(silu(V[t,k][c]) * bf16(w[t,k])))
kernel void k_vcombine(device const float* V [[buffer(0)]], device const float* w [[buffer(1)]], device float* v [[buffer(2)]],
                       constant int2& dk [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
    const int d = dk.x, k = dk.y, c = (int) g.x, t = (int) g.y;
    if (c >= d) return;
    float s = 0;
    for (int j = 0; j < k; ++j) s += bfr(silu_bf(V[((ulong) t * k + j) * d + c]) * bfr(w[t * k + j]));
    v[(ulong) t * d + c] = bfr(s);
}

// RoPE (non-traditional halves, theta = pos * base^(-2i/128)) on q [T][n_head*128] in place and k into the cache;
// v into the cache.  Caches: BF16 [cap][n_kv*128].
kernel void k_rope_kv(device float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                      device ushort* Kc [[buffer(3)]], device ushort* Vc [[buffer(4)]], device const RowInfo* ri [[buffer(5)]],
                      device const float* inv [[buffer(6)]], constant int2& hk [[buffer(7)]], uint2 g [[thread_position_in_grid]]) {
    const int n_head = hk.x, n_kv = hk.y, t = (int) g.y, i = (int) g.x;   // i: head * 64 + pair
    const int pos = ri[t].pos;
    const ulong at = (ulong) (ri[t].kv0 + pos);   // the cache row
    const int head = i / 64, p = i % 64;
    const float th = (float) pos * inv[p];
    const float c = cos(th), s = sin(th);
    if (head < n_head) {
        device float* qh = q + (ulong) t * n_head * ATT_HD + head * ATT_HD;
        const float a = qh[p], b = qh[p + 64];
        qh[p] = bfr(a * c - b * s);
        qh[p + 64] = bfr(b * c + a * s);
    }
    if (head < n_kv) {
        device const float* kh = k + (ulong) t * n_kv * ATT_HD + head * ATT_HD;
        const float a = kh[p], b = kh[p + 64];
        device ushort* kc = Kc + at * n_kv * ATT_HD + head * ATT_HD;
        kc[p] = tobf(a * c - b * s);
        kc[p + 64] = tobf(b * c + a * s);
        device const float* vh = v + (ulong) t * n_kv * ATT_HD + head * ATT_HD;
        device ushort* vc = Vc + at * n_kv * ATT_HD + head * ATT_HD;
        vc[p] = tobf(vh[p]);
        vc[p + 64] = tobf(vh[p + 64]);
    }
}

// The 8-bit KV cache (EngOpts ENG_KV_Q8): int8 [cap][n_kv*128] and one f32 scale per (position, KV head) [cap][n_kv]; a
// value is q * scale, read rounded to BF16 (kv_load4 / kv_load8), as the CUDA kernels do.  k_rope_kv_q8 quantizes the BF16
// values the BF16 cache would hold: scale = max |x| / 127 over the head (1 for a zero head), q = round(x / scale).  As
// k_rope_kv, plus Ks, Vs at buffers 8, 9; threadgroups of 64 threads, one head each (the scale's reduction).
kernel void k_rope_kv_q8(device float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]],
                         device char* Kc [[buffer(3)]], device char* Vc [[buffer(4)]], device const RowInfo* ri [[buffer(5)]],
                         device const float* inv [[buffer(6)]], constant int2& hk [[buffer(7)]], device float* Ks [[buffer(8)]],
                         device float* Vs [[buffer(9)]], uint2 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                         uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[2][2];
    const int n_head = hk.x, n_kv = hk.y, t = (int) g.y, i = (int) g.x;   // i: head * 64 + pair
    const int pos = ri[t].pos;
    const int head = i / 64, p = i % 64;
    const float th = (float) pos * inv[p];
    const float c = cos(th), s = sin(th);
    if (head < n_head) {
        device float* qh = q + (ulong) t * n_head * ATT_HD + head * ATT_HD;
        const float a = qh[p], b = qh[p + 64];
        qh[p] = bfr(a * c - b * s);
        qh[p + 64] = bfr(b * c + a * s);
    }
    if (head >= n_kv) return;   // uniform over the threadgroup
    device const float* kh = k + (ulong) t * n_kv * ATT_HD + head * ATT_HD;
    device const float* vh = v + (ulong) t * n_kv * ATT_HD + head * ATT_HD;
    const float a = kh[p], b = kh[p + 64];
    const float k0 = bfr(a * c - b * s), k1 = bfr(b * c + a * s), v0 = bfr(vh[p]), v1 = bfr(vh[p + 64]);
    const float mk = simd_max(max(fabs(k0), fabs(k1))), mv = simd_max(max(fabs(v0), fabs(v1)));
    if (lane == 0) { red[sgi][0] = mk; red[sgi][1] = mv; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float MK = max(red[0][0], red[1][0]), MV = max(red[0][1], red[1][1]);
    const float sk = MK > 0 ? MK / 127.0f : 1.0f, sv = MV > 0 ? MV / 127.0f : 1.0f;
    const ulong row = (ulong) (ri[t].kv0 + pos) * n_kv + head;
    device char* kc = Kc + row * ATT_HD;
    device char* vc = Vc + row * ATT_HD;
    kc[p] = (char) (int) rint(k0 / sk);
    kc[p + 64] = (char) (int) rint(k1 / sk);
    vc[p] = (char) (int) rint(v0 / sv);
    vc[p + 64] = (char) (int) rint(v1 / sv);
    if (p == 0) { Ks[row] = sk; Vs[row] = sv; }
}

// Cache reads for the attention kernels (templates over the cache element: ushort BF16, char 8-bit with scales s).
// kv_load4: 4 values at c + pos * stride (c at the head and the lane's dims); kv_load8: 8 values as BF16 bits at
// c + row * stride + off.  The scale of (pos, kvh) is s[pos * n_kv + kvh].
static inline float4 bfr4(float4 x) { return float4(bfr(x.x), bfr(x.y), bfr(x.z), bfr(x.w)); }
static inline float4 kv_load4(device const ushort* c, device const float* s, int pos, int stride, int n_kv, int kvh) {
    const uint2 u = *(device const uint2*) (c + (ulong) pos * stride);
    return float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xFFFF0000u), as_type<float>(u.y << 16), as_type<float>(u.y & 0xFFFF0000u));
}
static inline float4 kv_load4(device const char* c, device const float* s, int pos, int stride, int n_kv, int kvh) {
    const char4 u = *(device const char4*) (c + (ulong) pos * stride);
    return bfr4(float4(u) * s[(ulong) pos * n_kv + kvh]);
}
static inline uint4 kv_load8(device const ushort* c, device const float* s, int pos, int stride, int n_kv, int kvh, int off) {
    return *(device const uint4*) (c + (ulong) pos * stride + off);
}
static inline uint4 kv_load8(device const char* c, device const float* s, int pos, int stride, int n_kv, int kvh, int off) {
    const uint2 w = *(device const uint2*) (c + (ulong) pos * stride + off);
    const float sc = s[(ulong) pos * n_kv + kvh];
    const float4 x = float4(as_type<char4>(w.x)) * sc, y = float4(as_type<char4>(w.y)) * sc;
    return uint4((uint) tobf(x.x) | (uint) tobf(x.y) << 16, (uint) tobf(x.z) | (uint) tobf(x.w) << 16,
                 (uint) tobf(y.x) | (uint) tobf(y.y) << 16, (uint) tobf(y.z) | (uint) tobf(y.w) << 16);
}

// Split-key attention: threadgroup (split, kv head, row); simdgroup = one of the 4 query heads; lanes hold 4 dims each.
// Partials [row][head][split] = (m, l, acc[128]).
template <typename KT>
static inline void attn_split(constant AttnArgs& a, device const float* q, device const KT* Kc, device const KT* Vc,
                              device const float* Ks, device const float* Vs, device const RowInfo* ri, device float* part,
                              uint3 tg, uint lane, uint sgi) {
    const int split = (int) tg.x, kvh = (int) tg.y, t = (int) tg.z;
    const int qh = kvh * (a.n_head / a.n_kv) + (int) sgi;
    const int pos = ri[t].pos;
    const int nsr = min(a.n_splits, (pos + a.chunk) / a.chunk);   // this row's split count, as a single AR step computes it
    const int chunk = (pos + nsr) / nsr;                              // ceil((pos + 1) / nsr)
    const int p0 = split * chunk, p1 = min(pos + 1, p0 + chunk);      // splits >= nsr are empty (m = -inf)
    device const float* qr = q + (ulong) t * a.n_head * ATT_HD + qh * ATT_HD + lane * 4;
    const float4 qv = float4(qr[0], qr[1], qr[2], qr[3]) * a.scale;
    float m = -INFINITY, l = 0;
    float4 acc = 0;
    const int stride = a.n_kv * ATT_HD;
    device const KT* kb = Kc + kvh * ATT_HD + lane * 4;
    device const KT* vb = Vc + kvh * ATT_HD + lane * 4;
    // ATT_KU keys per step: their loads in flight together, ATT_KU independent reductions, one online-softmax update
    for (int p = p0; p < p1; p += ATT_KU) {
        float s[ATT_KU];
        float4 vf[ATT_KU];
        for (int u = 0; u < ATT_KU; ++u) {
            const int pu = ri[t].kv0 + min(p + u, p1 - 1);   // the cache row
            const float4 kv = kv_load4(kb, Ks, pu, stride, a.n_kv, kvh);
            vf[u] = kv_load4(vb, Vs, pu, stride, a.n_kv, kvh);
            s[u] = dot(qv, kv);
        }
        float mx = m;
        for (int u = 0; u < ATT_KU; ++u) {
            s[u] = p + u < p1 ? simd_sum(s[u]) : -INFINITY;
            mx = max(mx, s[u]);
        }
        const float cor = exp(m - mx);
        acc *= cor;
        l *= cor;
        for (int u = 0; u < ATT_KU; ++u) {
            const float e = exp(s[u] - mx);
            acc += vf[u] * e;
            l += e;
        }
        m = mx;
    }
    device float* pp = part + (((ulong) t * a.n_head + qh) * a.n_splits + split) * (ATT_HD + 2);
    if (lane == 0) { pp[0] = m; pp[1] = l; }
    pp[2 + lane * 4 + 0] = acc.x; pp[2 + lane * 4 + 1] = acc.y; pp[2 + lane * 4 + 2] = acc.z; pp[2 + lane * 4 + 3] = acc.w;
}
kernel void k_attn(constant AttnArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const ushort* Kc [[buffer(2)]],
                   device const ushort* Vc [[buffer(3)]], device const RowInfo* ri [[buffer(4)]], device float* part [[buffer(5)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                   uint sgi [[simdgroup_index_in_threadgroup]]) {
    attn_split(a, q, Kc, Vc, (device const float*) nullptr, (device const float*) nullptr, ri, part, tg, lane, sgi);
}
// k_attn on the 8-bit cache: scales Ks, Vs at buffers 6, 7
kernel void k_attn_q8(constant AttnArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const char* Kc [[buffer(2)]],
                      device const char* Vc [[buffer(3)]], device const RowInfo* ri [[buffer(4)]], device float* part [[buffer(5)]],
                      device const float* Ks [[buffer(6)]], device const float* Vs [[buffer(7)]],
                      uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
                      uint sgi [[simdgroup_index_in_threadgroup]]) {
    attn_split(a, q, Kc, Vc, Ks, Vs, ri, part, tg, lane, sgi);
}

// Reduce the splits, then the softplus output gate: o[t][h*128+d] = bf16(bf16(attn) * bf16(softplus_ln2(g)))
kernel void k_attn_reduce(constant AttnArgs& a [[buffer(0)]], device const float* part [[buffer(1)]], device const float* g [[buffer(2)]],
                          device float* o [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    const int h = (int) tg.x, t = (int) tg.y;
    device const float* pp = part + ((ulong) t * a.n_head + h) * a.n_splits * (ATT_HD + 2);
    float m = -INFINITY;
    for (int s = 0; s < a.n_splits; ++s) m = max(m, pp[s * (ATT_HD + 2)]);
    float l = 0, acc = 0;
    for (int s = 0; s < a.n_splits; ++s) {
        const float ms = pp[s * (ATT_HD + 2)];
        if (ms == -INFINITY) continue;
        const float c = exp(ms - m);
        l += pp[s * (ATT_HD + 2) + 1] * c;
        acc += pp[s * (ATT_HD + 2) + 2 + d] * c;
    }
    const float att = bfr(acc / l);
    const ulong i = (ulong) t * a.n_head * ATT_HD + h * ATT_HD + d;
    const float gx = g[i] * 0.69314718055994531f;   // softplus(x, beta = ln 2) = logaddexp(x ln2, 0) / ln2, in f32
    const float sp = bfr((max(gx, 0.0f) + log(1.0f + exp(-fabs(gx)))) / 0.69314718055994531f);
    o[i] = bfr(att * sp);
}

// ---- MLA (TransMLA): per-head maps, RoPE + latent cache write, latent attention ---------------------------------------

// Per-head maps (q_rope_mix, q_lat, v_up): simdgroup = (output row o, head h, token t); lanes stride the input; f32 sums.
// y[t][h][o] = bf16(W_h[o] . x), or with the gate bf16(bf16(W_h[o] . x) * bf16(softplus_ln2(g[t][h][o]))).
// Grid (O * 32, H, T), threadgroups of 256.
// A stacked [H][O][I] SEED4P4 (a.w 0) or SEED6P8 (a.w 1) per-head map decoded to BF16 (bf16(R32 2^e isum)), a block
// of 8 per thread: the expanded prompt path's GEMM operands, decoded once per forward instead of once per key tile.
kernel void k_heads_deq(device const uchar* W [[buffer(0)]], device const ushort* S [[buffer(1)]], device const ushort* B [[buffer(2)]],
                        device const uint* G [[buffer(3)]], device const uchar* EN [[buffer(4)]], device ushort* out [[buffer(5)]],
                        constant int4& a [[buffer(6)]], uint b [[thread_position_in_grid]]) {
    const int O = a.x, I = a.y, nbk = I / 8;
    if ((int) b >= a.z * O * nbk) return;
    const ulong row = b / (uint) nbk;
    const int j = (int) (b % (uint) nbk);
    float w[8];
    if (a.w) seed6p8_block(W, S, G, row, I, j, ((device const int*) B)[row / (ulong) O], EN, w);
    else seed4p4_block(W, S, G, row, I, j, ((device const int*) B)[row / (ulong) O], EN, w);
    for (int i = 0; i < 8; ++i) out[(ulong) b * 8 + i] = tobf(w[i]);
}
kernel void k_heads_mv(constant HmvArgs& a [[buffer(0)]], device const uchar* W [[buffer(1)]], device const float* x [[buffer(2)]],
                       device const float* g [[buffer(3)]], device float* y [[buffer(4)]], device const ushort* S [[buffer(5)]],
                       device const ushort* B [[buffer(6)]], device const uint* G [[buffer(7)]], device const uchar* EN [[buffer(8)]],
                       uint3 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    // a.pad[0] = R rows per simdgroup (1, 2 or 8: the host's choice by shape), 32 / R lanes a row
    const int R = a.pad[0] > 1 ? a.pad[0] : 1, L = 32 / R, h = (int) gid.y, t = (int) gid.z;
    const int o = (int) gid.x / 32 * R + (int) lane / L, ln = (int) lane % L;
    if ((int) gid.x / 32 * R >= a.O) return;   // whole simdgroups (the grid's x is a multiple of 32)
    // FC_FMT: BF16, Q8, Q4, SEED4P4 or SEED6P8 (stacked [H][O][I]: row h * O + o; the seeds' exponent bias per head in B)
    const int eb = FC_FMT == MF_SEED4P4 || FC_FMT == MF_SEED6P8 ? ((device const int*) B)[h] : 0;
    device const float* xv = x + (ulong) t * a.xs + (ulong) h * a.hs;
    const ulong row = (ulong) h * a.O + (ulong) min(o, a.O - 1);
    float s;
    if (R == 1) s = simd_sum(mv_lane(FC_FMT, W, S, B, G, eb, row, a.I, xv, lane, EN));
    else {   // L lanes a row, a block of 8 each step (SEED4P4 and Q4 in mv_lane's forms)
        float acc = 0;
        for (int j = ln; j < a.I / 8; j += L) {
            float w[8];
            if (FC_FMT == MF_SEED4P4) {   // mv_lane's form: two-op states, the block's dot without its weights
                const ulong k = row * (ulong) (a.I / 8) + j;
                const uint sd = ((device const ushort*) W)[k];
                float v[32];
                seed4_states(sd, G[sd], v);
                device const float4* xr = (device const float4*) (xv + (ulong) j * 8);
                const float4 x0 = xr[0], x1 = xr[1];
                acc += seed4_dot(v, (uint) S[k], eb + seed4_ecode(EN, k), x0, x1, -1.5f * dot(x0 + x1, float4(1.0f))) * (1.0f / 32767.0f);
                continue;
            } else if (FC_FMT == MF_SEED6P8) {   // mv_lane's P = 8 form
                const ulong k = row * (ulong) (a.I / 8) + j;
                float v[64];
                seed8_states(((device const ushort*) W)[k], G, v);
                device const float4* xr = (device const float4*) (xv + (ulong) j * 8);
                const float4 x0 = xr[0], x1 = xr[1];
                acc += seed8_dot(v, ((device const uint*) S)[k], eb + seed4_ecode(EN, k), x0, x1, -1.5f * dot(x0 + x1, float4(1.0f))) *
                       (1.0f / 32767.0f);
                continue;
            } else if (FC_FMT == MF_Q4) {
                const ulong gi = (row * (ulong) a.I + (ulong) j * 8) / 64;
                const float sc = bf(S[gi]), bi = bf(B[gi]);
                const uint q = ((device const uint*) W)[(row * (ulong) a.I) / 8 + (ulong) j];
                for (int i = 0; i < 8; ++i) w[i] = sc * (float) ((q >> (4 * i)) & 15u) + bi;
            } else wblock(FC_FMT, W, S, B, row, a.I, j, w);
            device const float4* xr = (device const float4*) (xv + (ulong) j * 8);
            acc += dot(float4(w[0], w[1], w[2], w[3]), xr[0]) + dot(float4(w[4], w[5], w[6], w[7]), xr[1]);
        }
        for (ushort d = 1; d < (ushort) L; d <<= 1) acc += simd_shuffle_xor(acc, d);
        s = acc;
    }
    if (ln == 0 && o < a.O) {
        const ulong yi = ((ulong) t * a.H + h) * a.O + o;
        float v = bfr(s);
        if (a.gate) v = bfr(v * softplus_ln2_bf(g[yi]));
        y[yi] = v;
    }
}

// One value v at dim d of a quantized cache row (len values, the first lh FP8, the rest FP4), stored when `store`: each
// block of consecutive dims lies on consecutive lanes of one simdgroup (d % 32 == lane), which share its max |v|; FP4
// pairs lanes into bytes.  Every lane of the simdgroup must call it (the shuffles run in uniform control flow).
static inline void kv_put(bool store, int lh, float v, device uchar* C, device uchar* S, ulong row, int len, int d) {
    float a16 = fabs(v);
    for (ushort o = 1; o < 16; o <<= 1) a16 = max(a16, simd_shuffle_xor(a16, o));
    const float a32 = max(a16, simd_shuffle_xor(a16, 16));
    const bool f8 = d < lh;
    const uint sc = f8 ? fp8_scale(a32) : fp4_scale(a16);
    const float s4 = e4m3_dec(sc);
    const uint c = f8 ? e4m3_enc_abs(fabs(v) * e8m0_dec(254u - sc)) : (s4 > 0 ? e2m1_enc(v, s4) : 0u);
    const uint hi = simd_shuffle_xor(c, 1);
    if (!store) return;
    C += row * kv_rowb(lh, len);
    S += row * kv_srowb(lh, len);
    const int e = d - lh;
    if (f8) {
        C[d] = (uchar) (c && v < 0 ? c | 0x80u : c);
        if (d % 32 == 0) S[d / 32] = (uchar) sc;
    } else {
        if (!(e & 1)) C[lh + e / 2] = (uchar) (c | hi << 4);
        if (e % 16 == 0) S[lh / 32 + e / 16] = (uchar) sc;
    }
}

// RoPE (as k_rope_kv) on the query RoPE parts qr [T][n_head][128] in place and on the RoPE key kr [T][128] into
// Kc [cap][128]; the latent c [T][r] into Vc [cap][r].  Grid (max((n_head + 1) * 64, r), T), threadgroups of 64.  Cache
// format FC_FMT: BF16, or FP8 / FP4 (the BF16 values' codes, block scales into Ks / Vs; r a multiple of 32).
// kv_put for an all-FP8 row (the RoPE key): the same codes, its own address arithmetic
static inline void kv_put8(bool store, float v, device uchar* C, device uchar* S, ulong row, int len, int d) {
    float amax = fabs(v);
    for (ushort o = 1; o < 32; o <<= 1) amax = max(amax, simd_shuffle_xor(amax, o));
    const uint sc = fp8_scale(amax), c = e4m3_enc_abs(fabs(v) * e8m0_dec(254u - sc));
    if (!store) return;
    C[row * (ulong) len + d] = (uchar) (c && v < 0 ? c | 0x80u : c);
    if (d % 32 == 0) S[row * (ulong) (len / 32) + d / 32] = (uchar) sc;
}
kernel void k_mla_rope(constant MlaArgs& a [[buffer(0)]], device float* qr [[buffer(1)]], device const float* kr [[buffer(2)]],
                       device const float* c [[buffer(3)]], device uchar* Kc [[buffer(4)]], device uchar* Vc [[buffer(5)]],
                       device const RowInfo* ri [[buffer(6)]], device const float* inv [[buffer(7)]],
                       device uchar* Ks [[buffer(8)]], device uchar* Vs [[buffer(9)]],
                       uint2 g [[thread_position_in_grid]]) {
    const int i = (int) g.x, t = (int) g.y, pos = ri[t].pos;
    const ulong at = (ulong) (ri[t].kv0 + pos);   // the cache row
    const int head = i / 64, p = i % 64;
    const bool lat = i < a.r, key = head == a.n_head;
    const float v = lat ? c[(ulong) t * a.r + i] : 0.0f;
    float y0 = 0, y1 = 0;
    if (head <= a.n_head) {
        const float th = (float) pos * inv[p];
        const float cs = cos(th), sn = sin(th);
        device float* qh = qr + ((ulong) t * a.n_head + (key ? 0 : head)) * ATT_HD;
        device const float* xh = key ? kr + (ulong) t * ATT_HD : (device const float*) qh;
        const float x0 = xh[p], x1 = xh[p + 64];
        y0 = x0 * cs - x1 * sn;
        y1 = x1 * cs + x0 * sn;
        if (!key) { qh[p] = bfr(y0); qh[p + 64] = bfr(y1); }
    }
    if (FC_FMT == KV_BF16) {
        if (lat) ((device ushort*) Vc)[at * a.r + i] = tobf(v);
        if (key) { ((device ushort*) Kc)[at * ATT_HD + p] = tobf(y0); ((device ushort*) Kc)[at * ATT_HD + p + 64] = tobf(y1); }
    } else {   // every thread runs the shuffles; latent dims d = i, RoPE-key dims p and p + 64
        kv_put(lat, LLH(a.r), bfr(v), Vc, Vs, at, a.r, i);
        kv_put8(key, bfr(y0), Kc, Ks, at, ATT_HD, p);
        kv_put8(key, bfr(y1), Kc, Ks, at, ATT_HD, p + 64);
    }
}

// Latent attention (absorbed MLA: multi-query over the shared latent), flash style.  Threadgroup (split, block of
// MLAF_Q query heads, row) of MLAF_SG simdgroups; per tile of MLAF_K keys:
//   scores: S^T [keys][heads] = [c, k] . [ql, qr]^T, MLAF_DC dims at a time: the tile's keys and the heads' queries are
//     staged in threadgroup memory (BF16: the cache's values, decoded when FC_FMT is FP8 / FP4, and the queries'
//     values, exact) with wide loads, then each
//     simdgroup accumulates its 8x8 block on f32 simdgroup matrices; times the scale;
//   online softmax per head (max, rescale, row sums) in threadgroup memory;
//   O [heads][r] += P c, MLAF_DC latent dims at a time (staged as the keys): each simdgroup one 8-dim column, both
//     head blocks.
// Keys past the row's split are zero / -inf.  Sums f32.  n_splits > 1: partials [row][head][split] = (m, l, acc[r])
// for k_mla_reduce; n_splits == 1: olat[row][head] = bf16(acc / l).
static inline float2 bf2(uint w) { return float2(as_type<float>(w << 16), as_type<float>(w & 0xFFFF0000u)); }
// Staging (N8 / N4 > 0: a full chunk, the index arithmetic constant-folded; 0: n8 / n4 at run time).  Keys k0.. (before
// p1) of the slot at kv0, dims [dc, dc + 8 n8) of [c, k] into Kt [key][dim]; queries of heads h0.. into Qt [dim][head].
template <int N8>
static inline void mlaf_keys(threadgroup ushort* Kt, device const uchar* Kc, device const uchar* Vc, device const uchar* Ks,
                             device const uchar* Vs, ulong kv0, int k0, int p1, int dc, int r, int n8, uint tid) {
    const int n = N8 > 0 ? N8 : n8;
    for (int e = (int) tid; e < MLAF_K * n; e += 32 * MLAF_SG) {
        const int key = e / n, d = dc + (e % n) * 8, p = k0 + key;
        uint4 w = 0;
        if (p < p1) w = d < r ? kv_get8(FC_FMT, LLH(r), Vc, Vs, kv0 + (ulong) p, r, d) : kv_get8(FC_FMT, KLH, Kc, Ks, kv0 + (ulong) p, ATT_HD, d - r);
        threadgroup uint* kt = (threadgroup uint*) (Kt + key * MLAF_KLD + d - dc);
        kt[0] = w.x; kt[1] = w.y; kt[2] = w.z; kt[3] = w.w;
    }
}
template <int N4>
static inline void mlaf_queries(threadgroup ushort* Qt, device const float* ql, device const float* qr, int t, int H, int h0,
                                int dc, int r, int n4, uint tid) {
    const int n = N4 > 0 ? N4 : n4;
    for (int e = (int) tid; e < MLAF_Q * n; e += 32 * MLAF_SG) {
        const int q = e / n, d = dc + (e % n) * 4, h = h0 + q;
        float4 v = 0;
        if (h < H) v = *(device const float4*) (d < r ? ql + ((ulong) t * H + h) * r + d : qr + ((ulong) t * H + h) * ATT_HD + d - r);
        for (int j = 0; j < 4; ++j) Qt[(d - dc + j) * MLAF_QLD + q] = tobf(v[j]);
    }
}
kernel void k_mla_attn(constant MlaArgs& a [[buffer(0)]], device const float* ql [[buffer(1)]], device const float* qr [[buffer(2)]],
                       device const uchar* Kc [[buffer(3)]], device const uchar* Vc [[buffer(4)]],
                       device const RowInfo* ri [[buffer(5)]], device float* out [[buffer(6)]],
                       device const uchar* Ks [[buffer(7)]], device const uchar* Vs [[buffer(8)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Kbuf[MLAF_K * MLAF_KLD / 2];     // staged keys (or latent values) [key][dim] (BF16); at the end, out
    threadgroup ushort* Kt = (threadgroup ushort*) Kbuf;
    threadgroup ushort Qt[MLAF_DC * MLAF_QLD];         // staged queries [dim][head]
    threadgroup float St[MLAF_K][MLAF_Q];              // scores
    threadgroup float P[MLAF_Q][MLAF_K];               // probabilities (against the running max)
    threadgroup float ms[MLAF_Q], ls[MLAF_Q];           // running max, row sum
    threadgroup float Dc[MLAF_Q / 8][8 * 8];            // this tile's rescale as diagonal matrices (O = Dc O)
    const int split = (int) tg.x, h0 = (int) tg.y * MLAF_Q, t = (int) tg.z, r = a.r, H = a.n_head, D = r + ATT_HD;
    const int pos = ri[t].pos;
    const ulong kv0 = (ulong) ri[t].kv0;
    const int nsr = min(a.n_splits, (pos + a.chunk) / a.chunk);   // this row's split count, as a single AR step's
    const int chunk = (pos + nsr) / nsr;
    const int p0 = split * chunk, p1 = min(pos + 1, p0 + chunk);
    const int qid = (int) lane / 4, fm = (qid & 4) + ((int) lane / 2) % 4, fn = (qid & 2) * 2 + ((int) lane % 2) * 2;
    // scores: simdgroup sgi takes key block sgi % 8 against both head blocks (one key operand load for two products) over
    // its half (sgi / 8) of each chunk's dims; the halves are summed in St
    const int kb = (int) sgi % (MLAF_K / 8), hf = (int) sgi / (MLAF_K / 8);
    const int nc = FC_T > 0 ? FC_T : (r + MLAF_DC - 1) / MLAF_DC;   // latent chunks (pipelines specialize it: the
                                                                     // unused accumulators then take no registers)
    simdgroup_float8x8 O[MLAF_OC][2];                   // column sgi*8 of each latent chunk, both head blocks
#pragma unroll
    for (int i = 0; i < MLAF_OC; ++i) { O[i][0] = simdgroup_float8x8(0.0f); O[i][1] = simdgroup_float8x8(0.0f); }
    if (tid < MLAF_Q) { ms[tid] = -INFINITY; ls[tid] = 0; }
    for (int e = (int) tid; e < MLAF_Q * 8; e += 32 * MLAF_SG) Dc[e / 64][e % 64] = 0;
    for (int k0 = p0; k0 < p1; k0 += MLAF_K) {
        simdgroup_float8x8 S0 = simdgroup_float8x8(0.0f), S1 = simdgroup_float8x8(0.0f);
        for (int dc = 0; dc < D; dc += MLAF_DC) {
            const int cw = min(MLAF_DC, D - dc);
            threadgroup_barrier(mem_flags::mem_threadgroup);   // the previous chunk's (or phase's) readers are done
            if (cw == MLAF_DC) {
                mlaf_keys<MLAF_DC / 8>(Kt, Kc, Vc, Ks, Vs, kv0, k0, p1, dc, r, 0, tid);
                mlaf_queries<MLAF_DC / 4>(Qt, ql, qr, t, H, h0, dc, r, 0, tid);
            } else {
                mlaf_keys<0>(Kt, Kc, Vc, Ks, Vs, kv0, k0, p1, dc, r, cw / 8, tid);
                mlaf_queries<0>(Qt, ql, qr, t, H, h0, dc, r, cw / 4, tid);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const int nb = cw / 8, ib0 = hf ? (nb + 1) / 2 : 0, ib1 = hf ? nb : (nb + 1) / 2;
            for (int i = ib0; i < ib1; ++i) {
                simdgroup_bfloat8x8 A, B0, B1;   // A: keys kb*8.., dims i*8..; B: dims i*8.., heads 0.. / 8.. (exact products, f32 sums)
                simdgroup_load(A, (threadgroup const bfloat*) Kt + kb * 8 * MLAF_KLD + i * 8, MLAF_KLD);
                simdgroup_load(B0, (threadgroup const bfloat*) Qt + i * 8 * MLAF_QLD, MLAF_QLD);
                simdgroup_load(B1, (threadgroup const bfloat*) Qt + i * 8 * MLAF_QLD + 8, MLAF_QLD);
                simdgroup_multiply_accumulate(S0, A, B0, S0);
                simdgroup_multiply_accumulate(S1, A, B1, S1);
            }
        }
        if (hf) { simdgroup_store(S0, &St[kb * 8][0], MLAF_Q); simdgroup_store(S1, &St[kb * 8][8], MLAF_Q); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!hf) {   // St = the second half's + this half's (the identity times St, accumulated)
            const simdgroup_float8x8 I = simdgroup_float8x8(1.0f);
            simdgroup_float8x8 T0, T1;
            simdgroup_load(T0, &St[kb * 8][0], MLAF_Q);
            simdgroup_load(T1, &St[kb * 8][8], MLAF_Q);
            simdgroup_multiply_accumulate(S0, I, T0, S0);
            simdgroup_multiply_accumulate(S1, I, T1, S1);
            simdgroup_store(S0, &St[kb * 8][0], MLAF_Q);
            simdgroup_store(S1, &St[kb * 8][8], MLAF_Q);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // online softmax: thread (head q, keys k + MLAF_TH u); a head's threads are adjacent lanes
            const int q = (int) tid / MLAF_TH, k = (int) tid % MLAF_TH;
            float s[MLAF_K / MLAF_TH], mx = -INFINITY;
            for (int u = 0; u < MLAF_K / MLAF_TH; ++u) {
                const int kk = k + MLAF_TH * u;
                s[u] = k0 + kk < p1 ? St[kk][q] * a.scale : -INFINITY;
                mx = max(mx, s[u]);
            }
            for (ushort o = 1; o < MLAF_TH; o <<= 1) mx = max(mx, simd_shuffle_xor(mx, o));
            const float mo = ms[q], mn = max(mo, mx);
            float sum = 0;
            for (int u = 0; u < MLAF_K / MLAF_TH; ++u) {
                const float pv = s[u] == -INFINITY ? 0.0f : exp(s[u] - mn);
                P[q][k + MLAF_TH * u] = pv;
                sum += pv;
            }
            for (ushort o = 1; o < MLAF_TH; o <<= 1) sum += simd_shuffle_xor(sum, o);
            if (k == 0) {
                const float c = mn == -INFINITY ? 1.0f : exp(mo - mn);
                Dc[q / 8][(q % 8) * 9] = c;
                ls[q] = ls[q] * c + sum;
                ms[q] = mn;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   // O = O * rescale + P c, a latent chunk at a time
            simdgroup_float8x8 D0, D1;
            simdgroup_load(D0, Dc[0], 8);
            simdgroup_load(D1, Dc[1], 8);
#pragma unroll
            for (int ci = 0; ci < MLAF_OC; ++ci) {
                if (ci >= nc) continue;
                simdgroup_multiply(O[ci][0], D0, O[ci][0]);
                simdgroup_multiply(O[ci][1], D1, O[ci][1]);
            }
            const int nkb = min(MLAF_K / 8, (p1 - k0 + 7) / 8);
#pragma unroll
            for (int ci = 0; ci < MLAF_OC; ++ci) {
                if (ci >= nc) continue;   // unrolled with constant indices: O stays in registers
                const int dc = ci * MLAF_DC, cw = min(MLAF_DC, r - dc);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (cw == MLAF_DC) mlaf_keys<MLAF_DC / 8>(Kt, Kc, Vc, Ks, Vs, kv0, k0, p1, dc, r, 0, tid);   // latent values (d < r)
                else mlaf_keys<0>(Kt, Kc, Vc, Ks, Vs, kv0, k0, p1, dc, r, cw / 8, tid);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if ((int) sgi * 8 >= cw) continue;
                for (int j = 0; j < nkb; ++j) {
                    simdgroup_float8x8 A0, A1, B;   // P: heads x keys j*8..; B: keys j*8 + fm, dims sgi*8 + fn, +1
                    simdgroup_load(A0, &P[0][j * 8], MLAF_K);
                    simdgroup_load(A1, &P[8][j * 8], MLAF_K);
                    thread auto& be = B.thread_elements();
                    const float2 v = bf2(*(threadgroup const uint*) (Kt + (j * 8 + fm) * MLAF_KLD + (int) sgi * 8 + fn));
                    be[0] = v.x; be[1] = v.y;
                    simdgroup_multiply_accumulate(O[ci][0], A0, B, O[ci][0]);
                    simdgroup_multiply_accumulate(O[ci][1], A1, B, O[ci][1]);
                }
            }
        }
    }
    // out: through threadgroup memory, a latent chunk at a time
    threadgroup float* Ot = Kbuf;   // [MLAF_Q][MLAF_DC]
#pragma unroll
    for (int ci = 0; ci < MLAF_OC; ++ci) {
        if (ci >= nc) continue;
        const int dc = ci * MLAF_DC, cw = min(MLAF_DC, r - dc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((int) sgi * 8 < cw) {
            simdgroup_store(O[ci][0], Ot + (int) sgi * 8, MLAF_DC);
            simdgroup_store(O[ci][1], Ot + 8 * MLAF_DC + (int) sgi * 8, MLAF_DC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int) tid; e < MLAF_Q * cw; e += 32 * MLAF_SG) {
            const int q = e / cw, d = e % cw, h = h0 + q;
            if (h >= H) continue;
            const float v = Ot[q * MLAF_DC + d];
            if (a.n_splits == 1) out[((ulong) t * H + h) * r + dc + d] = bfr(v / ls[q]);
            else out[(((ulong) t * H + h) * a.n_splits + split) * (r + 2) + 2 + dc + d] = v;
        }
    }
    if (a.n_splits > 1 && tid < MLAF_Q && h0 + (int) tid < H) {
        device float* pp = out + (((ulong) t * H + h0 + tid) * a.n_splits + split) * (r + 2);
        pp[0] = ms[tid];
        pp[1] = ls[tid];
    }
}

// MLA capture: s[c] += sum over T rows of x[t * xs + c]^2, c < n (one thread per column)
kernel void k_sumsq(device const float* x [[buffer(0)]], device float* s [[buffer(1)]], constant int4& a [[buffer(2)]],
                    uint c [[thread_position_in_grid]]) {
    if ((int) c >= a.x) return;
    float acc = 0;
    for (int t = 0; t < a.y; ++t) { const float v = x[(ulong) t * a.z + c]; acc += v * v; }
    s[c] += acc;
}
// MLA capture for GPTQ: H[b] += X_b^T X_b, X_b the T x D block b of x (row stride a.z, blocks a.w apart; a.x = D, a.y = T),
// on H's lower-triangle 32 x 32 tiles.  Grid (D / 32, D / 32, blocks) threadgroups of 4 simdgroups (16 x 16 each); 8 rows
// at a time staged in threadgroup memory (rows past T as zeros), f32 simdgroup matrices accumulating onto H.
kernel void k_xtx(device const float* x [[buffer(0)]], device float* H [[buffer(1)]], constant int4& a [[buffer(2)]],
                  uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (tg.y < tg.x) return;   // upper tiles: the transposes of lower ones
    const int D = a.x, T = a.y, i0 = (int) tg.y * 32, j0 = (int) tg.x * 32, si = (sg / 2) * 16, sj = (sg % 2) * 16;
    device const float* xb = x + (ulong) tg.z * (ulong) a.w;
    device float* hb = H + (ulong) tg.z * (ulong) D * (ulong) D;
    threadgroup float xi[8 * 32], xj[8 * 32];
    simdgroup_float8x8 acc[2][2], A[2], B[2];
    for (short u = 0; u < 2; ++u)
        for (short v = 0; v < 2; ++v) simdgroup_load(acc[u][v], hb + (ulong) (i0 + si + 8 * u) * D + j0 + sj + 8 * v, (ulong) D);
    for (int t0 = 0; t0 < T; t0 += 8) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k = tid; k < 256; k += 128) {
            const int t = t0 + k / 32, c = k % 32;
            xi[k] = t < T ? xb[(ulong) t * a.z + i0 + c] : 0.0f;
            xj[k] = t < T ? xb[(ulong) t * a.z + j0 + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short u = 0; u < 2; ++u) {
            simdgroup_load(A[u], xi + si + 8 * u, 32, ulong2(0, 0), true);   // A[i][t] = x[t][i]
            simdgroup_load(B[u], xj + sj + 8 * u, 32);
        }
        for (short u = 0; u < 2; ++u)
            for (short v = 0; v < 2; ++v) simdgroup_multiply_accumulate(acc[u][v], A[u], B[v], acc[u][v]);
    }
    for (short u = 0; u < 2; ++u)
        for (short v = 0; v < 2; ++v) simdgroup_store(acc[u][v], hb + (ulong) (i0 + si + 8 * u) * D + j0 + sj + 8 * v, (ulong) D);
}
// y[i] = value i of n / len consecutive cache rows of len values (format FC_FMT; codes C and scales S from the first row)
kernel void k_kv_f32(device const uchar* C [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                     device const uchar* S [[buffer(3)]], constant uint& len [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = kv_get(FC_FMT, LLH((int) len), C, S, i / len, (int) len, (int) (i % len));
}
// y[i] = x[i]: BF16 to f32 (the MLA prompt path's latent rows from the cache)
kernel void k_bf16_f32(device const ushort* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = bf(x[i]);
}

// MLA prompt attention with the latent expanded per head (MlpArgs; prompt rows): o[t][h] = softmax_p(scale (qn_th .
// kn_ph + qr_th . kr_p)) vn_ph over p <= pos_t, kn / vn the per-head expansions of the latent (rows kb0 .. of this pass),
// kr the cache's RoPE keys.  About a sixth of the absorbed kernel's arithmetic for long prompts.  Threadgroup (block of
// 8 MLPF_R rows of one slot, head) of MLPF_R simdgroups of 8 rows; per tile of MLPF_BK keys: S = qn kn^T + qr kr^T on f32
// simdgroup matrices (qn, qr, kn, vn and the state from device memory; the tile's RoPE keys staged as f32), online
// softmax per row (4 lanes a row), O = D O + P vn with D the rescale as a diagonal matrix (accumulators only through
// simdgroup-matrix operations).  Keys past the row or the pass: -inf.  The state (m, l, O) carries across passes; the
// last writes o = bf16(bf16(O / l) * bf16(softplus_ln2(g))).  qn, qr and the state have rows up to the block's end, kn
// and vn MLPF_BK rows past the pass (finite: zero or earlier values).
kernel void k_mla_prefill(constant MlpArgs& a [[buffer(0)]], device const float* qn [[buffer(1)]], device const float* qr [[buffer(2)]],
                          device const float* kn [[buffer(3)]], device const float* vn [[buffer(4)]],
                          device const uchar* Kc [[buffer(5)]], device const RowInfo* ri [[buffer(6)]], device float* ms [[buffer(7)]],
                          device float* ls [[buffer(8)]], device float* Os [[buffer(9)]], device const float* g [[buffer(10)]],
                          device float* o [[buffer(11)]], device const uchar* Ks [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float Kr[MLPF_BK * ATT_HD];            // the tile's RoPE keys [key][dim]
    threadgroup float Pt[MLPF_R][8 * MLPF_BK];         // per simdgroup: scores, then probabilities [row][key]
    threadgroup float Dg[MLPF_R][64];                  // per simdgroup: the rescale (diagonal), at the end 1 / l
    threadgroup float Ox[MLPF_R][8 * ATT_HD];          // per simdgroup: output staging [row][dim]
    const int H = a.n_head, T = a.T, h = (int) tg.y, ld = H * ATT_HD;
    const int t0 = (int) tg.x * 8 * MLPF_R, r0 = t0 + (int) sgi * 8;
    const ulong kv0 = (ulong) ri[t0].kv0;              // the block's rows: one slot
    const int kend = min(a.kb1, ri[min(t0 + 8 * MLPF_R, T) - 1].pos + 1);
    const int sr = (int) lane / 4, sk = (int) lane % 4;   // softmax: the lane's row and keys sk, sk + 4, ..
    const int row = r0 + sr, mypos = ri[min(row, T - 1)].pos;
    device const float* qnb = qn + (ulong) r0 * ld + h * ATT_HD;
    device const float* qrb = qr + (ulong) r0 * ld + h * ATT_HD;
    simdgroup_float8x8 O[ATT_HD / 8];
    float m = -INFINITY, l = 0;
    if (a.first) {
        for (int j = 0; j < ATT_HD / 8; ++j) O[j] = simdgroup_float8x8(0.0f);
    } else {
        for (int j = 0; j < ATT_HD / 8; ++j) simdgroup_load(O[j], Os + (ulong) r0 * ld + h * ATT_HD + j * 8, ld);
        m = ms[(ulong) row * H + h];
        l = ls[(ulong) row * H + h];
    }
    Dg[sgi][lane] = 0;
    Dg[sgi][lane + 32] = 0;
    for (int k0 = a.kb0; k0 < kend; k0 += MLPF_BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int) tid; e < MLPF_BK * ATT_HD; e += 32 * MLPF_R) {
            const int p = k0 + e / ATT_HD;
            Kr[e] = p < kend ? kv_get(FC_FMT, KLH, Kc, Ks, kv0 + (ulong) p, ATT_HD, e % ATT_HD) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float* knb = kn + (ulong) (k0 - a.kb0) * ld + h * ATT_HD;
        device const float* vnb = vn + (ulong) (k0 - a.kb0) * ld + h * ATT_HD;
        for (int kb = 0; kb < MLPF_BK / 8; ++kb) {   // scores of keys kb*8..: rows x keys
            simdgroup_float8x8 S = simdgroup_float8x8(0.0f);
            for (int i = 0; i < ATT_HD / 8; ++i) {
                simdgroup_float8x8 A, B;
                simdgroup_load(A, qnb + i * 8, ld);
                simdgroup_load(B, knb + (ulong) kb * 8 * ld + i * 8, ld, ulong2(0, 0), true);   // kn^T: dims x keys
                simdgroup_multiply_accumulate(S, A, B, S);
                simdgroup_load(A, qrb + i * 8, ld);
                simdgroup_load(B, Kr + kb * 8 * ATT_HD + i * 8, ATT_HD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(S, A, B, S);
            }
            simdgroup_store(S, &Pt[sgi][kb * 8], MLPF_BK);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {   // online softmax of the lane's row
            float s[MLPF_BK / 4], mx = -INFINITY;
            for (int u = 0; u < MLPF_BK / 4; ++u) {
                const int k = sk + 4 * u, p = k0 + k;
                s[u] = p < kend && p <= mypos ? Pt[sgi][sr * MLPF_BK + k] * a.scale : -INFINITY;
                mx = max(mx, s[u]);
            }
            mx = max(mx, simd_shuffle_xor(mx, (ushort) 1));
            mx = max(mx, simd_shuffle_xor(mx, (ushort) 2));
            const float mn = max(m, mx), c = mn == -INFINITY ? 1.0f : exp(m - mn);
            float sum = 0;
            for (int u = 0; u < MLPF_BK / 4; ++u) {
                const float pv = s[u] == -INFINITY ? 0.0f : exp(s[u] - mn);
                Pt[sgi][sr * MLPF_BK + sk + 4 * u] = pv;
                sum += pv;
            }
            sum += simd_shuffle_xor(sum, (ushort) 1);
            sum += simd_shuffle_xor(sum, (ushort) 2);
            l = l * c + sum;
            m = mn;
            if (sk == 0) Dg[sgi][sr * 9] = c;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 Dm;
        simdgroup_load(Dm, Dg[sgi], 8);
        for (int j = 0; j < ATT_HD / 8; ++j) simdgroup_multiply(O[j], Dm, O[j]);
        for (int kb = 0; kb < MLPF_BK / 8; ++kb) {
            simdgroup_float8x8 A;
            simdgroup_load(A, &Pt[sgi][kb * 8], MLPF_BK);
            for (int j = 0; j < ATT_HD / 8; ++j) {
                simdgroup_float8x8 B;
                simdgroup_load(B, vnb + (ulong) kb * 8 * ld + j * 8, ld);
                simdgroup_multiply_accumulate(O[j], A, B, O[j]);
            }
        }
    }
    // the state, or the gated output; through threadgroup memory (rows past T are not written)
    for (int j = 0; j < ATT_HD / 8; ++j) simdgroup_store(O[j], &Ox[sgi][j * 8], ATT_HD);
    if (sk == 0) Dg[sgi][sr] = l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (!a.last && sk == 0 && row < T) { ms[(ulong) row * H + h] = m; ls[(ulong) row * H + h] = l; }
    for (int e = (int) lane; e < 8 * ATT_HD; e += 32) {
        const int rr = e / ATT_HD, t = r0 + rr;
        if (t >= T) continue;
        const ulong i = (ulong) t * ld + h * ATT_HD + e % ATT_HD;
        if (a.last) o[i] = bfr(bfr(Ox[sgi][e] / Dg[sgi][rr]) * softplus_ln2_bf(g[i]));
        else Os[i] = Ox[sgi][e];
    }
}

// Reduce the latent attention's splits: olat[t][h][d] = bf16(acc / l).  Threadgroup (head, row); threads stride d.
kernel void k_mla_reduce(constant MlaArgs& a [[buffer(0)]], device const float* part [[buffer(1)]], device float* olat [[buffer(2)]],
                         uint2 tg [[threadgroup_position_in_grid]], uint d0 [[thread_index_in_threadgroup]],
                         uint2 nt2 [[threads_per_threadgroup]]) {
    const uint nt = nt2.x;
    const int h = (int) tg.x, t = (int) tg.y, r = a.r, w = r + 2;
    device const float* pp = part + ((ulong) t * a.n_head + h) * a.n_splits * w;
    float m = -INFINITY;
    for (int s = 0; s < a.n_splits; ++s) m = max(m, pp[s * w]);
    for (int d = (int) d0; d < r; d += (int) nt) {
        float l = 0, acc = 0;
        for (int s = 0; s < a.n_splits; ++s) {
            const float ms = pp[s * w];
            if (ms == -INFINITY) continue;
            const float c = exp(ms - m);
            l += pp[s * w + 1] * c;
            acc += pp[s * w + 2 + d] * c;
        }
        olat[((ulong) t * a.n_head + h) * r + d] = bfr(acc / l);
    }
}

// Prefill attention (rows > MV_MAXT): threadgroup (block of 8 x ATTF_RS rows, KV head) with one simdgroup per (query head
// of the group, 8 rows); every K/V tile of ATTF_BK keys is shared by the group's ATTF_G query heads and all the rows.
// MLX's prefill rounding points: S = bf16(q * scale) K^T and O += bf16(P) V with f32 products and sums (f32 simdgroup
// matrices; the BF16 values are exact in f32); online softmax per row in f32 (causal: key <= the row's position; P
// rounded against the running max; the row sum from the unrounded P); then the softplus gate as k_attn_reduce:
// o[t][h*128+d] = bf16(bf16(attn) * bf16(softplus_ln2(g))).
// Q, K (transposed), V stay BF16 in threadgroup memory; each lane builds its two elements of a fragment (lane l holds (fm, fn),
// (fm, fn + 1): Apple's 8x8 layout, as MLX's BaseMMAFrag), so only O and S live in registers.  S and P stay in the
// fragments: a row's elements are on the lanes that differ in bits 0 and 3.
#if ATTF_BF
typedef simdgroup_bfloat8x8 attf_frag;
static inline void frag_bf(thread simdgroup_bfloat8x8& f, threadgroup const ushort* p, int ld, int fm, int fn) {
    thread auto& e = f.thread_elements();
    const uint w = *(threadgroup const uint*) (p + fm * ld + fn);
    e[0] = as_type<bfloat>((ushort) (w & 0xFFFFu));
    e[1] = as_type<bfloat>((ushort) (w >> 16));
}
#else
typedef simdgroup_float8x8 attf_frag;
static inline void frag_bf(thread simdgroup_float8x8& f, threadgroup const ushort* p, int ld, int fm, int fn) {
    thread auto& e = f.thread_elements();
    const uint w = *(threadgroup const uint*) (p + fm * ld + fn);
    e[0] = as_type<float>(w << 16);
    e[1] = as_type<float>(w & 0xFFFF0000u);
}
#endif
// Threadgroup memory (declared by each kernel: Metal allows it only at kernel scope): BF16 bits; row strides padded by
// one word so a fragment's 8 rows fall in different banks.  Kt: K transposed [dim][key]; Vt: [key][dim]; per simdgroup
// qs: bf16(q * scale) [row][dim] and ox: output staging.
#define ATTF_QLD (ATT_HD + 2)
#define ATTF_VLD (ATT_HD + 2)
#define ATTF_KLD (ATTF_BK + 2)
#if ATTF_DIAG_F32
typedef float attf_q;
#else
typedef ushort attf_q;
#endif
#define ATTF_SHARED                                                                                                    \
    threadgroup ushort Kt[ATT_HD * ATTF_KLD];                                                                          \
    threadgroup ushort Vt[ATTF_BK * ATTF_VLD];                                                                         \
    threadgroup attf_q Qt[ATTF_G * ATTF_RS][8 * ATTF_QLD];                                                             \
    threadgroup float Ox[ATTF_G * ATTF_RS][64];
template <typename KT>
static inline void attn_prefill(constant AttnArgs& a, device const float* q, device const KT* Kc, device const KT* Vc,
                                device const float* Ks, device const float* Vs, device const RowInfo* ri, device const float* g,
                                device float* o, int T, uint2 tg, uint tid, uint lane, uint sgi, threadgroup ushort* Kt,
                                threadgroup ushort* Vt, threadgroup attf_q* qs, threadgroup float* ox) {
    constexpr int QLD = ATTF_QLD, VLD = ATTF_VLD, KLD = ATTF_KLD;
    const int nthr = 32 * ATTF_G * ATTF_RS;
    const int kvh = (int) tg.y, hq = (int) sgi / ATTF_RS, h = kvh * ATTF_G + hq;
    const int t0 = (int) tg.x * 8 * ATTF_RS, r0 = t0 + ((int) sgi % ATTF_RS) * 8;
    const int kend = ri[min(t0 + 8 * ATTF_RS, T) - 1].pos + 1;
    const bool live = r0 < T;
    const int qid = (int) lane / 4, fm = (qid & 4) + ((int) lane / 2) % 4, fn = (qid & 2) * 2 + ((int) lane % 2) * 2;
    const int mypos = ri[min(r0 + fm, T - 1)].pos;
    const int lastpos = ri[min(r0 + 7, T - 1)].pos;
    for (int e = (int) lane; e < 8 * ATT_HD; e += 32) {   // rows past T repeat row T - 1
        const int rr = min(r0 + e / ATT_HD, T - 1);
#if ATTF_DIAG_F32
        qs[(e / ATT_HD) * QLD + e % ATT_HD] = q[((ulong) rr * a.n_head + h) * ATT_HD + e % ATT_HD] * a.scale;
#else
        qs[(e / ATT_HD) * QLD + e % ATT_HD] = tobf(q[((ulong) rr * a.n_head + h) * ATT_HD + e % ATT_HD] * a.scale);
#endif
    }
    simdgroup_float8x8 O[16];
    for (int j = 0; j < 16; ++j) O[j] = simdgroup_float8x8(0.0f);
    float m = -INFINITY, l = 0.0f;
    const int stride = a.n_kv * ATT_HD;
    for (int k0 = 0; k0 < kend; k0 += ATTF_BK) {
        for (int e = (int) tid; e < ATTF_BK * ATT_HD / 8; e += nthr) {
            const int key = e / (ATT_HD / 8), d8 = (e % (ATT_HD / 8)) * 8, p = k0 + key;
            uint4 kk = 0, vv = 0;
            if (p < kend) {
                kk = kv_load8(Kc, Ks, ri[t0].kv0 + p, stride, a.n_kv, kvh, kvh * ATT_HD + d8);   // a block's rows: one slot
                vv = kv_load8(Vc, Vs, ri[t0].kv0 + p, stride, a.n_kv, kvh, kvh * ATT_HD + d8);
            }
            const uint kw[4] = {kk.x, kk.y, kk.z, kk.w}, vw[4] = {vv.x, vv.y, vv.z, vv.w};
            for (int u = 0; u < 4; ++u) {
                Kt[(d8 + 2 * u) * KLD + key] = (ushort) (kw[u] & 0xFFFFu);
                Kt[(d8 + 2 * u + 1) * KLD + key] = (ushort) (kw[u] >> 16);
                *(threadgroup uint*) (Vt + key * VLD + d8 + 2 * u) = vw[u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (live && k0 <= lastpos) {
            simdgroup_float8x8 S[ATTF_BK / 8];
            for (int kb = 0; kb < ATTF_BK / 8; ++kb) S[kb] = simdgroup_float8x8(0.0f);
            for (int i = 0; i < 16; ++i) {
                attf_frag qf;
#if ATTF_DIAG_F32
                {
                    thread auto& qe = qf.thread_elements();
                    qe[0] = qs[i * 8 + fm * QLD + fn];
                    qe[1] = qs[i * 8 + fm * QLD + fn + 1];
                }
#else
                frag_bf(qf, qs + i * 8, QLD, fm, fn);
#endif
                for (int kb = 0; kb < ATTF_BK / 8; ++kb) {
                    attf_frag kt;
                    frag_bf(kt, Kt + i * 8 * KLD + kb * 8, KLD, fm, fn);
                    simdgroup_multiply_accumulate(S[kb], qf, kt, S[kb]);
                }
            }
            float rmax = -INFINITY;
            for (int kb = 0; kb < ATTF_BK / 8; ++kb) {
                thread auto& se = S[kb].thread_elements();
                for (int u = 0; u < 2; ++u) {
                    if (k0 + kb * 8 + fn + u > mypos) se[u] = -INFINITY;
                    rmax = max(rmax, se[u]);
                }
            }
            rmax = max(rmax, simd_shuffle_xor(rmax, (ushort) 1));
            rmax = max(rmax, simd_shuffle_xor(rmax, (ushort) 8));
            const float mn = max(m, rmax);
            const float cor = mn == -INFINITY ? 1.0f : exp(m - mn);
            float rs = 0;
            for (int kb = 0; kb < ATTF_BK / 8; ++kb) {
                thread auto& se = S[kb].thread_elements();
                for (int u = 0; u < 2; ++u) {
                    const float pv = se[u] == -INFINITY ? 0.0f : exp(se[u] - mn);
                    rs += pv;
                    se[u] = ATTF_DIAG_F32 ? pv : bfr(pv);
                }
            }
            rs += simd_shuffle_xor(rs, (ushort) 1);
            rs += simd_shuffle_xor(rs, (ushort) 8);
            l = l * cor + rs;
            m = mn;
            if (!simd_all(cor == 1.0f))
                for (int j = 0; j < 16; ++j) { thread auto& oe = O[j].thread_elements(); oe[0] *= cor; oe[1] *= cor; }
#if ATTF_BF
            attf_frag P[ATTF_BK / 8];
            for (int kb = 0; kb < ATTF_BK / 8; ++kb) {
                thread auto& se = S[kb].thread_elements();
                thread auto& pe = P[kb].thread_elements();
                pe[0] = as_type<bfloat>(tobf(se[0]));
                pe[1] = as_type<bfloat>(tobf(se[1]));
            }
#else
            thread simdgroup_float8x8 (&P)[ATTF_BK / 8] = S;
#endif
            for (int j = 0; j < 16; ++j)
                for (int kb = 0; kb < ATTF_BK / 8; ++kb) {
                    attf_frag vt;
                    frag_bf(vt, Vt + kb * 8 * VLD + j * 8, VLD, fm, fn);
                    simdgroup_multiply_accumulate(O[j], P[kb], vt, O[j]);
                }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!live) return;
    for (int j = 0; j < 16; ++j) {
        simdgroup_store(O[j], ox, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = (int) lane; e < 64; e += 32) {
            const int rr = e / 8, t = r0 + rr;
            const float lr = simd_shuffle(l, (ushort) ((rr & 4) * 4 + (rr & 3) * 2));   // a lane holding row rr
            if (t < T) {
                const ulong i = ((ulong) t * a.n_head + h) * ATT_HD + j * 8 + e % 8;
                const float att = bfr(ox[e] / lr);
                const float gx = g[i] * 0.69314718055994531f;
                const float sp = bfr((max(gx, 0.0f) + log(1.0f + exp(-fabs(gx)))) / 0.69314718055994531f);
                o[i] = bfr(att * sp);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void k_attn_prefill(constant AttnArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const ushort* Kc [[buffer(2)]],
                           device const ushort* Vc [[buffer(3)]], device const RowInfo* ri [[buffer(4)]], device const float* g [[buffer(5)]],
                           device float* o [[buffer(6)]], constant int& T [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]]) {
    ATTF_SHARED
    attn_prefill(a, q, Kc, Vc, (device const float*) nullptr, (device const float*) nullptr, ri, g, o, T, tg, tid, lane, sgi, Kt, Vt,
                 Qt[sgi], Ox[sgi]);
}
// k_attn_prefill on the 8-bit cache: scales Ks, Vs at buffers 8, 9
kernel void k_attn_prefill_q8(constant AttnArgs& a [[buffer(0)]], device const float* q [[buffer(1)]], device const char* Kc [[buffer(2)]],
                              device const char* Vc [[buffer(3)]], device const RowInfo* ri [[buffer(4)]], device const float* g [[buffer(5)]],
                              device float* o [[buffer(6)]], constant int& T [[buffer(7)]], device const float* Ks [[buffer(8)]],
                              device const float* Vs [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                              uint sgi [[simdgroup_index_in_threadgroup]]) {
    ATTF_SHARED
    attn_prefill(a, q, Kc, Vc, Ks, Vs, ri, g, o, T, tg, tid, lane, sgi, Kt, Vt, Qt[sgi], Ox[sgi]);
}

// Greedy argmax over the vocabulary (ties: the lowest id); one threadgroup of 1024 per row.
kernel void k_argmax(device const float* logits [[buffer(0)]], device int* out [[buffer(1)]], constant int& V [[buffer(2)]],
                     uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]], uint sgi [[simdgroup_index_in_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup int bi[32];
    device const float* l = logits + (ulong) t * V;
    float best = -INFINITY;
    int idx = 0;
    for (int i = (int) tid; i < V; i += 1024) if (l[i] > best) { best = l[i]; idx = i; }
    for (int o = 16; o > 0; o /= 2) {
        const float b2 = simd_shuffle_down(best, (ushort) o);
        const int i2 = simd_shuffle_down(idx, (ushort) o);
        if (b2 > best || (b2 == best && i2 < idx)) { best = b2; idx = i2; }
    }
    if (lane == 0) { bv[sgi] = best; bi[sgi] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = bv[0];
        int ix = bi[0];
        for (int s = 1; s < 32; ++s) if (bv[s] > b || (bv[s] == b && bi[s] < ix)) { b = bv[s]; ix = bi[s]; }
        out[t] = ix;
    }
}

#endif

#endif   // NSLM_KERNELS_MOE_METAL
