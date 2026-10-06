// nslm/search.metal - the P = 3 seed search on the GPU (contract: nslm/search_gpu.h).
//
// One threadgroup searches SG_ROWS rows of one column group (8 input channels) over every seed, building table
// entries (U(s) * sqrt(h), inverse Gram, Cholesky factor) SG_NCH seeds at a time in threadgroup memory.
//
// Exactness vs nslm/lib_search.c: -fno-fast-math and contraction off, so every CPU f32 expression is evaluated as
// written.  The CPU table is built in double; here float-float (~48 bits) rounded to f32, so an entry can differ only
// within ~2^-48 of an f32 rounding boundary; fragile seeds use exact emulated double instead (k_seed_exact).
//
// Prune: LS(s) = |x|^2 - b^T Gi b (b = U^T x), the real least-squares residual, lower-bounds any quantized candidate.
// Computed with FMAs; a seed is skipped only if LS(s) > best + (pm_s + 1e-5) |x|^2, pm_s = 4e-6 (sum |G|)(sum |Gi|)
// (f32 error of LS for that seed's conditioning; 1e-5 covers the CPU candidate formula).  So the arg-min and every
// running best are the CPU's.  Near-singular seeds get huge pm_s and are never skipped.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#include "search_gpu.h"

#define MAGIC 12582912.0f
// function constant 0: the full-transform mode; a separate pipeline, so the sqrt(h) search carries no 8 x 8 matrix
constant bool kFullFC [[function_constant(0)]];
constant bool kFull = is_function_constant_defined(kFullFC) && kFullFC;
constant constexpr float kR32 = as_type<float>(0x38000100u);   // fl(1 / 32767), NSLM_R32

// ---- float-float (emulating the CPU's double table build) --------------------------------------------------------
struct ff { float hi, lo; };
static inline ff ff_fast(float a, float b) { const float s = a + b; return ff{s, b - (s - a)}; }   // |a| >= |b|
static inline ff ff_two_sum(float a, float b) { const float s = a + b, v = s - a; return ff{s, (a - (s - v)) + (b - v)}; }
static inline ff ff_prod(float a, float b) { const float p = a * b; return ff{p, fma(a, b, -p)}; }
static inline ff ff_add(ff x, ff y) {
    const ff s = ff_two_sum(x.hi, y.hi), t = ff_two_sum(x.lo, y.lo);
    const ff v = ff_fast(s.hi, s.lo + t.hi);
    return ff_fast(v.hi, v.lo + t.lo);
}
static inline ff ff_sub(ff x, ff y) { return ff_add(x, ff{-y.hi, -y.lo}); }
static inline ff ff_mul(ff x, ff y) {
    const ff p = ff_prod(x.hi, y.hi);
    return ff_fast(p.hi, p.lo + (x.hi * y.lo + x.lo * y.hi));
}
static inline ff ff_div(ff x, ff y) {
    const float q1 = x.hi / y.hi;
    ff r = ff_sub(x, ff_mul(ff{q1, 0.0f}, y));
    const float q2 = r.hi / y.hi;
    r = ff_sub(r, ff_mul(ff{q2, 0.0f}, y));
    const float q3 = r.hi / y.hi;
    return ff_add(ff_fast(q1, q2), ff{q3, 0.0f});
}
static inline ff ff_sqrt(ff x) {   // sqrt of a non-negative value; one Newton step
    if (!(x.hi > 0.0f)) return ff{0.0f, 0.0f};
    const float s = sqrt(x.hi);
    const ff r = ff_sub(x, ff_prod(s, s));
    return ff_fast(s, r.hi / (2.0f * s));
}
static inline ff ff_fmax0(ff x) { return x.hi > 0.0f || (x.hi == 0.0f && x.lo > 0.0f) ? x : ff{0.0f, 0.0f}; }

// ---- IEEE binary64, bit-exact, on 64-bit integers (fragile seeds) ----------------------------------------------
//
// For ill-conditioned Gram matrices (near-singular seeds, extreme sqrt(h) ratios) float-float's ~2^-48 error survives
// rounding to f32.  Those seeds are rebuilt with seedtab_one's exact operation sequence in emulated IEEE double:
// round to nearest even, normal numbers and zero only (the table never meets subnormals, infinities or NaNs).
struct sd { ulong m; int e; bool s; };   // (-1)^s * m * 2^e, m in [2^52, 2^53) or m = 0

static inline sd sd_from_float(float f) {
    if (f == 0.0f) return sd{0ul, 0, (as_type<uint>(f) >> 31) != 0u};
    const uint u = as_type<uint>(f);
    return sd{((ulong) ((u & 0x7FFFFFu) | 0x800000u)) << 29, (int) ((u >> 23) & 255u) - 127 - 52, (u >> 31) != 0u};
}
// m * 2^e exactly, plus a nonzero remainder below m's last bit when sticky: rounded to 53 bits, nearest even.
static inline sd sd_norm(ulong m, int e, bool s, bool sticky) {
    if (m == 0ul) return sd{0ul, 0, s};
    const int top = 63 - (int) clz(m);
    if (top <= 52) return sd{m << (52 - top), e - (52 - top), s};   // callers keep enough bits: exact here
    const int k = top - 52;
    const ulong rem = m & ((1ul << k) - 1ul), hlf = 1ul << (k - 1);
    m >>= k;
    e += k;
    if (rem > hlf || (rem == hlf && (sticky || (m & 1ul)))) {
        m += 1ul;
        if (m == (1ul << 53)) { m >>= 1; e += 1; }
    }
    return sd{m, e, s};
}
static inline sd sd_add(sd a, sd b) {
    if (a.m == 0ul && b.m == 0ul) return sd{0ul, 0, a.s && b.s};   // +0 + -0 = +0
    if (a.m == 0ul) return b;
    if (b.m == 0ul) return a;
    if (a.e < b.e || (a.e == b.e && a.m < b.m)) { const sd t = a; a = b; b = t; }   // |a| >= |b|
    ulong ma = a.m << 10, mb = b.m << 10;   // 10 guard bits: a shift of up to 10 loses only zeros
    const int d = a.e - b.e;
    bool sticky = false;
    if (d > 0) {
        if (d >= 63) { sticky = mb != 0ul; mb = 0ul; }
        else { sticky = (mb & ((1ul << d) - 1ul)) != 0ul; mb >>= d; }
    }
    if (a.s == b.s) return sd_norm(ma + mb, a.e - 10, a.s, sticky);
    ulong m = ma - mb;
    if (sticky) m -= 1ul;   // the dropped part of b borrows one unit; the remainder stays (sticky)
    if (m == 0ul && !sticky) return sd{0ul, 0, false};   // exact cancellation: +0 (round to nearest)
    return sd_norm(m, a.e - 10, a.s, sticky);
}
static inline sd sd_neg(sd a) { return sd{a.m, a.e, !a.s}; }
static inline sd sd_sub(sd a, sd b) { return sd_add(a, sd_neg(b)); }
static inline sd sd_mul(sd a, sd b) {
    if (a.m == 0ul || b.m == 0ul) return sd{0ul, 0, a.s != b.s};
    const ulong hi = mulhi(a.m, b.m), lo = a.m * b.m;   // the 105/106-bit product
    const int top = 64 + 63 - (int) clz(hi), k = top - 62;   // keep 63 bits
    const ulong m = (hi << (64 - k)) | (lo >> k);
    return sd_norm(m, a.e + b.e + k, a.s != b.s, (lo & ((1ul << k) - 1ul)) != 0ul);
}
static inline sd sd_div(sd a, sd b) {   // b != 0
    if (a.m == 0ul) return sd{0ul, 0, a.s != b.s};
    ulong r = a.m, q = 0ul;
    int e = a.e - b.e;
    if (r < b.m) { r <<= 1; e -= 1; }   // r in [b, 2b): the first quotient bit is 1
    for (short i = 0; i < 60; ++i) {
        q <<= 1;
        if (r >= b.m) { r -= b.m; q |= 1ul; }
        r <<= 1;
    }
    return sd_norm(q, e - 59, a.s != b.s, r != 0ul);
}
static inline sd sd_sqrt(sd a) {   // a >= 0
    if (a.m == 0ul) return sd{0ul, 0, false};
    ulong m = a.m;
    int e = a.e;
    if (e & 1) { m <<= 1; e -= 1; }   // even exponent: sqrt(m 2^e) = sqrt(m 2^56) 2^((e - 56) / 2)
    // digit-by-digit integer square root of N = m * 2^56 (< 2^110), 128-bit as (nh, nl)
    ulong nh = m >> 8, nl = m << 56, rh = 0ul, rl = 0ul, bh = 1ul << 46, bl = 0ul;   // bit = 4^55
    for (short it = 0; it < 56; ++it) {
        // t = r + bit
        const ulong tl = rl + bl, th = rh + bh + (tl < rl ? 1ul : 0ul);
        const bool ge = nh > th || (nh == th && nl >= tl);
        // r >>= 1
        rl = (rl >> 1) | (rh << 63);
        rh >>= 1;
        if (ge) {
            const ulong nl2 = nl - tl;
            nh = nh - th - (nl < tl ? 1ul : 0ul);
            nl = nl2;
            const ulong rl2 = rl + bl;
            rh = rh + bh + (rl2 < rl ? 1ul : 0ul);
            rl = rl2;
        }
        // bit >>= 2
        bl = (bl >> 2) | (bh << 62);
        bh >>= 2;
    }
    return sd_norm(rl, (e - 56) / 2, false, (nh | nl) != 0ul);   // the root fits 55 bits
}
static inline float sd_to_float(sd a) {   // round to nearest even (normal f32 range)
    if (a.m == 0ul) return a.s ? -0.0f : 0.0f;
    ulong m = a.m >> 29;
    const ulong rem = a.m & ((1ul << 29) - 1ul), hlf = 1ul << 28;
    int e = a.e + 29;
    if (rem > hlf || (rem == hlf && (m & 1ul))) { m += 1ul; if (m == (1ul << 24)) { m >>= 1; e += 1; } }
    return as_type<float>((a.s ? 0x80000000u : 0u) | ((uint) (e + 23 + 127) << 23) | ((uint) m & 0x7FFFFFu));
}
static inline sd sd_fmax0(sd a) { return a.s && a.m != 0ul ? sd{0ul, 0, false} : sd{a.m, a.e, false}; }

// seedtab_one's double block, operation for operation (lib_search.c): Gi and R rounded to f32.
static inline void gram_exact(thread const float (&U)[24], thread float (&gi)[6], thread float (&R)[6]) {
    sd g[3][3];
    for (short i = 0; i < 3; ++i)
        for (short j = 0; j < 3; ++j) {
            sd a = sd{0ul, 0, false};
            for (short c = 0; c < 8; ++c) a = sd_add(a, sd_mul(sd_from_float(U[c * 3 + i]), sd_from_float(U[c * 3 + j])));
            g[i][j] = a;
        }
    const sd det = sd_add(sd_sub(sd_mul(g[0][0], sd_sub(sd_mul(g[1][1], g[2][2]), sd_mul(g[1][2], g[2][1]))),
                                 sd_mul(g[0][1], sd_sub(sd_mul(g[1][0], g[2][2]), sd_mul(g[1][2], g[2][0])))),
                          sd_mul(g[0][2], sd_sub(sd_mul(g[1][0], g[2][1]), sd_mul(g[1][1], g[2][0]))));
    const sd r00 = sd_sqrt(g[0][0]), r01 = sd_div(g[0][1], r00), r02 = sd_div(g[0][2], r00);
    const sd r11 = sd_sqrt(sd_fmax0(sd_sub(g[1][1], sd_mul(r01, r01))));
    const sd r12 = (r11.m != 0ul && !r11.s) ? sd_div(sd_sub(g[1][2], sd_mul(r01, r02)), r11) : sd{0ul, 0, false};
    const sd r22 = sd_sqrt(sd_fmax0(sd_sub(sd_sub(g[2][2], sd_mul(r02, r02)), sd_mul(r12, r12))));
    R[0] = sd_to_float(r00); R[1] = sd_to_float(r01); R[2] = sd_to_float(r02);
    R[3] = sd_to_float(r11); R[4] = sd_to_float(r12); R[5] = sd_to_float(r22);
    for (short k = 0; k < 6; ++k) gi[k] = 0.0f;
    if (det.m != 0ul && !det.s) {
        gi[0] = sd_to_float(sd_div(sd_sub(sd_mul(g[1][1], g[2][2]), sd_mul(g[1][2], g[2][1])), det));
        gi[1] = sd_to_float(sd_div(sd_sub(sd_mul(g[0][2], g[2][1]), sd_mul(g[0][1], g[2][2])), det));
        gi[2] = sd_to_float(sd_div(sd_sub(sd_mul(g[0][1], g[1][2]), sd_mul(g[0][2], g[1][1])), det));
        gi[3] = sd_to_float(sd_div(sd_sub(sd_mul(g[0][0], g[2][2]), sd_mul(g[0][2], g[2][0])), det));
        gi[4] = sd_to_float(sd_div(sd_sub(sd_mul(g[0][2], g[1][0]), sd_mul(g[0][0], g[1][2])), det));
        gi[5] = sd_to_float(sd_div(sd_sub(sd_mul(g[0][0], g[1][1]), sd_mul(g[0][1], g[1][0])), det));
    }
}

// The 24 states after seed s from the per-seed stream table (nslm/lfsr.h lfsr_stream24).
static inline void states(uint s, device const uint * G, thread uint (&v)[24]) {
    const uint g = G[s], lo = s | (g << 16);
    for (short k = 1; k <= 16; ++k) v[k - 1] = extract_bits(lo, (uint) k, 16u);
    for (short k = 17; k <= 24; ++k) v[k - 1] = extract_bits(g, (uint) (k - 16), 16u);
}

// U(s) scaled by sh, exactly as seedtab_one: fl(fl(S * R32) * sh_c).  With a full transform A: U' = A U, rows summed
// in order k = 0 .. c (A lower triangular).
static inline void scaled_u(uint s, device const uint * G, thread const float (&sh)[8], thread uint (&v)[24], thread float (&U)[24],
                            bool full, thread const float (&A)[64]) {
    states(s, G, v);
    if (!full) {
        for (short k = 0; k < 24; ++k) { U[k] = (float) ((int) v[k] - 32768) * kR32; U[k] = U[k] * sh[k / 3]; }
        return;
    }
    float u0[24];
    for (short k = 0; k < 24; ++k) u0[k] = (float) ((int) v[k] - 32768) * kR32;
    for (short c = 0; c < 8; ++c)
        for (short p = 0; p < 3; ++p) {
            float acc = 0.0f;
            for (short k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * u0[k * 3 + p];
            U[c * 3 + p] = acc;
        }
}

// seedtab_one's double block in float-float: the Gram matrix, its inverse (Gi, rounded to f32; zero when singular)
// and Cholesky factor (R, rounded to f32), and sum |G| for the prune margin.
static inline void gram(thread const float (&U)[24], thread float (&gi)[6], thread float (&R)[6], thread float & sg) {
    ff g[3][3];
    for (short i = 0; i < 3; ++i)
        for (short j = 0; j < 3; ++j) {
            ff a = ff{0.0f, 0.0f};
            for (short c = 0; c < 8; ++c) a = ff_add(a, ff_prod(U[c * 3 + i], U[c * 3 + j]));
            g[i][j] = a;
        }
    const ff det = ff_add(ff_sub(ff_mul(g[0][0], ff_sub(ff_mul(g[1][1], g[2][2]), ff_mul(g[1][2], g[2][1]))),
                                 ff_mul(g[0][1], ff_sub(ff_mul(g[1][0], g[2][2]), ff_mul(g[1][2], g[2][0])))),
                          ff_mul(g[0][2], ff_sub(ff_mul(g[1][0], g[2][1]), ff_mul(g[1][1], g[2][0]))));
    const ff r00 = ff_sqrt(g[0][0]), r01 = ff_div(g[0][1], r00), r02 = ff_div(g[0][2], r00);
    const ff r11 = ff_sqrt(ff_fmax0(ff_sub(g[1][1], ff_mul(r01, r01))));
    const ff r12 = r11.hi > 0.0f ? ff_div(ff_sub(g[1][2], ff_mul(r01, r02)), r11) : ff{0.0f, 0.0f};
    const ff r22 = ff_sqrt(ff_fmax0(ff_sub(ff_sub(g[2][2], ff_mul(r02, r02)), ff_mul(r12, r12))));
    R[0] = r00.hi; R[1] = r01.hi; R[2] = r02.hi; R[3] = r11.hi; R[4] = r12.hi; R[5] = r22.hi;
    for (short k = 0; k < 6; ++k) gi[k] = 0.0f;
    if (det.hi > 0.0f) {
        gi[0] = ff_div(ff_sub(ff_mul(g[1][1], g[2][2]), ff_mul(g[1][2], g[2][1])), det).hi;
        gi[1] = ff_div(ff_sub(ff_mul(g[0][2], g[2][1]), ff_mul(g[0][1], g[2][2])), det).hi;
        gi[2] = ff_div(ff_sub(ff_mul(g[0][1], g[1][2]), ff_mul(g[0][2], g[1][1])), det).hi;
        gi[3] = ff_div(ff_sub(ff_mul(g[0][0], g[2][2]), ff_mul(g[0][2], g[2][0])), det).hi;
        gi[4] = ff_div(ff_sub(ff_mul(g[0][2], g[1][0]), ff_mul(g[0][0], g[1][2])), det).hi;
        gi[5] = ff_div(ff_sub(ff_mul(g[0][0], g[1][1]), ff_mul(g[0][1], g[1][0])), det).hi;
    }
    sg = 0.0f;
    for (short i = 0; i < 3; ++i) for (short j = 0; j < 3; ++j) sg += fabs(g[i][j].hi);
}

// The table entry of seed s: U[24], Gi[6], R[6], prune margin (units of |x|^2).
static inline float gi_sum(thread const float (&gi)[6]) {
    return fabs(gi[0]) + fabs(gi[3]) + fabs(gi[5]) + 2.0f * (fabs(gi[1]) + fabs(gi[2]) + fabs(gi[4]));
}
static inline bool fragile(float sg, float si, float exact_kappa) { return exact_kappa <= 0.0f || !(sg * si <= exact_kappa); }

// The table's Gi and R: float-float, replaced by the pre-pass's exact double values when the seed is fragile
// (k_seed_exact wrote them to ex; the flag says so).
static inline void table_gram(thread const float (&U)[24], thread float (&gi)[6], thread float (&R)[6], thread float & sg,
                              thread float & si, device const uchar * flag, device const float * ex) {
    gram(U, gi, R, sg);
    if (*flag) {
        for (short k = 0; k < 6; ++k) { gi[k] = ex[k]; R[k] = ex[6 + k]; }
    }
    si = gi_sum(gi);
}

static inline void build_entry(uint s, device const uint * G, thread const float (&sh)[8], threadgroup float * e,
                               bool full, thread const float (&A)[64], device const uchar * flags, device const float * ex) {
    uint v[24];
    float U[24], gi[6], R[6], sg, si;
    scaled_u(s, G, sh, v, U, full, A);
    table_gram(U, gi, R, sg, si, flags + s, ex + (ulong) s * 12);
    for (short k = 0; k < 24; ++k) e[k] = U[k];
    for (short k = 0; k < 6; ++k) { e[24 + k] = gi[k]; e[30 + k] = R[k]; }
    e[36] = 4e-6f * sg * si + 1e-5f;
    e[37] = e[38] = e[39] = 0.0f;
}

static inline float pow2f(int e) { return as_type<float>((uint) (e + 127) << 23); }
static inline float clampq(float r) { return r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r); }

// The CPU's decoded error of a (seed, exponent, q) candidate: BF16 decode exactly as nslm_decode_block, then
// sum_c (x_c - sh_c w_hat_c)^2 in order.
static inline float decoded_err(thread const float (&x)[8], thread const float (&sh)[8], thread const uint (&v)[24],
                                int e, int q0, int q1, int q2, bool full, thread const float (&A)[64]) {
    const float sc = kR32 * pow2f(e);
    float bf[8];
    for (short c = 0; c < 8; ++c) {
        const int isum = ((int) v[c * 3] - 32768) * q0 + ((int) v[c * 3 + 1] - 32768) * q1 + ((int) v[c * 3 + 2] - 32768) * q2;
        uint u = as_type<uint>((float) isum * sc);
        u += 0x7FFFu + ((u >> 16) & 1u);
        bf[c] = as_type<float>(u & 0xFFFF0000u);
    }
    float er = 0.0f;
    for (short c = 0; c < 8; ++c) {
        float ab;
        if (full) { ab = 0.0f; for (short k = 0; k <= c; ++k) ab = ab + A[c * 8 + k] * bf[k]; }
        else ab = sh[c] * bf[c];
        const float d = x[c] - ab;
        er = er + d * d;
    }
    return er;
}

kernel void k_seed_search(constant SearchArgs & a [[buffer(0)]], device const float * W [[buffer(1)]],
                          device const float * SH [[buffer(2)]], device const uint * G [[buffer(3)]],
                          device ushort * seed_out [[buffer(4)]], device ushort * nib_out [[buffer(5)]],
                          device float * err_out [[buffer(6)]], device const uchar * FLAGS [[buffer(7)]],
                          device const float * EX [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    device const uchar * flags = FLAGS + (ulong) tg.x * 65536;   // this column group's fragile seeds (k_seed_exact)
    device const float * ex = EX + (ulong) tg.x * 65536 * 12;
    threadgroup float tab[SG_NCH * SG_ENT];
    const int g = a.g0 + (int) tg.x, ng = a.cols / 8, lo = a.bias, hi = a.bias + 15;
    const bool full = kFull;
    float sh[8], A[64];
    for (short c = 0; c < 8; ++c) sh[c] = full ? 1.0f : SH[g * 8 + c];
    for (short k = 0; k < 64; ++k) A[k] = full ? SH[tg.x * 64 + k] : 0.0f;
    float x[SG_BPT][8], wn[SG_BPT], best[SG_BPT];
    int bs[SG_BPT], be[SG_BPT], row[SG_BPT];
    for (short j = 0; j < SG_BPT; ++j) {
        row[j] = (int) tg.y * SG_ROWS + j * SG_TPB + tid;
        const int r = min(row[j], a.rows - 1);
        if (full) {
            float w[8];
            for (short c = 0; c < 8; ++c) w[c] = W[(ulong) r * a.cols + g * 8 + c];
            for (short c = 0; c < 8; ++c) { float acc = 0.0f; for (short k = 0; k <= c; ++k) acc = acc + A[c * 8 + k] * w[k]; x[j][c] = acc; }
        } else
            for (short c = 0; c < 8; ++c) x[j][c] = W[(ulong) r * a.cols + g * 8 + c] * sh[c];
        float n = 0.0f;
        for (short c = 0; c < 8; ++c) n = n + x[j][c] * x[j][c];
        wn[j] = n; best[j] = INFINITY; bs[j] = 1; be[j] = lo;
    }
    for (int s0 = 1; s0 <= a.n_seeds; s0 += SG_NCH) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < SG_NCH && s0 + tid <= a.n_seeds) build_entry((uint) (s0 + tid), G, sh, tab + tid * SG_ENT, full, A, flags, ex);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int nk = min(SG_NCH, a.n_seeds - s0 + 1);
        for (int k = 0; k < nk; ++k) {
            threadgroup const float * T = tab + k * SG_ENT;
            const float4 u0 = ((threadgroup const float4 *) T)[0], u1 = ((threadgroup const float4 *) T)[1],
                         u2 = ((threadgroup const float4 *) T)[2], u3 = ((threadgroup const float4 *) T)[3],
                         u4 = ((threadgroup const float4 *) T)[4], u5 = ((threadgroup const float4 *) T)[5],
                         i0 = ((threadgroup const float4 *) T)[6];
            const float2 i1 = ((threadgroup const float2 *) T)[14];
            const float pm = T[36];
            // U row c = (U[3c], U[3c+1], U[3c+2]): rows 0..7 from u0..u5
            const float U[24] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w, u2.x, u2.y, u2.z, u2.w,
                                 u3.x, u3.y, u3.z, u3.w, u4.x, u4.y, u4.z, u4.w, u5.x, u5.y, u5.z, u5.w};
            for (short j = 0; j < SG_BPT; ++j) {
                if (a.prune) {   // fused lower bound: skip seeds that cannot reach the running best
                    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
                    for (short c = 0; c < 8; ++c) { b0 = fma(U[c * 3], x[j][c], b0); b1 = fma(U[c * 3 + 1], x[j][c], b1); b2 = fma(U[c * 3 + 2], x[j][c], b2); }
                    const float t0 = fma(i0.x, b0, fma(i0.y, b1, i0.z * b2));
                    const float t1 = fma(i0.y, b0, fma(i0.w, b1, i1.x * b2));
                    const float t2 = fma(i0.z, b0, fma(i1.x, b1, i1.y * b2));
                    const float ls = wn[j] - fma(b0, t0, fma(b1, t1, b2 * t2));
                    if (ls > best[j] + pm * wn[j]) continue;
                }
                // the CPU's candidate loop, expression for expression (nslm_search_ref)
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
                for (short c = 0; c < 8; ++c) {
                    b0 = b0 + U[c * 3] * x[j][c];
                    b1 = b1 + U[c * 3 + 1] * x[j][c];
                    b2 = b2 + U[c * 3 + 2] * x[j][c];
                }
                const float t0 = (i0.x * b0 + i0.y * b1) + i0.z * b2;
                const float t1 = (i0.y * b0 + i0.w * b1) + i1.x * b2;
                const float t2 = (i0.z * b0 + i1.x * b1) + i1.y * b2;
                const float m = fmax(fmax(fabs(t0), fabs(t1)), fabs(t2));
                int e0 = (int) ((as_type<uint>(m) >> 23) & 255u) - 127 - 2;
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
                    const float er = rec > 4.0f * wn[j] ? INFINITY : (wn[j] - (2.0f * sc) * qb) + rec;
                    if (er < best[j]) { best[j] = er; bs[j] = s0 + k; be[j] = e; }
                }
            }
        }
    }
    // finish_block: the winner's U and Gi again (same functions, same inputs: the scan's values), then plain rounding
    // or the best of the 3^P neighbourhood by decoded error
    for (short j = 0; j < SG_BPT; ++j) {
        if (row[j] >= a.rows) continue;
        uint v[24];
        float U[24], gi[6], R[6], sg, si;
        scaled_u((uint) bs[j], G, sh, v, U, full, A);
        table_gram(U, gi, R, sg, si, flags + bs[j], ex + (ulong) bs[j] * 12);
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
        for (short c = 0; c < 8; ++c) { b0 = b0 + U[c * 3] * x[j][c]; b1 = b1 + U[c * 3 + 1] * x[j][c]; b2 = b2 + U[c * 3 + 2] * x[j][c]; }
        const float t0 = (gi[0] * b0 + gi[1] * b1) + gi[2] * b2;
        const float t1 = (gi[1] * b0 + gi[3] * b1) + gi[4] * b2;
        const float t2 = (gi[2] * b0 + gi[4] * b1) + gi[5] * b2;
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
        const ulong k = (ulong) row[j] * (ulong) ng + (ulong) g;
        seed_out[k] = (ushort) bs[j];
        nib_out[k] = (ushort) (((e - a.bias) & 15) | ((bq0 & 15) << 4) | ((bq1 & 15) << 8) | ((bq2 & 15) << 12));
        err_out[k] = bde;
    }
}

// The pre-pass: for every (column group of the dispatch, seed), the float-float table's condition estimate; fragile
// seeds get seedtab_one's exact double Gi and R in EX[group][seed][12] and FLAGS[group][seed] = 1.
// Grid (column groups, 65536 / 256), 256 threads.
kernel void k_seed_exact(constant SearchArgs & a [[buffer(0)]], device const float * SH [[buffer(2)]],
                         device const uint * G [[buffer(3)]], device uchar * FLAGS [[buffer(7)]], device float * EX [[buffer(8)]],
                         uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint s = tg.y * 256 + tid;
    const ulong slot = (ulong) tg.x * 65536 + s;
    if (s < 1u || s > (uint) a.n_seeds) { if (s < 65536u) FLAGS[slot] = 0; return; }
    const int g = a.g0 + (int) tg.x;
    const bool full = kFull;
    float sh[8], A[64];
    for (short c = 0; c < 8; ++c) sh[c] = full ? 1.0f : SH[g * 8 + c];
    for (short k = 0; k < 64; ++k) A[k] = full ? SH[tg.x * 64 + k] : 0.0f;
    uint v[24];
    float U[24], gi[6], R[6], sg;
    scaled_u(s, G, sh, v, U, full, A);
    gram(U, gi, R, sg);
    if (!fragile(sg, gi_sum(gi), a.exact_kappa)) { FLAGS[slot] = 0; return; }
    gram_exact(U, gi, R);
    device float * e = EX + slot * 12;
    for (short k = 0; k < 6; ++k) { e[k] = gi[k]; e[6 + k] = R[k]; }
    FLAGS[slot] = 1;
}

// Test hook (tests/test_search_gpu.m): table entries of seeds 1 .. 65535 for sqrt(h) = sh (unweighted: ones), U then
// Gi then R (36 floats per seed), with the float-float build (mode 0) or the exact double emulation (mode 1).
kernel void k_seedtab_test(device const float * SH [[buffer(0)]], device const uint * G [[buffer(1)]], device float * out [[buffer(2)]],
                           constant int & mode [[buffer(3)]], uint s [[thread_position_in_grid]]) {
    if (s < 1u || s > 65535u) return;
    float sh[8], A[64], U[24], gi[6], R[6], sg;
    for (short c = 0; c < 8; ++c) sh[c] = SH[c];
    for (short k = 0; k < 64; ++k) A[k] = 0.0f;
    uint v[24];
    scaled_u(s, G, sh, v, U, false, A);
    if (mode) gram_exact(U, gi, R); else gram(U, gi, R, sg);
    device float * o = out + (ulong) s * 36;
    for (short k = 0; k < 24; ++k) o[k] = U[k];
    for (short k = 0; k < 6; ++k) { o[24 + k] = gi[k]; o[30 + k] = R[k]; }
}
