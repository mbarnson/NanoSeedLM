// nslm/lfsr.h - seed -> U(s) decode spec (header-only C99).  The scalar functions are the spec; every other generator
// (word-parallel, state cache, Metal) must match them bit for bit.
//
// LFSR (SeedLM paper sec. 3.1): Fibonacci, K = 16, taps (0, 1, 3, 12) = x^16 + x^12 + x^3 + x + 1, period 65535.
//   b = s0 ^ s1 ^ s3 ^ s12;  s = (s >> 1) | (b << 15)
// The seed IS the initial state s in [1, 65535]; V(s) is filled row-major with the C*P states that FOLLOW it:
//   V[c][p] = state_{c*P + p + 1}, state_0 = s.
// Normalisation (exact, backend-independent): S = V - 32768 in [-32767, 32767];  U = fl(S * R32), R32 = fl(1 / 32767).
// Block decode (C = 8, P = 3, int4 q, exponent e): isum_c = sum_p S[c][p] * q_p (exact, |isum| < 2^20),
//   w_c = bf16_rne( fl((float) isum_c * fl(R32 * 2^e)) ).  R32 * 2^e is exact: one IEEE multiply per weight.
#pragma once
#include <stdint.h>
#include <string.h>

#define LFSR_K 16
#define LFSR_TAPS ((1u << 0) | (1u << 1) | (1u << 3) | (1u << 12))
#define LFSR_PERIOD 65535
#define NSLM_R32 (1.0f / 32767.0f)   // correctly rounded by the compiler: the spec's R32

static inline uint16_t lfsr_step(uint16_t s) {
    const uint16_t b = (uint16_t) ((s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1u);
    return (uint16_t) ((s >> 1) | (b << 15));
}

// The n states that follow seed s (scalar spec).
static inline void lfsr_states(uint16_t s, int n, uint16_t* out) {
    for (int k = 0; k < n; ++k) { s = lfsr_step(s); out[k] = s; }
}

// Word-parallel: state_n = x[n .. n+15] of the bit stream x; x[n+16] = x[n] ^ x[n+1] ^ x[n+3] ^ x[n+12] has shortest
// lag 4, so 4 new bits per word operation.  Produces states 1..n (n <= 48).
static inline void lfsr_states_wordpar(uint16_t s, int n, uint16_t* out) {
    uint64_t r = s;
    for (int i = 0; i < n; i += 4) {
        const uint64_t nb = (r >> i) ^ (r >> (i + 1)) ^ (r >> (i + 3)) ^ (r >> (i + 12));
        r |= (nb & 0xFull) << (i + 16);
    }
    for (int k = 1; k <= n; ++k) out[k - 1] = (uint16_t) (r >> k);
}

// The 24 stream bits after seed s, x[16 .. 39] (bit k = bit 15 of state_{k+1}): the per-seed stream table of the GPU
// kernels.  state_k = bits k .. k + 15 of the 40-bit stream (s | g << 16).  g is GF(2)-linear in s:
// g(s) = g(s & 0xFF) ^ g(s & 0xFF00).
static inline uint32_t lfsr_stream24(uint16_t s) {
    uint16_t st[24];
    lfsr_states(s, 24, st);
    uint32_t g = 0;
    for (int k = 0; k < 24; ++k) g |= (uint32_t) (st[k] >> 15) << k;
    return g;
}

// The 32 stream bits after seed s (covers states 17..32 for P = 4 blocks): bit k = bit 15 of state_{k+1}.
static inline uint32_t lfsr_stream32(uint16_t s) {
    uint16_t st[32];
    lfsr_states(s, 32, st);
    uint32_t g = 0;
    for (int k = 0; k < 32; ++k) g |= (uint32_t) (st[k] >> 15) << k;
    return g;
}

// The paper's state cache: cache[i] = i-th state of the cycle from state 1, padded with LFSR_PAD entries repeating the
// start so a window never wraps; pos[s] = index of s.  U(s)'s states are cache[pos[s] + 1 .. pos[s] + n].
#define LFSR_PAD 64
typedef struct {
    uint16_t cache[LFSR_PERIOD + LFSR_PAD];
    uint16_t pos[65536];
} LfsrCache;

static inline void lfsr_cache_build(LfsrCache* c) {
    uint16_t s = 1;
    for (int i = 0; i < LFSR_PERIOD; ++i) { c->cache[i] = s; c->pos[s] = (uint16_t) i; s = lfsr_step(s); }
    for (int i = 0; i < LFSR_PAD; ++i) c->cache[LFSR_PERIOD + i] = c->cache[i];
    c->pos[0] = 0;   // never a seed
}
static inline void lfsr_states_cache(const LfsrCache* c, uint16_t s, int n, uint16_t* out) {
    const int p = c->pos[s];
    for (int k = 0; k < n; ++k) out[k] = c->cache[p + 1 + k];
}

// ---- 4-bit block (C = 8, P = 3): 16-bit seed + 16-bit nibble word = 32 bits per 8 weights ----

#define NSLM_C 8
#define NSLM_P 3
#define NSLM_CP (NSLM_C * NSLM_P)

// nibble word: bits 0-3 exponent code (e = bias + code), bits 4-7 q0, 8-11 q1, 12-15 q2 (int4 two's complement)
static inline uint16_t nslm_pack(int ecode, const int q[NSLM_P]) {
    return (uint16_t) ((ecode & 15) | ((q[0] & 15) << 4) | ((q[1] & 15) << 8) | ((q[2] & 15) << 12));
}
static inline int nslm_ecode(uint16_t nib) { return nib & 15; }
static inline int nslm_q(uint16_t nib, int p) {
    const int v = (nib >> (4 + 4 * p)) & 15;
    return v >= 8 ? v - 16 : v;
}

// fl(R32 * 2^e) built from exponent bits (exact; e in a normal range).
static inline float nslm_scale(int e) {
    uint32_t u = (uint32_t) (e + 127) << 23;
    float p2;
    memcpy(&p2, &u, 4);
    return NSLM_R32 * p2;
}

static inline uint16_t nslm_f2bf(float f) {   // round to nearest even (finite inputs)
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
static inline float nslm_bf2f(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Decodes one block (the spec) to BF16 bits.
static inline void nslm_decode_block(uint16_t seed, uint16_t nib, int exp_bias, uint16_t out_bf16[NSLM_C]) {
    uint16_t st[NSLM_CP];
    lfsr_states(seed, NSLM_CP, st);
    const float sc = nslm_scale(exp_bias + nslm_ecode(nib));
    const int q0 = nslm_q(nib, 0), q1 = nslm_q(nib, 1), q2 = nslm_q(nib, 2);
    for (int c = 0; c < NSLM_C; ++c) {
        const int32_t isum = ((int32_t) st[c * 3] - 32768) * q0 + ((int32_t) st[c * 3 + 1] - 32768) * q1 +
                             ((int32_t) st[c * 3 + 2] - 32768) * q2;
        out_bf16[c] = nslm_f2bf((float) isum * sc);
    }
}
