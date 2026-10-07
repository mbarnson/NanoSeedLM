// nslm/search4.h - 4.5-bit SeedLM blocks: C = 8 weights, P = 4 int4 coefficients, 4-bit exponent code, 16-bit seed
// (36 bits per 8 weights).  Decode spec and the scalar C reference seed search.
//
// Decode: S[c][p] = state_(4c+p+1) - 32768 (lfsr_states(seed, 32)), e = ecode + bias,
//   w_c = bf16( R32 * 2^e * isum_c ),  isum_c = sum_p S[c][p] * q_p  (exact integer), as nslm_decode_block for P = 3.
// Coefficient word: q_p (two's complement) in bits 4p .. 4p+3.
//
// Search, per column group (8 input channels), weighting sh = sqrt(h) (or 1):
//   x = w * sh;  for every seed s:  U = (S * R32) * sh  (8 x 4),  G = U^T U = L L^T (f32 Cholesky),
//   b = U^T x,  t = G^-1 b (forward / back substitution with L),  e0 = floor(log2 max|t|) - 2  (clamped to the window),
//   for each exponent e0 + exp_delta[k] (clamped):  q = clamp(rne(t / 2^e), -8, 7),
//     err = (|x|^2 - 2 * 2^e * q.b) + 2^2e * |L^T q|^2,  rejected (INFINITY) when 2^2e |L^T q|^2 > 4 |x|^2.
//   The arg-min wins (lowest seed, then the first exponent, on exact ties).  Seeds whose Cholesky fails are skipped.
//   Refit: the winner's q among its 3^4 neighbours by the error of the DECODED weights, sum_c (x_c - sh_c w_c)^2.
// nslm/search4.metal evaluates the identical f32 expressions in the identical order (no fast math, no contraction):
// bit-identical results.
#pragma once
#include <stdint.h>

#include "lfsr.h"

#define NSLM4_C 8
#define NSLM4_P 4
// floats per seed-table entry: U[32], L[10] (l00 l10 l11 l20 l21 l22 l30 l31 l32 l33), 1/diag[4], pad[2]
#define NSLM4_ENT 48

static inline int nslm4_q(uint16_t coef, int p) { return (int32_t) ((uint32_t) coef << (28 - 4 * p)) >> 28; }   // q_p

static inline uint16_t nslm4_pack(const int q[NSLM4_P]) {
    return (uint16_t) ((q[0] & 15) | ((q[1] & 15) << 4) | ((q[2] & 15) << 8) | ((q[3] & 15) << 12));
}
static inline float nslm4_bf2f(uint16_t h) { union { uint32_t u; float f; } v; v.u = (uint32_t) h << 16; return v.f; }
static inline uint16_t nslm4_f2bf(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    return (uint16_t) (v.u >> 16);
}

// The 8 decoded BF16 weights of a block.
void nslm4_decode_block(uint16_t seed, uint16_t coef, int e, uint16_t out[NSLM4_C]);

typedef struct {
    int n_seeds;        // seeds 1 .. n_seeds (65535: the full budget)
    int n_exp;          // exponent candidates: e0 + exp_delta[0 .. n_exp - 1] (exp_delta[0] = 0)
    int exp_delta[3];
    int refit;
} Search4Opts;

// One seed's table entry for a column group (sh: 8 floats, or NULL for 1).  Returns 0 if the Cholesky fails.
int nslm4_seed_entry(int s, const float* sh, float* ent);

// nb blocks of one column group: w[nb][8] UNSCALED, sh[8] (or NULL), exponent window [bias, bias + 15].  Writes seed,
// coef, ecode (e - bias) and decoded weighted error per block.  tab: 65536 * NSLM4_ENT floats from nslm4_seed_entry
// for this column group (entry 0 unused); ok[s] = 0 marks seeds whose entry failed.
void nslm4_search_ref(const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, int bias,
                      const Search4Opts* o, uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err);

// The same search with a lower-triangular 8 x 8 transform A (row-major) in place of sh: x = A w, U = A (S R32), error
// |A (w - w')|^2 (GPTQ: A = T^T, T the inverse of the column group's 8 x 8 block of the upper Cholesky factor of H^-1).
// sh = the diagonal of A gives nslm4_seed_entry / nslm4_search_ref's results bit for bit.
int nslm4_seed_entry_a(int s, const float A[64], float* ent);
void nslm4_search_ref_a(const float* tab, const uint8_t* ok, const float* w, int nb, const float A[64], int bias,
                        const Search4Opts* o, uint16_t* seed, uint16_t* coef, uint8_t* ecode, float* err);
