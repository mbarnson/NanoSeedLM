// nslm/search8.h - 6.5-bit SeedLM blocks: C = 8 weights, P = 8 int4 coefficients, 4-bit exponent code, 16-bit seed
// (52 bits per 8 weights).  Decode spec and the scalar C reference seed search.
//
// Decode: S[c][p] = state_(8c+p+1) - 32768 (lfsr_states(seed, 64)), e = ecode + bias,
//   w_c = bf16( R32 * 2^e * isum_c ),  isum_c = sum_p S[c][p] * q_p  (exact integer).
// Coefficient word (32 bits): q_p (two's complement) in bits 4p .. 4p+3.
//
// Search, per column group: x = A w (A lower triangular, or diag(sh), or I), U = A (S * R32) (8 x 8),
// G = U^T U = L L^T (f32 Cholesky), b = U^T x, t = G^-1 b, e0 = floor(log2 max|t|) - 2 (clamped to the window),
// for each exponent e0 + exp_delta[k] (clamped): q = clamp(rne(t / 2^e), -8, 7),
//   err = (|x|^2 - 2 * 2^e * q.b) + 2^2e * |L^T q|^2,  rejected (INFINITY) when 2^2e |L^T q|^2 > 4 |x|^2.
// The arg-min wins (lowest seed, then the first exponent, on exact ties).  Seeds whose Cholesky fails are skipped.
// Refit: the winner's q among its 3^8 neighbours by the error of the DECODED weights, |A (w - w')|^2.
#pragma once
#include <stdint.h>

#include "lfsr.h"
#include "search4.h"   // Search4Opts, nslm4_bf2f / nslm4_f2bf

#define NSLM8_C 8
#define NSLM8_P 8
// floats per seed-table entry: U[64], L[36] (row-major lower triangle), 1/diag[8]
#define NSLM8_ENT 108

static inline int nslm8_q(uint32_t coef, int p) { return (int32_t) (coef << (28 - 4 * p)) >> 28; }   // q_p

static inline uint32_t nslm8_pack(const int q[NSLM8_P]) {
    uint32_t c = 0;
    for (int p = 0; p < NSLM8_P; ++p) c |= (uint32_t) (q[p] & 15) << (4 * p);
    return c;
}

// The 8 decoded BF16 weights of a block.
void nslm8_decode_block(uint16_t seed, uint32_t coef, int e, uint16_t out[NSLM8_C]);

// One seed's table entry for a column group: A (64 floats, lower triangular) or sh (8 floats) or neither (identity).
// Returns 0 if the Cholesky fails.
int nslm8_seed_entry(int s, const float* sh, const float* A, float* ent);

// nb blocks of one column group: w[nb][8] UNSCALED, exponent window [bias, bias + 15]; tab: 65536 * NSLM8_ENT floats
// from nslm8_seed_entry for this column group, ok[s] = 0 for failed seeds.  Writes seed, coef, ecode (e - bias) and the
// decoded weighted error per block.
void nslm8_search_ref(const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, const float* A, int bias,
                      const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err);
