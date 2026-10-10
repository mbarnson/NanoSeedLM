// nslm/searchp.h - SeedLM blocks of C = 8 weights with P int4 coefficients (P = 3 or 8), a 4-bit exponent code and a
// 16-bit seed: the decode spec and the scalar C reference seed search, with a per-column-group transform (GPTQ).
//   P = 8: SEED6P8 (52 bits per 8 weights, 6.5 bits per weight); P = 3: SEED4 (lfsr.h; 32 bits, 4.0 bits per weight).
//
// Decode: S[c][p] = state_(P c + p + 1) - 32768 (lfsr_states(seed, 8 P)), e = ecode + bias,
//   w_c = bf16( R32 * 2^e * isum_c ),  isum_c = sum_p S[c][p] * q_p  (exact integer).
// Coefficient word (32 bits): q_p (two's complement) in bits 4p .. 4p+3.  (SEED4's nibble word is ecode | word << 4.)
//
// Search, per column group: x = A w (A lower triangular, or diag(sh), or I), U = A (S * R32) (8 x P),
// G = U^T U = L L^T (f32 Cholesky), b = U^T x, t = G^-1 b, e0 = floor(log2 max|t|) - 2 (clamped to the window),
// for each exponent e0 + exp_delta[k] (clamped): q = clamp(rne(t / 2^e), -8, 7),
//   err = (|x|^2 - 2 * 2^e * q.b) + 2^2e * |L^T q|^2,  rejected (INFINITY) when 2^2e |L^T q|^2 > 4 |x|^2.
// The arg-min wins (lowest seed, then the first exponent, on exact ties).  Seeds whose Cholesky fails are skipped.
// Refit: the winner's q among its 3^P neighbours by the error of the DECODED weights, |A (w - w')|^2.
#pragma once
#include <stdint.h>

#include "lfsr.h"
#include "search4.h"   // Search4Opts, nslm4_bf2f / nslm4_f2bf

#define NSLMP_MAXP 8
// floats per seed-table entry: U[8 P], L[P (P + 1) / 2] (row-major lower triangle), 1/diag[P]
#define NSLMP_ENT(P) (8 * (P) + (P) * ((P) + 1) / 2 + (P))

static inline int nslmp_q(uint32_t coef, int p) { return (int32_t) (coef << (28 - 4 * p)) >> 28; }   // q_p

static inline uint32_t nslmp_pack(int P, const int* q) {
    uint32_t c = 0;
    for (int p = 0; p < P; ++p) c |= (uint32_t) (q[p] & 15) << (4 * p);
    return c;
}

// The 8 decoded BF16 weights of a block.
void nslmp_decode_block(int P, uint16_t seed, uint32_t coef, int e, uint16_t out[8]);

// One seed's table entry (NSLMP_ENT(P) floats) for a column group: A (64 floats, lower triangular) or sh (8 floats) or
// neither (identity).  Returns 0 if the Cholesky fails.
int nslmp_seed_entry(int P, int s, const float* sh, const float* A, float* ent);

// nb blocks of one column group: w[nb][8] UNSCALED, exponent window [bias, bias + 15]; tab: 65536 * NSLMP_ENT(P) floats
// from nslmp_seed_entry for this column group, ok[s] = 0 for failed seeds.  Writes seed, coef, ecode (e - bias) and the
// decoded weighted error per block.
void nslmp_search_ref(int P, const float* tab, const uint8_t* ok, const float* w, int nb, const float* sh, const float* A, int bias,
                      const Search4Opts* o, uint16_t* seed, uint32_t* coef, uint8_t* ecode, float* err);
