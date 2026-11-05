// nslm/searchc.h - SeedLM blocks of C <= 16 weights with P <= 8 int4 coefficients, a 4-bit exponent code and a 16-bit
// seed: the decode spec and the scalar reference search with per-weight input scales, for probes of layouts that no
// kernel decodes yet (the SeedLM paper's 3-bit layout C = 12, P = 4; or C = 16, P = 7).  At C = 8 it is nslm/searchp.h's
// search, bit for bit (tests/test_searchc.c).
//
// Decode: S[c][p] = state_(P c + p + 1) - 32768 (lfsr_states(seed, C P)), w_c = bf16(R32 * 2^e * sum_p S[c][p] q_p).
// Bits for each weight: (16 seed + 4 P coefficients + 4 exponent) / C.
#pragma once
#include <stdint.h>

#include "search4.h"   // Search4Opts, nslm4_bf2f / nslm4_f2bf

#define NSLMC_MAXC 16
#define NSLMC_MAXP 8

static inline double nslmc_bits(int C, int P) { return (20.0 + 4.0 * P) / C; }

// The C decoded BF16 weights of a block (coef: q_p in bits 4p .. 4p + 3, as nslmp_pack).
void nslmc_decode_block(int C, int P, uint16_t seed, uint32_t coef, int e, uint16_t* out);

// nb blocks w[nb][C] (unscaled) with input scales sh[nb][C] (the square roots of the channels' energies), exponents
// [bias, bias + 15]: seeds 1 .. o->n_seeds by |sh (w - w')|^2, as nslmp_search_ref.  err: the decoded error of each block.
void nslmc_search_ref(int C, int P, const float* w, const float* sh, int nb, int bias, const Search4Opts* o, uint16_t* seed,
                      uint32_t* coef, uint8_t* ecode, float* err);
