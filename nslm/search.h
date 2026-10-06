// nslm/search.h - SeedLM seed search for 4-bit blocks (C = 8, P = 3, K = 16): paper Algorithm 1.
//
// Per block w and seed s: b = U(s)^T w, t = (U^T U)^-1 b, E = max_i floor(log2 |t_i|), e = E - 2,
// q = clamp(rne(t / 2^e), -8, 7), error eps = |w|^2 - 2 * 2^e * q.b + 2^2e * q^T G q (no reconstruction).
// q^T G q = |R q|^2 (R = Cholesky factor): stays >= 0 and accurate for near-singular U(s).
// Candidates with 2^2e |R q|^2 > 4 |w|^2 are rejected: they cannot beat q = 0, and every surviving term stays
// O(|w|^2), so f32 cancellation cannot make a spurious minimum.
// Ties: lowest seed (seeds ascend), then the earlier exponent candidate.
//   exponent search  also try e + exp_delta[1..] per seed (e - 1 finer, e + 1 coarser)
//   refit            for the winning (seed, e), pick q among the 3^P neighbours of rne(t / 2^e) by the error of the
//                    DECODED BF16 weights (as nslm_decode_block)
// The exponent code is relative to a per-tensor bias: e in [bias, bias + 15].
// nslm_search_ref (scalar C) is the oracle; nslm_search_vec (Clang vectors) must return identical (seed, nibble):
// same operations, same order, no FMA contraction (-ffp-contract=off).
#pragma once
#include <stdint.h>

#include "lfsr.h"

typedef struct {
    float U[NSLM_CP];   // U(s) row-major [c][p]
    float Gi[6];        // (U^T U)^-1: 00 01 02 11 12 22
    float R[6];         // Cholesky factor of U^T U = R^T R (upper): 00 01 02 11 12 22, so q^T G q = |R q|^2 >= 0
    float pad[4];       // 40 floats = 160 bytes
} SeedTab;

typedef struct {
    int n_seeds;        // seeds 1 .. n_seeds (65535 = the full budget)
    int n_exp;          // exponent candidates per seed: the paper's e, then e + exp_delta[1], e + exp_delta[2]
    int exp_delta[3];   // exp_delta[0] must be 0 (the paper's exponent is always evaluated first)
    int refit;
} SearchOpts;

// tab[65536]; tab[s] for s in 1..65535.  Returns the number of exactly singular seeds (never selected; 0 for K = 16).
int nslm_seedtab_build(SeedTab* tab);

// Searches nb blocks w[nb][8], exponent range [bias, bias + 15].  Writes seed[nb], nib[nb] and, if err is non-NULL,
// the squared error of the decoded BF16 block.  Blocks are independent; results do not depend on nb.
// Activation-weighted: with sh = sqrt(h) per input channel (h = mean squared activation), pass w pre-scaled
// (w_c * sh_c), a table from nslm_seedtab_build_weighted(sh), and sh (refit error = sum_c h_c (w_c - w_hat_c)^2).
// sh = NULL: unweighted (data-free).
void nslm_search_ref(const SeedTab* tab, const float* w, int nb, int bias, const SearchOpts* o, uint16_t* seed,
                     uint16_t* nib, float* err, const float* sh);
void nslm_search_vec(const SeedTab* tab, const float* w, int nb, int bias, const SearchOpts* o, uint16_t* seed,
                     uint16_t* nib, float* err, const float* sh);
// The table for one column group: U(s) rows scaled by sh[8] (seeds s0 .. s1-1; for parallel builds).
void nslm_seedtab_build_weighted(SeedTab* tab, const float sh[8], int s0, int s1);

// The search's error of one candidate (seed table entry T, exponent e already inside the window) for block x with
// wn = sum x^2: the expression the search loop minimises (INFINITY when rejected).  For checking other searches.
float nslm_candidate_err(const SeedTab* T, const float* x, float wn, int e);

// The search's choice of exponent bias for a tensor: the top of the 16-exponent window sits one above the largest
// block's floor(log2(2 |w|_2)) - 2.  *clamped_low (optional) counts blocks whose norm is below the window.
int nslm_choose_bias(const float* w, int64_t nblocks, int64_t* clamped_low);
