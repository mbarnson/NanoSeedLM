// nslm/trellis.h - trellis-coded weights (QTIP's bitshift trellis with its computed 1MAD code), plan T2.
//
// A row of `cols` weights is cut into blocks of T. A block is a bit stream of n = K T + L - K bits: weight t of the
// block reads the L-bit window s_t = bits [K t, K t + L) of the stream (bit j of s_t is stream bit K t + j), so each
// weight adds K new bits and shares L - K with its neighbour. Its value is scale * q(s_t), with q the integer code
//   x = 34038481 s + 76625530 (mod 2^32),  q(s) = (sum of the 4 bytes of x) - 510   in [-510, 510]
// (the sum of 4 near-uniform bytes is near-Gaussian: mean 510, standard deviation 147.8). Each block has one BF16
// scale. Consecutive windows overlap, so s_t = (s_{t-1} >> K) | (b_t << (L - K)), b_t the K new bits: a trellis with
// 2^L states and 2^K branches a state. The encoder picks the bits by a Viterbi search over the trellis, for the least
// importance-weighted squared error sum_t h_t (w_t - a q(s_t))^2.
//
// Storage: a row's blocks are concatenated (block b at bit b n), little-endian in 32-bit words (stream bit i is bit
// (i & 31) of word i >> 5), padded to whole words plus one spare word so every window is one 64-bit read.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define NSLM_TQ_MAXL 16   // states of at most 16 bits
#define NSLM_TQ_MAXK 8    // at most 8 bits a weight

static inline int nslm_tq_code(uint32_t s) {   // q(s): the 1MAD code
    const uint32_t x = s * 34038481u + 76625530u;
    return (int) ((x & 255u) + (x >> 8 & 255u) + (x >> 16 & 255u) + (x >> 24)) - 510;
}
static inline uint32_t nslm_tq_window(const uint32_t* row, uint64_t bit, int L) {   // stream bits [bit, bit + L)
    const uint64_t w = (uint64_t) row[bit >> 5] | (uint64_t) row[(bit >> 5) + 1] << 32;
    return (uint32_t) (w >> (bit & 31)) & ((1u << L) - 1u);
}
static inline uint64_t nslm_tq_block_bits(int T, int L, int K) { return (uint64_t) K * (uint64_t) T + (uint64_t) (L - K); }
// 32-bit words of one row's stream, the spare word included
static inline size_t nslm_tq_row_words(int cols, int T, int L, int K) {
    const uint64_t bits = (uint64_t) ((cols + T - 1) / T) * nslm_tq_block_bits(T, L, K);
    return (size_t) ((bits + 31) / 32) + 1;
}

// Scratch of one encoder thread: 2^L-state costs and the back pointers of a block of up to T weights.
typedef struct { float *cost, *ncost, *m, *ac; uint8_t* back; int* st; int L, K, T; } NsTqScratch;
int nslm_tq_scratch_init(NsTqScratch* z, int T, int L, int K);   // 0, or -1 (no memory, or L, K out of range)
void nslm_tq_scratch_free(NsTqScratch* z);

// One row: w[cols] with importance h[cols] (NULL: all 1) to row[nslm_tq_row_words] and scales[ceil(cols / T)] (BF16).
// Three Viterbi passes, each followed by the weighted least-squares scale for its path; the last scale is rounded to
// BF16. Writes the decoded row to out (optional). Returns sum h (w - decoded)^2.
double nslm_tq_encode_row(const float* w, const float* h, int cols, int T, NsTqScratch* z, uint32_t* row, uint16_t* scales,
                          float* out);
void nslm_tq_decode_row(const uint32_t* row, const uint16_t* scales, int cols, int T, int L, int K, float* out);
