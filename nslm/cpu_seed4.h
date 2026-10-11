// nslm/cpu_seed4.h - P = 4 SeedLM mat-vec on the CPU (C99, Clang vector extensions): the weights are regenerated from
// their seeds inside the dot product, 16 rows (16 independent LFSR blocks) per inner step.
//
// y_r = sum_b dot(W[r, 8b .. 8b + 7], x[8b .. 8b + 7]) for a matrix of rows x kb blocks (seeds and coefs [rows][kb],
// codes: two 4-bit exponent codes per byte, block g = r * kb + b in nibble g & 1 of byte g >> 1), e = bias + code.
// Per block, with v[c][p] = 1 + state_(4c+p+1) / 2^16 (exact in f32) and X_b = sum_c x_c:
//   sum_c x_c isum_c / 2^16 = sum_p q_p A_p - 1.5 (sum_p q_p) X_b,   A_p = sum_c x_c v[c][p]
// The states come from the 48-bit stream window s | table[s] << 16 (table = lfsr_stream32): no serial LFSR steps.
#pragma once
#include <stdint.h>

// The 65536-entry stream table (lfsr_stream32 of every seed), built once.
uint32_t* cpu_seed4_table(void);

// Rows [r0, r1) of y.
void cpu_seed4_matvec_rows(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                           const uint32_t* table, const float* x, int kb, int r0, int r1, float* y);

// All rows on `threads` threads (dynamic claiming of 64-row chunks; threads <= 1: the calling thread).
void cpu_seed4_matvec_mt(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                         const uint32_t* table, const float* x, int rows, int kb, float* y, int threads);

// The same matrix tiled for the CPU: per tile of CPU_SEED4_T rows, block b's T seeds and T coefficient words are
// contiguous, and per pair of blocks T code bytes (block 2j in the low nibble): vector loads instead of gathers. Same
// bytes per block; a partial last tile is padded with zero coefficients. kb must be even.
#define CPU_SEED4_T 16
typedef struct {
    uint16_t *seeds, *coefs;
    uint8_t* codes;
    int bias, rows, kb, tiles;
} CpuSeed4Tiled;
void cpu_seed4_tile(CpuSeed4Tiled* t, const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                    int rows, int kb);
void cpu_seed4_tiled_free(CpuSeed4Tiled* t);
void cpu_seed4_tiled_matvec_mt(const CpuSeed4Tiled* t, const uint32_t* table, const float* x, float* y, int threads);
