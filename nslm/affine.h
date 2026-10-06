// nslm/affine.h - MLX-compatible affine quantization (mx.quantize / mx.dequantize, group 64), bit for bit.
//
// Per group of 64 BF16 weights (along a row), in f32:
//   delta = max((max - min) / (2^bits - 1), 1e-7);  edge = the larger-magnitude extreme (min if |min| > |max|)
//   scale = edge == min ? delta : -delta;  q0 = round_half_away(edge / scale)
//   if q0 != 0: scale = edge / q0, bias = edge  else bias = 0
//   q = clamp(round_half_away((w - bias) / scale), 0, 2^bits - 1)
// The scale and bias are stored as BF16 (round to nearest even); the codes are packed little-endian into uint32
// words, 32 / bits per word, element i of the word at bits [bits * i, bits * (i + 1)), exactly as MLX stores them.
// Dequantization: bf16(scale * q + bias) in f32.  Checked against MLX's output by tests/test_affine.c.
#pragma once
#include <stdint.h>

// w: rows x cols BF16 (cols % 64 == 0).  words: rows * cols * bits / 32; scales, biases: rows * cols / 64 BF16.
void nslm_affine_quantize(const uint16_t* w, int rows, int cols, int bits, uint32_t* words, uint16_t* scales,
                          uint16_t* biases);
void nslm_affine_dequantize(const uint32_t* words, const uint16_t* scales, const uint16_t* biases, int rows, int cols,
                            int bits, uint16_t* out);
