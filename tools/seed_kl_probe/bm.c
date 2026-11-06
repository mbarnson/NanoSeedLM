// bm.c - probe: Berlekamp-Massey linear complexity (the shortest LFSR that outputs the exact bits) of real BF16 expert
// weights, against an LFSR stream as a control.   bm gu.f32 NWEIGHTS
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lfsr.h"
static int bm(const uint8_t* s, int n) {   // GF(2) Berlekamp-Massey, returns the linear complexity L
    const int W = (n + 64) / 64;
    uint64_t* C = calloc(W, 8), *B = calloc(W, 8), *T = calloc(W, 8);
    C[0] = B[0] = 1;
    int L = 0, m = -1;
    for (int i = 0; i < n; ++i) {
        int d = s[i];
        for (int j = 1; j <= L; ++j) d ^= (int) ((C[j >> 6] >> (j & 63)) & 1) & s[i - j];
        if (!d) continue;
        memcpy(T, C, W * 8);
        const int sh = i - m;   // C ^= B << sh
        for (int k = W - 1; k >= 0; --k) {
            const int ws = sh >> 6, bs = sh & 63;
            uint64_t v = 0;
            if (k - ws >= 0) v = B[k - ws] << bs;
            if (bs && k - ws - 1 >= 0) v |= B[k - ws - 1] >> (64 - bs);
            C[k] ^= v;
        }
        if (2 * L <= i) { L = i + 1 - L; m = i; memcpy(B, T, W * 8); }
    }
    free(C); free(B); free(T);
    return L;
}
int main(int argc, char** argv) {
    FILE* f = fopen(argv[1], "rb");
    const int nw = atoi(argv[2]);
    float* w = malloc(4 * nw);
    if (fread(w, 4, nw, f) != (size_t) nw) return 1;
    const int n = 16 * nw;
    uint8_t* s = malloc(n);
    // control: the 16-bit LFSR's stream from seed 12345
    uint16_t st = 12345;
    for (int i = 0; i < n; ++i) { st = lfsr_step(st); s[i] = st >> 15; }
    printf("LFSR stream, %d bits: linear complexity %d\n", n, bm(s, n));
    for (int i = 0; i < nw; ++i) {   // BF16 bits (the upper half of the f32), LSB first
        uint32_t u; memcpy(&u, w + i, 4); const uint16_t h = (uint16_t) (u >> 16);
        for (int b = 0; b < 16; ++b) s[16 * i + b] = (h >> b) & 1;
    }
    printf("BF16 weights, %d weights = %d bits: linear complexity %d (n/2 = %d)\n", nw, n, bm(s, n), n / 2);
    // sign + exponent bits only (the compressible part)
    int k = 0;
    for (int i = 0; i < nw; ++i) { uint32_t u; memcpy(&u, w + i, 4); const uint16_t h = (uint16_t) (u >> 16); for (int b = 7; b < 16; ++b) s[k++] = (h >> b) & 1; }
    printf("sign + exponent bits only, %d bits: linear complexity %d (n/2 = %d)\n", k, bm(s, k), k / 2);
    return 0;
}
