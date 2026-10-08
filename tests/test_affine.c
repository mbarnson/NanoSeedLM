// tests/test_affine.c - the MLX-compatible affine quantizer (nslm/affine.h) against MLX's own output, bit for bit:
// packed codes, BF16 scales and biases, and the dequantized BF16 weights, at 8 and 4 bits, group 64, on real MoVA
// tensors and a synthetic one with edge cases (golden files: tools/mova_affine_golden.py -> out/test/affine).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "affine.h"

static void* slurp(const char* path, size_t want) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    void* b = malloc(want);
    const size_t got = fread(b, 1, want, f);
    fclose(f);
    if (got != want) { free(b); return NULL; }
    return b;
}

int main(void) {
    const char* dir = "out/test/affine";
    char path[512], name[128];
    snprintf(path, sizeof path, "%s/index.txt", dir);
    FILE* idx = fopen(path, "r");
    if (!idx) { printf("SKIP: no golden files (run tools/mova_affine_golden.py)\n"); return 77; }
    int fails = 0, n = 0, rows, cols;
    while (fscanf(idx, "%127s %d %d", name, &rows, &cols) == 3) {
        const size_t N = (size_t) rows * cols, G = N / 64;
        snprintf(path, sizeof path, "%s/%s.bf16", dir, name);
        uint16_t* w = slurp(path, 2 * N);
        if (!w) { printf("FAIL: %s\n", path); return 1; }
        for (int bits = 8; bits >= 4; bits -= 4) {
            const size_t nw = N * (size_t) bits / 32;
            char p2[512];
            snprintf(p2, sizeof p2, "%s/%s.q%d.w", dir, name, bits);
            uint32_t* W = slurp(p2, 4 * nw);
            snprintf(p2, sizeof p2, "%s/%s.q%d.s", dir, name, bits);
            uint16_t* S = slurp(p2, 2 * G);
            snprintf(p2, sizeof p2, "%s/%s.q%d.b", dir, name, bits);
            uint16_t* B = slurp(p2, 2 * G);
            snprintf(p2, sizeof p2, "%s/%s.q%d.deq", dir, name, bits);
            uint16_t* DQ = slurp(p2, 2 * N);
            if (!W || !S || !B || !DQ) { printf("FAIL: missing q%d goldens for %s\n", bits, name); return 1; }
            uint32_t* w2 = malloc(4 * nw);
            uint16_t* s2 = malloc(2 * G), *b2 = malloc(2 * G), *d2 = malloc(2 * N);
            nslm_affine_quantize(w, rows, cols, bits, w2, s2, b2);
            nslm_affine_dequantize(w2, s2, b2, rows, cols, bits, d2);
            size_t bw = 0, bs = 0, bb = 0, bd = 0;
            for (size_t i = 0; i < nw; ++i) bw += w2[i] != W[i];
            for (size_t i = 0; i < G; ++i) { bs += s2[i] != S[i]; bb += b2[i] != B[i]; }
            for (size_t i = 0; i < N; ++i) bd += d2[i] != DQ[i];
            if (bw || bs || bb || bd) {
                ++fails;
                printf("FAIL %s q%d: words %zu, scales %zu, biases %zu, dequantized %zu differ\n", name, bits, bw, bs, bb, bd);
            }
            free(W); free(S); free(B); free(DQ); free(w2); free(s2); free(b2); free(d2);
        }
        free(w);
        ++n;
    }
    fclose(idx);
    printf("%d tensors x {q8, q4}: %s\n", n, fails ? "FAIL" : "PASS");
    return fails != 0;
}
