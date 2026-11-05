// tests/test_trellis.c - trellis-coded rows (nslm/trellis.h): error on Gaussian rows near the bound for each K, the
// decoder equal to the encoder's own output, importance weights, the same words from run to run, short last blocks.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trellis.h"

static int failures;
#define CHECK(c, ...) do { if (!(c)) { ++failures; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rs = 12345;
static double gauss(void) {   // Box-Muller on an LCG
    rs = rs * 1664525u + 1013904223u;
    const double u = ((rs >> 8) + 0.5) / 16777216.0;
    rs = rs * 1664525u + 1013904223u;
    const double v = ((rs >> 8) + 0.5) / 16777216.0;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * v);
}

// Encodes rows of N(0, 0.02^2) weights; returns sum h e^2 / sum h w^2 (h NULL: plain), checks decode == encoder output
static double run(int rows, int cols, int T, int L, int K, const float* h, uint32_t** words_out) {
    NsTqScratch z;
    CHECK(nslm_tq_scratch_init(&z, T, L, K) == 0, "scratch L %d K %d", L, K);
    const size_t nw = nslm_tq_row_words(cols, T, L, K), nb = (size_t) (cols + T - 1) / T;
    float* w = malloc(sizeof(float) * cols), *enc = malloc(sizeof(float) * cols), *dec = malloc(sizeof(float) * cols);
    uint32_t* words = malloc(sizeof(uint32_t) * nw * rows);
    uint16_t* sc = malloc(sizeof(uint16_t) * nb);
    double num = 0, den = 0;
    int differ = 0;
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < cols; ++j) w[j] = (float) (0.02 * gauss());
        num += nslm_tq_encode_row(w, h, cols, T, &z, words + nw * r, sc, enc);
        nslm_tq_decode_row(words + nw * r, sc, cols, T, L, K, dec);
        differ += memcmp(enc, dec, sizeof(float) * cols) != 0;
        for (int j = 0; j < cols; ++j) den += (h ? h[j] : 1.0) * (double) w[j] * w[j];
    }
    CHECK(!differ, "L %d K %d T %d: the decoder differs from the encoder on %d rows", L, K, T, differ);
    nslm_tq_scratch_free(&z);
    free(w); free(enc); free(dec); free(sc);
    if (words_out) *words_out = words; else free(words);
    return num / den;
}

int main(void) {
    // The bound for Gaussian weights at R bits is 2^-2R: 0.0625, 0.0156, 0.0039 at 2, 3, 4. L = 12 comes within 14%, 20%
    // and 33% of it; the limits leave a margin
    const double bound[5] = {0, 0, 0.0625, 0.015625, 0.00390625}, limit[5] = {0, 0, 1.2, 1.3, 1.45};
    for (int K = 2; K <= 4; ++K) {
        const double e = run(8, 2560, 256, 12, K, NULL, NULL);
        printf("L 12 K %d, rows of 2560 in blocks of 256 (%.3f bits a weight): relative error %.5f (bound %.5f)\n", K,
               K + (12.0 - K + 16) / 256, e, bound[K]);
        CHECK(e > bound[K] && e < limit[K] * bound[K], "L 12 K %d: error %.5f, bound %.5f", K, e, bound[K]);
    }
    const double e16 = run(2, 256, 256, 16, 3, NULL, NULL);
    printf("L 16 K 3, one block of 256: relative error %.5f\n", e16);
    CHECK(e16 < 1.15 * bound[3], "L 16 K 3: error %.5f", e16);
    const double ed = run(16, 640, 320, 12, 3, NULL, NULL);   // the experts' down rows
    printf("L 12 K 3, rows of 640 in blocks of 320: relative error %.5f\n", ed);
    CHECK(ed < limit[3] * bound[3], "down rows: error %.5f", ed);
    const double es = run(8, 300, 256, 12, 3, NULL, NULL);   // a short last block
    CHECK(es < 1.4 * bound[3], "short last block: error %.5f", es);
    {   // importance: half the columns weigh 100 times more; the weighted search must win on the weighted error
        float* h = malloc(sizeof(float) * 2560);
        for (int j = 0; j < 2560; ++j) h[j] = j % 2 ? 100.0f : 1.0f;
        const uint32_t seed = rs;
        const double ew = run(4, 2560, 256, 12, 3, h, NULL);
        rs = seed;   // the same rows, encoded without h, scored with h
        NsTqScratch z;
        nslm_tq_scratch_init(&z, 256, 12, 3);
        const size_t nw = nslm_tq_row_words(2560, 256, 12, 3);
        float* w = malloc(sizeof(float) * 2560), *y = malloc(sizeof(float) * 2560);
        uint32_t* words = malloc(sizeof(uint32_t) * nw);
        uint16_t sc[10];
        double num = 0, den = 0;
        for (int r = 0; r < 4; ++r) {
            for (int j = 0; j < 2560; ++j) w[j] = (float) (0.02 * gauss());
            nslm_tq_encode_row(w, NULL, 2560, 256, &z, words, sc, y);
            for (int j = 0; j < 2560; ++j) { const double d = w[j] - y[j]; num += h[j] * d * d; den += h[j] * (double) w[j] * w[j]; }
        }
        printf("importance-weighted error: %.5f weighted search, %.5f plain search\n", ew, num / den);
        CHECK(ew < 0.97 * (num / den), "weighted %.5f, plain %.5f", ew, num / den);
        nslm_tq_scratch_free(&z);
        free(h); free(w); free(y); free(words);
    }
    {   // the same words from run to run
        uint32_t *a, *b;
        rs = 777; run(2, 2560, 256, 12, 2, NULL, &a);
        rs = 777; run(2, 2560, 256, 12, 2, NULL, &b);
        CHECK(!memcmp(a, b, sizeof(uint32_t) * nslm_tq_row_words(2560, 256, 12, 2) * 2), "the words differ between runs");
        free(a); free(b);
    }
    {   // q(s): mean 510 and standard deviation about 147.8 before the shift, over all 2^16 states
        double m = 0, v = 0;
        for (uint32_t s = 0; s < 65536; ++s) { const double q = nslm_tq_code(s); m += q; v += q * q; }
        m /= 65536; v = sqrt(v / 65536 - m * m);
        printf("q over 2^16 states: mean %.3f, standard deviation %.3f\n", m, v);
        CHECK(fabs(m) < 2 && fabs(v - 147.8) < 3, "q: mean %.3f, standard deviation %.3f", m, v);
    }
    printf("test_trellis: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
