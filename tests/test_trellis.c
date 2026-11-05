// tests/test_trellis.c - trellis-coded rows (nslm/trellis.h): error on Gaussian rows near the bound for each K, the
// decoder equal to the encoder's own output, importance weights, the same words from run to run, short last blocks, and
// the same words as a plain Viterbi on the window labels (the tie rules a GPU encoder must keep).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"
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

// The reference encoder: the same search on the window labels s, with the tie rules of trellis.h, one weight at a time
static void ref_encode(const float* w, const float* h, int cols, int T, int L, int K, uint32_t* row, uint16_t* sc) {
    const int N = 1 << L, Q = 1 << (L - K), H = 1 << K;
    float* cost = malloc(sizeof(float) * N), *nc = malloc(sizeof(float) * N);
    uint8_t* back = malloc((size_t) T * N);
    int* st = malloc(sizeof(int) * T);
    const uint64_t nbits = nslm_tq_block_bits(T, L, K);
    memset(row, 0, sizeof(uint32_t) * nslm_tq_row_words(cols, T, L, K));
    for (int b = 0, j0 = 0; j0 < cols; ++b, j0 += T) {
        const int n = cols - j0 < T ? cols - j0 : T;
        const float* x = w + j0, *hx = h ? h + j0 : NULL;
        double sw = 0, sh = 0;
        for (int t = 0; t < n; ++t) { const double ht = hx ? hx[t] : 1.0; sw += ht * x[t] * x[t]; sh += ht; }
        float a = sh > 0 ? (float) (sqrt(sw / sh) / 147.8) : 0.0f;
        for (int pass = 0; pass < 3 && a != 0.0f; ++pass) {
            for (int s = 0; s < N; ++s) { const float d = x[0] - a * (float) nslm_tq_code(s); cost[s] = (hx ? hx[0] : 1.0f) * d * d; }
            for (int t = 1; t < n; ++t) {
                for (int s = 0; s < N; ++s) {   // s's predecessors: (s mod 2^(L-K)) << K | j; the lowest j wins a tie
                    int jb = 0;
                    for (int j = 1; j < H; ++j) if (cost[(s & (Q - 1)) << K | j] < cost[(s & (Q - 1)) << K | jb]) jb = j;
                    const float d = x[t] - a * (float) nslm_tq_code(s);
                    nc[s] = cost[(s & (Q - 1)) << K | jb] + (hx ? hx[t] : 1.0f) * d * d;
                    back[(size_t) t * N + s] = (uint8_t) jb;
                }
                float* sw2 = cost; cost = nc; nc = sw2;
            }
            int s = 0;
            for (int i = 1; i < N; ++i) if (cost[i] < cost[s]) s = i;   // the lowest s wins a tie
            for (int t = n - 1; t >= 0; --t) { st[t] = s; if (t) s = (s & (Q - 1)) << K | back[(size_t) t * N + s]; }
            double num = 0, den = 0;
            for (int t = 0; t < n; ++t) { const double ht = hx ? hx[t] : 1.0, q = nslm_tq_code(st[t]); num += ht * x[t] * q; den += ht * q * q; }
            if (den > 0) a = (float) (num / den);
        }
        if (a == 0.0f) memset(st, 0, sizeof(int) * n);
        sc[b] = nslm_f2bf(a);
        for (int t = 0; t < n; ++t)   // window t is stream bits [K t, K t + L): write each window, they agree on overlaps
            for (int i = 0; i < L; ++i) {
                const uint64_t bit = (uint64_t) b * nbits + (uint64_t) K * t + i;
                if (st[t] >> i & 1) row[bit >> 5] |= 1u << (bit & 31);
            }
    }
    free(cost); free(nc); free(back); free(st);
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
    {   // the same words and scales as the reference. Steps of zero importance give paths of equal cost, and a zero at
        // the next-to-last step lets more states compete at the end, where two states with one code tie
        const int cfg[5][5] = {{6, 3, 32, 100, 24}, {8, 4, 64, 200, 12}, {8, 2, 64, 200, 12}, {10, 3, 64, 160, 6}, {12, 3, 256, 300, 4}};
        for (int c = 0; c < 5; ++c) {
            const int L = cfg[c][0], K = cfg[c][1], T = cfg[c][2], cols = cfg[c][3], rows = cfg[c][4];
            const size_t nw = nslm_tq_row_words(cols, T, L, K), nb = (size_t) (cols + T - 1) / T;
            NsTqScratch z;
            nslm_tq_scratch_init(&z, T, L, K);
            float* w = malloc(sizeof(float) * cols), *hh = malloc(sizeof(float) * cols);
            for (int j = 0; j < cols; ++j) {
                const int t = j % T, n = cols - (j - t) < T ? cols - (j - t) : T;
                hh[j] = (t >= T / 4 && t < T / 4 + 3) || t == n - 2 ? 0.0f : 1.0f + (float) (j % 3);
            }
            uint32_t* a = malloc(sizeof(uint32_t) * nw), *b = malloc(sizeof(uint32_t) * nw);
            uint16_t sa[8], sb[8];
            int differ = 0;
            for (int r = 0; r < rows; ++r) {
                for (int j = 0; j < cols; ++j) w[j] = (float) (0.02 * gauss());
                if (r == rows - 1) memset(w, 0, sizeof(float) * cols);   // the zero row
                const float* hr = r % 3 ? hh : NULL;
                nslm_tq_encode_row(w, hr, cols, T, &z, a, sa, NULL);
                ref_encode(w, hr, cols, T, L, K, b, sb);
                differ += memcmp(a, b, sizeof(uint32_t) * nw) || memcmp(sa, sb, sizeof(uint16_t) * nb);
            }
            CHECK(!differ, "L %d K %d T %d: %d of %d rows differ from the reference", L, K, T, differ, rows);
            nslm_tq_scratch_free(&z);
            free(w); free(hh); free(a); free(b);
        }
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
