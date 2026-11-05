// nslm/lib_trellis.c - see trellis.h.
#include "trellis.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"

int nslm_tq_scratch_init(NsTqScratch* z, int T, int L, int K) {
    memset(z, 0, sizeof *z);
    if (L < 2 || L > NSLM_TQ_MAXL || K < 1 || K > NSLM_TQ_MAXK || K >= L || T < 1) return -1;
    const size_t N = (size_t) 1 << L, Q = (size_t) 1 << (L - K);
    z->cost = malloc(sizeof(float) * N); z->ncost = malloc(sizeof(float) * N); z->ac = malloc(sizeof(float) * N);
    z->m = malloc(sizeof(float) * Q);
    z->back = malloc(Q * (size_t) T); z->st = malloc(sizeof(int) * (size_t) T);
    z->L = L; z->K = K; z->T = T;
    if (!z->cost || !z->ncost || !z->ac || !z->m || !z->back || !z->st) { nslm_tq_scratch_free(z); return -1; }
    return 0;
}
void nslm_tq_scratch_free(NsTqScratch* z) {
    free(z->cost); free(z->ncost); free(z->ac); free(z->m); free(z->back); free(z->st);
    memset(z, 0, sizeof *z);
}

// The least sum_t h_t (w_t - a q(s_t))^2 over all paths: s_t's predecessors are ((s_t mod 2^(L-K)) << K) | j,
// j < 2^K, contiguous, so one min over them serves the 2^K states that share s_t's low L - K bits. Path into z->st.
static void viterbi(const float* w, const float* h, int n, float a, NsTqScratch* z) {
    const int L = z->L, K = z->K, N = 1 << L, Q = 1 << (L - K), H = 1 << K;
    for (int s = 0; s < N; ++s) z->ac[s] = a * (float) nslm_tq_code((uint32_t) s);
    for (int s = 0; s < N; ++s) { const float d = w[0] - z->ac[s]; z->cost[s] = (h ? h[0] : 1.0f) * d * d; }
    for (int t = 1; t < n; ++t) {
        uint8_t* bk = z->back + (size_t) t * Q;
        for (int q = 0; q < Q; ++q) {
            const float* c = z->cost + (size_t) q * H;
            int b = 0;
            for (int j = 1; j < H; ++j) if (c[j] < c[b]) b = j;
            z->m[q] = c[b];
            bk[q] = (uint8_t) b;
        }
        const float wt = w[t], ht = h ? h[t] : 1.0f;
        for (int hi = 0; hi < H; ++hi)
            for (int q = 0; q < Q; ++q) {
                const int s = q | hi << (L - K);
                const float d = wt - z->ac[s];
                z->ncost[s] = z->m[q] + ht * d * d;
            }
        float* sw = z->cost; z->cost = z->ncost; z->ncost = sw;
    }
    int s = 0;
    for (int i = 1; i < N; ++i) if (z->cost[i] < z->cost[s]) s = i;
    for (int t = n - 1; t > 0; --t) {
        z->st[t] = s;
        const int q = s & (Q - 1);
        s = q << K | z->back[(size_t) t * Q + q];
    }
    z->st[0] = s;
}

static void put_bits(uint32_t* row, uint64_t bit, uint32_t v, int nb) {   // nb <= 16 bits of v at stream bit `bit`
    for (int i = 0; i < nb; ++i, ++bit)
        if (v >> i & 1u) row[bit >> 5] |= 1u << (bit & 31); else row[bit >> 5] &= ~(1u << (bit & 31));
}

double nslm_tq_encode_row(const float* w, const float* h, int cols, int T, NsTqScratch* z, uint32_t* row, uint16_t* scales,
                          float* out) {
    const int L = z->L, K = z->K;
    const uint64_t nbits = nslm_tq_block_bits(T, L, K);
    memset(row, 0, sizeof(uint32_t) * nslm_tq_row_words(cols, T, L, K));
    double err = 0;
    for (int b = 0, j0 = 0; j0 < cols; ++b, j0 += T) {
        const int n = cols - j0 < T ? cols - j0 : T;
        const float* wb = w + j0, *hb = h ? h + j0 : NULL;
        double sw = 0, sh = 0;
        for (int t = 0; t < n; ++t) { const double ht = hb ? hb[t] : 1.0; sw += ht * wb[t] * wb[t]; sh += ht; }
        float a = sh > 0 ? (float) (sqrt(sw / sh) / 147.8) : 0.0f;   // q has standard deviation about 147.8
        for (int pass = 0; pass < 3 && a != 0.0f; ++pass) {
            viterbi(wb, hb, n, a, z);
            double num = 0, den = 0;
            for (int t = 0; t < n; ++t) {
                const double ht = hb ? hb[t] : 1.0, q = nslm_tq_code((uint32_t) z->st[t]);
                num += ht * wb[t] * q; den += ht * q * q;
            }
            if (den > 0) a = (float) (num / den);
        }
        if (a == 0.0f) memset(z->st, 0, sizeof(int) * (size_t) n);
        scales[b] = nslm_f2bf(a);
        const float ar = nslm_bf2f(scales[b]);
        const uint64_t base = (uint64_t) b * nbits;
        put_bits(row, base, (uint32_t) z->st[0], L);
        for (int t = 1; t < n; ++t) put_bits(row, base + (uint64_t) K * t + (uint64_t) (L - K), (uint32_t) z->st[t] >> (L - K), K);
        for (int t = 0; t < n; ++t) {
            const float v = ar * (float) nslm_tq_code((uint32_t) z->st[t]), d = wb[t] - v;
            err += (hb ? hb[t] : 1.0) * d * d;
            if (out) out[j0 + t] = v;
        }
    }
    return err;
}

void nslm_tq_decode_row(const uint32_t* row, const uint16_t* scales, int cols, int T, int L, int K, float* out) {
    const uint64_t nbits = nslm_tq_block_bits(T, L, K);
    for (int j = 0; j < cols; ++j) {
        const int b = j / T, t = j % T;
        out[j] = nslm_bf2f(scales[b]) * (float) nslm_tq_code(nslm_tq_window(row, (uint64_t) b * nbits + (uint64_t) K * t, L));
    }
}
