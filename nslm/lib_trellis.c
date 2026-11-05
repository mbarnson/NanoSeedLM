// nslm/lib_trellis.c - see trellis.h.
#include "trellis.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"

static int rev_bits(int x, int n) { int r = 0; for (int i = 0; i < n; ++i) r |= (x >> i & 1) << (n - 1 - i); return r; }

int nslm_tq_scratch_init(NsTqScratch* z, int T, int L, int K) {
    memset(z, 0, sizeof *z);
    if (L < 2 || L > NSLM_TQ_MAXL || K < 1 || K > NSLM_TQ_MAXK || K >= L || T < 1) return -1;
    const size_t N = (size_t) 1 << L, Q = (size_t) 1 << (L - K);
    z->cost = malloc(sizeof(float) * N); z->ncost = malloc(sizeof(float) * N); z->ac = malloc(sizeof(float) * N);
    z->m = malloc(sizeof(float) * Q); z->jm = malloc(sizeof(float) * Q);
    z->back = malloc(Q * (size_t) T); z->st = malloc(sizeof(int) * (size_t) T); z->rev = malloc(sizeof(int) * N);
    z->L = L; z->K = K; z->T = T;
    if (!z->cost || !z->ncost || !z->ac || !z->m || !z->back || !z->st || !z->rev || !z->jm) { nslm_tq_scratch_free(z); return -1; }
    for (size_t u = 0; u < N; ++u) z->rev[u] = rev_bits((int) u, L);
    return 0;
}
void nslm_tq_scratch_free(NsTqScratch* z) {
    free(z->cost); free(z->ncost); free(z->ac); free(z->m); free(z->back); free(z->st); free(z->rev); free(z->jm);
    memset(z, 0, sizeof *z);
}

// The least sum_t h_t (w_t - a q(s_t))^2 over all paths, on bit-reversed labels u = rev(s): the window update
// s_t = s_{t-1} >> K | b << (L - K) becomes u_t = (u_{t-1} << K | b') mod 2^L, whose predecessors (u >> K) | j << (L - K)
// lie in 2^K contiguous runs, so both loops vectorize. Path (windows s_t) into z->st.
static void viterbi(const float* w, const float* h, int n, float a, NsTqScratch* z) {
    const int L = z->L, K = z->K, N = 1 << L, Q = 1 << (L - K), H = 1 << K;
    float* restrict cost = z->cost, *restrict ncost = z->ncost, *restrict m = z->m, *restrict ac = z->ac;
    const int* restrict rev = z->rev;
    for (int u = 0; u < N; ++u) ac[u] = a * (float) nslm_tq_code((uint32_t) rev[u]);
    for (int u = 0; u < N; ++u) { const float d = w[0] - ac[u]; cost[u] = (h ? h[0] : 1.0f) * d * d; }
    for (int t = 1; t < n; ++t) {
        float* restrict jm = z->jm;   // the argmin as a float (exact below 2^24), so both selects follow one compare
        uint8_t* restrict bk = z->back + (size_t) t * Q;
        for (int q = 0; q < Q; ++q) { m[q] = cost[q]; jm[q] = 0.0f; }
        for (int j = 1; j < H; ++j) {   // j: the K bits that drop out of s, in order, so a tie keeps the lowest j
            const int r = rev_bits(j, K);
            const float* restrict c = cost + (size_t) r * Q;
            const float fr = (float) r;
            for (int q = 0; q < Q; ++q) { const float cq = c[q], mq = m[q], jq = jm[q]; m[q] = cq < mq ? cq : mq; jm[q] = cq < mq ? fr : jq; }
        }
        for (int q = 0; q < Q; ++q) bk[q] = (uint8_t) jm[q];
        const float wt = w[t], ht = h ? h[t] : 1.0f;
        for (int q = 0; q < Q; ++q) {   // the 2^K successors of group q are contiguous
            const float mq = m[q], *restrict aq = ac + (size_t) q * H;
            float* restrict nq = ncost + (size_t) q * H;
            for (int b = 0; b < H; ++b) { const float d = wt - aq[b]; nq[b] = mq + ht * d * d; }
        }
        float* sw = cost; cost = ncost; ncost = sw;
    }
    int u = 0;
    for (int i = 1; i < N; ++i) if (cost[i] < cost[u] || (cost[i] == cost[u] && rev[i] < rev[u])) u = i;   // ties: lowest s
    for (int t = n - 1; t > 0; --t) {
        z->st[t] = rev[u];
        const int q = u >> K;
        u = q | z->back[(size_t) t * Q + q] << (L - K);
    }
    z->st[0] = rev[u];
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
