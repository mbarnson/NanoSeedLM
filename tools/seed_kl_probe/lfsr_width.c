// lfsr_width.c - probe: weight error of SeedLM blocks (C = 8, P = 3 or 4, int4 q, shared exponent) against the
// width K of the LFSR (seeds 1 .. 2^K - 1), on real expert rows.  Same search as nslm/lib_searchp.c (f32 Cholesky,
// exponents e0, e0 - 1, e0 + 1, the 3^P refit by decoded BF16 error), unweighted, exponent window not clamped.
// States are the top 16 bits of the K-bit state, so the decode arithmetic (S = V - 32768, R32) is unchanged.
//
//   lfsr_width P MODE K_or_log2N gu.f32 dn.f32 NBLK
//   MODE: lfsr (seeds of a K-bit LFSR) | rand (2^N codebook entries of i.i.d. 16-bit states)
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define R32 (1.0f / 32767.0f)
#define MAGIC 12582912.0f
#define MAXE 48
static int P, K, MODE, ENT;   // MODE 0 lfsr, 1 rand
static int QB = 4;            // coefficient bits (env QB)
static int STRIDE = 1;        // LFSR steps between used states (env STRIDE)
static float QMIN = -8.0f, QMAX = 7.0f;
static uint32_t TAPS;
static uint32_t NS;           // number of codebook entries (seed ids 1 .. NS)
static float* TAB;
static uint8_t* OK;

static inline uint32_t step(uint32_t s) {
    const uint32_t b = (uint32_t) __builtin_parity(s & TAPS);
    return (s >> 1) | (b << (K - 1));
}
static uint64_t period(uint32_t taps) {
    TAPS = taps;
    uint32_t s = 1;
    uint64_t n = 0;
    do { s = step(s); ++n; } while (s != 1 && n <= (1ull << K));
    return n;
}
static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; x ^= x >> 31;
    return x;
}
// The n 16-bit states that follow seed id s.
static void states(uint32_t s, int n, uint16_t* out) {
    if (MODE == 0) {
        for (int k = 0; k < n; ++k) { for (int j = 0; j < STRIDE; ++j) s = step(s); out[k] = (uint16_t) (s >> (K - 16)); }
    } else {
        for (int k = 0; k < n; ++k) out[k] = (uint16_t) mix64(((uint64_t) s << 8) | (uint64_t) k);
    }
}
static inline float pow2f(int e) { union { uint32_t u; float f; } v; v.u = (uint32_t) (e + 127) << 23; return v.f; }
static inline int flog2f(float x) { union { uint32_t u; float f; } v; v.f = x; return (int) ((v.u >> 23) & 255u) - 127; }
static inline float clampq(float r) { return r < QMIN ? QMIN : (r > QMAX ? QMAX : r); }
static inline float bfr(float f) {   // round to BF16 (RNE) and back
    union { uint32_t u; float f; } v; v.f = f; v.u += 0x7FFFu + ((v.u >> 16) & 1u); v.u &= 0xFFFF0000u; return v.f;
}

// entry: U[8P], L lower packed (P(P+1)/2), D[P]
static int entry(uint32_t s, float* e) {
    uint16_t st[32];
    states(s, 8 * P, st);
    float* U = e;
    for (int k = 0; k < 8 * P; ++k) U[k] = (float) ((int32_t) st[k] - 32768) * R32;
    float G[4][4];
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < P; ++j) { float g = 0; for (int c = 0; c < 8; ++c) g += U[P * c + i] * U[P * c + j]; G[i][j] = g; }
    float* L = U + 8 * P, *D = L + P * (P + 1) / 2;
    for (int i = 0; i < P; ++i) {
        for (int j = 0; j <= i; ++j) {
            float a = G[i][j];
            for (int k = 0; k < j; ++k) a -= L[i * (i + 1) / 2 + k] * L[j * (j + 1) / 2 + k];
            if (i == j) { if (!(a > 0.0f)) return 0; L[i * (i + 1) / 2 + i] = sqrtf(a); D[i] = 1.0f / sqrtf(a); }
            else L[i * (i + 1) / 2 + j] = a * D[j];
        }
    }
    return 1;
}
static inline void solve(const float* L, const float* D, const float* b, float* t) {
    float y[4];
    for (int i = 0; i < P; ++i) { float a = b[i]; for (int k = 0; k < i; ++k) a -= L[i * (i + 1) / 2 + k] * y[k]; y[i] = a * D[i]; }
    for (int i = P - 1; i >= 0; --i) { float a = y[i]; for (int k = i + 1; k < P; ++k) a -= L[k * (k + 1) / 2 + i] * t[k]; t[i] = a * D[i]; }
}
static inline float cand(const float* L, const float* q, const float* b, float wn, float sc) {
    float qb = 0, qgq = 0;
    for (int p = 0; p < P; ++p) qb += q[p] * b[p];
    for (int j = 0; j < P; ++j) { float z = 0; for (int i = j; i < P; ++i) z += L[i * (i + 1) / 2 + j] * q[i]; qgq += z * z; }
    const float rec = sc * sc * qgq;
    return rec > 4.0f * wn ? INFINITY : (wn - 2.0f * sc * qb) + rec;
}
static float decoded(const float* x, uint32_t s, int e, const int* q) {
    uint16_t st[32];
    states(s, 8 * P, st);
    const float sc = R32 * pow2f(e);
    float er = 0;
    for (int c = 0; c < 8; ++c) {
        int32_t is = 0;
        for (int p = 0; p < P; ++p) is += ((int32_t) st[P * c + p] - 32768) * q[p];
        const float d = x[c] - bfr((float) is * sc);
        er += d * d;
    }
    return er;
}

static float* X;        // [NB][8]
static int NB, NT = 4;
static float *BEST, *PROJ;
static uint32_t* BS;
static int* BE;

#define CHUNK 4096
static void* run(void* arg) {
    const int id = (int) (intptr_t) arg;
    for (uint32_t s0 = 1; s0 <= NS; s0 += CHUNK) {
        const uint32_t s1 = s0 + CHUNK - 1 < NS ? s0 + CHUNK - 1 : NS;
        for (int k = id; k < NB; k += NT) {
            const float* x = X + 8 * k;
            float wn = 0;
            for (int c = 0; c < 8; ++c) wn += x[c] * x[c];
            float best = BEST[k], proj = PROJ[k];
            uint32_t bs = BS[k];
            int be = BE[k];
            for (uint32_t s = s0; s <= s1; ++s) {
                if (!OK[s]) continue;
                const float* U = TAB + (size_t) s * ENT, *L = U + 8 * P, *D = L + P * (P + 1) / 2;
                float b[4] = {0, 0, 0, 0}, t[4];
                for (int c = 0; c < 8; ++c)
                    for (int p = 0; p < P; ++p) b[p] += U[P * c + p] * x[c];
                solve(L, D, b, t);
                float m = 0, r2 = 0;
                for (int p = 0; p < P; ++p) m = fmaxf(m, fabsf(t[p]));
                for (int c = 0; c < 8; ++c) { float r = x[c]; for (int p = 0; p < P; ++p) r -= U[P * c + p] * t[p]; r2 += r * r; }
                if (r2 < proj) proj = r2;
                const int e0 = flog2f(m) - (QB - 2);
                static const int de[3] = {0, -1, 1};
                for (int ci = 0; ci < 3; ++ci) {
                    const int e = e0 + de[ci];
                    const float inv = pow2f(-e), sc = pow2f(e);
                    float q[4];
                    for (int p = 0; p < P; ++p) q[p] = clampq((t[p] * inv + MAGIC) - MAGIC);
                    const float er = cand(L, q, b, wn, sc);
                    if (er < best) { best = er; bs = s; be = e; }
                }
            }
            BEST[k] = best; PROJ[k] = proj; BS[k] = bs; BE[k] = be;
        }
    }
    return NULL;
}

static float* load(const char* f, int* n) {
    FILE* fp = fopen(f, "rb");
    fseek(fp, 0, SEEK_END);
    const long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    float* w = malloc(sz);
    if (fread(w, 1, sz, fp) != (size_t) sz) exit(1);
    fclose(fp);
    *n = (int) (sz / 4);
    return w;
}

int main(int argc, char** argv) {
    P = atoi(argv[1]);
    MODE = strcmp(argv[2], "rand") == 0;
    K = atoi(argv[3]);
    const int nblk = atoi(argv[6]);
    ENT = 8 * P + P * (P + 1) / 2 + P;
    if (getenv("STRIDE")) STRIDE = atoi(getenv("STRIDE"));
    if (getenv("QB")) { QB = atoi(getenv("QB")); QMIN = (float) -(1 << (QB - 1)); QMAX = (float) ((1 << (QB - 1)) - 1); }
    if (MODE == 0) {
        const uint64_t full = (1ull << K) - 1;
        uint32_t taps = 0;
        if (K == 16) taps = (1u << 0) | (1u << 1) | (1u << 3) | (1u << 12);
        else {
            for (int j = 1; j < K && !taps; ++j) if (period(1u | (1u << j)) == full) taps = 1u | (1u << j);
            for (int a = 1; a < K && !taps; ++a)
                for (int b2 = a + 1; b2 < K && !taps; ++b2)
                    for (int c = b2 + 1; c < K && !taps; ++c)
                        if (period(1u | (1u << a) | (1u << b2) | (1u << c)) == full) taps = 1u | (1u << a) | (1u << b2) | (1u << c);
        }
        if (period(taps) != full) { fprintf(stderr, "no primitive taps\n"); return 1; }
        TAPS = taps;
        NS = (uint32_t) full;
        fprintf(stderr, "K %d taps 0x%x period %llu\n", K, taps, (unsigned long long) full);
    } else NS = (1u << K) - 1;
    // blocks: nblk from gu (3/4) and dn (1/4), spread evenly
    int ngu, ndn;
    float* gu = load(argv[4], &ngu), *dn = load(argv[5], &ndn);
    const int bg = nblk * 3 / 4, bd = nblk - bg;
    NB = nblk;
    X = malloc(sizeof(float) * 8 * NB);
    for (int i = 0; i < bg; ++i) memcpy(X + 8 * i, gu + 8 * (size_t) ((long) i * (ngu / 8) / bg), 32);
    for (int i = 0; i < bd; ++i) memcpy(X + 8 * (bg + i), dn + 8 * (size_t) ((long) i * (ndn / 8) / bd), 32);
    TAB = malloc(sizeof(float) * ENT * ((size_t) NS + 1));
    OK = calloc((size_t) NS + 1, 1);
    struct timespec t0, t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (uint32_t s = 1; s <= NS; ++s) OK[s] = (uint8_t) entry(s, TAB + (size_t) s * ENT);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    BEST = malloc(4 * NB); PROJ = malloc(4 * NB); BS = malloc(4 * NB); BE = malloc(4 * NB);
    for (int k = 0; k < NB; ++k) { BEST[k] = INFINITY; PROJ[k] = INFINITY; BS[k] = 1; BE[k] = 0; }
    pthread_t th[64];
    for (int i = 0; i < NT; ++i) pthread_create(&th[i], NULL, run, (void*) (intptr_t) i);
    for (int i = 0; i < NT; ++i) pthread_join(th[i], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    // refit and totals
    double sw[2] = {0, 0}, se[2] = {0, 0}, sp[2] = {0, 0};
    uint32_t nok = 0;
    for (uint32_t s = 1; s <= NS; ++s) nok += OK[s];
    for (int k = 0; k < NB; ++k) {
        const float* x = X + 8 * k;
        float wn = 0;
        for (int c = 0; c < 8; ++c) wn += x[c] * x[c];
        float err = wn;
        if (isfinite(BEST[k])) {
            const float* U = TAB + (size_t) BS[k] * ENT, *L = U + 8 * P, *D = L + P * (P + 1) / 2;
            float b[4] = {0, 0, 0, 0}, t[4];
            for (int c = 0; c < 8; ++c) for (int p = 0; p < P; ++p) b[p] += U[P * c + p] * x[c];
            solve(L, D, b, t);
            int q[4], bq[4];
            const float inv = pow2f(-BE[k]);
            for (int p = 0; p < P; ++p) bq[p] = q[p] = (int) clampq((t[p] * inv + MAGIC) - MAGIC);
            err = decoded(x, BS[k], BE[k], q);
            int nn = 1;
            for (int p = 0; p < P; ++p) nn *= 3;
            for (int d = 0; d < nn; ++d) {
                int cq[4], okq = 1, dd = d;
                for (int p = 0; p < P; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq &= cq[p] >= (int) QMIN && cq[p] <= (int) QMAX; }
                if (!okq) continue;
                const float ee = decoded(x, BS[k], BE[k], cq);
                if (ee < err) err = ee;
            }
        }
        const int g = k < bg ? 0 : 1;
        sw[g] += wn; se[g] += err; sp[g] += PROJ[k] > 0 ? PROJ[k] : 0;
    }
    const double ts = (t2.tv_sec - t1.tv_sec) + 1e-9 * (t2.tv_nsec - t1.tv_nsec);
    printf("P %d q%d st%d %s %2d  seeds %9u ok %9u  mse gu %.5f dn %.5f all %.5f  proj-only all %.5f  search %.1f s (%.1f ns/seed/blk/thread)\n",
           P, QB, STRIDE, MODE ? "rand" : "lfsr", K, NS, nok, se[0] / sw[0], se[1] / sw[1], (se[0] + se[1]) / (sw[0] + sw[1]),
           (sp[0] + sp[1]) / (sw[0] + sw[1]), ts, ts * 1e9 * NT / ((double) NS * NB));
    return 0;
}
