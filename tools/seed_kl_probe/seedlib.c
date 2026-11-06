// seedlib.c - probe library (ctypes): SeedLM block search for one column group, any LFSR width K, P = 3 or 4.
// Blocks are 8 weights of one row in one column group.  Metric |A (w - w')|^2 with a lower-triangular 8 x 8 A per call
// (identity, diag(sqrt h), or GPTQ's T^T), as nslm/lib_searchp.c: f32 Cholesky, exponents e0, e0 - 1, e0 + 1 with
// e0 = floor(log2 max|t|) - 2, int4 q, then the 3^P refit by decoded BF16 error.  Exponent window not clamped.
// States: top 16 bits of a K-bit Fibonacci LFSR, taking every STRIDE-th state.  Vectorized over 8 blocks
// (ext_vector_type), threads over blocks, seeds in chunks.
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef float v8f __attribute__((ext_vector_type(8)));
typedef int v8i __attribute__((ext_vector_type(8)));

#define R32 (1.0f / 32767.0f)
#define MAGIC 12582912.0f
#define NT 4
#define CHUNK 2048

static int gP, gK, gSTRIDE, gENT;
static uint32_t gTAPS, gNS;
static int16_t* gS;   // [NS + 1][8P] centred states
static float* gTAB;   // [NS + 1][ENT]
static uint8_t* gOK;

static inline uint32_t step(uint32_t s) { return (s >> 1) | ((uint32_t) __builtin_parity(s & gTAPS) << (gK - 1)); }
static uint64_t period(uint32_t taps) {
    gTAPS = taps;
    uint32_t s = 1;
    uint64_t n = 0;
    do { s = step(s); ++n; } while (s != 1 && n <= (1ull << gK));
    return n;
}

// Returns the tap mask, or 0.
uint32_t seed_setup(int P, int K, int stride) {
    gP = P; gK = K; gSTRIDE = stride; gENT = 8 * P + P * (P + 1) / 2 + P;
    const uint64_t full = (1ull << K) - 1;
    uint32_t taps = 0;
    if (K == 16) taps = (1u << 0) | (1u << 1) | (1u << 3) | (1u << 12);
    else {
        for (int j = 1; j < K && !taps; ++j) if (period(1u | (1u << j)) == full) taps = 1u | (1u << j);
        for (int a = 1; a < K && !taps; ++a)
            for (int b = a + 1; b < K && !taps; ++b)
                for (int c = b + 1; c < K && !taps; ++c)
                    if (period(1u | (1u << a) | (1u << b) | (1u << c)) == full) taps = 1u | (1u << a) | (1u << b) | (1u << c);
    }
    if (!taps || period(taps) != full) return 0;
    gTAPS = taps;
    gNS = (uint32_t) full;
    free(gS); free(gTAB); free(gOK);
    gS = malloc(sizeof(int16_t) * 8 * P * ((size_t) gNS + 1));
    gTAB = malloc(sizeof(float) * gENT * ((size_t) gNS + 1));
    gOK = calloc((size_t) gNS + 1, 1);
    for (uint32_t s0 = 1; s0 <= gNS; ++s0) {
        uint32_t s = s0;
        for (int k = 0; k < 8 * P; ++k) {
            for (int j = 0; j < stride; ++j) s = step(s);
            gS[(size_t) s0 * 8 * P + k] = (int16_t) ((int32_t) (s >> (K - 16)) - 32768);
        }
    }
    return taps;
}

static const float* gA;
static int entry(uint32_t s, float* e) {
    const int P = gP;
    const int16_t* st = gS + (size_t) s * 8 * P;
    float u0[32], *U = e;
    for (int k = 0; k < 8 * P; ++k) u0[k] = (float) st[k] * R32;
    for (int c = 0; c < 8; ++c)
        for (int p = 0; p < P; ++p) {
            float acc = 0.0f;
            for (int k = 0; k <= c; ++k) acc = acc + gA[c * 8 + k] * u0[P * k + p];
            U[P * c + p] = acc;
        }
    float G[4][4];
    for (int i = 0; i < P; ++i)
        for (int j = 0; j < P; ++j) { float g = 0; for (int c = 0; c < 8; ++c) g += U[P * c + i] * U[P * c + j]; G[i][j] = g; }
    float* L = U + 8 * P, *D = L + P * (P + 1) / 2;
    for (int i = 0; i < P; ++i)
        for (int j = 0; j <= i; ++j) {
            float a = G[i][j];
            for (int k = 0; k < j; ++k) a -= L[i * (i + 1) / 2 + k] * L[j * (j + 1) / 2 + k];
            if (i == j) { if (!(a > 0.0f)) return 0; L[i * (i + 1) / 2 + i] = sqrtf(a); D[i] = 1.0f / sqrtf(a); }
            else L[i * (i + 1) / 2 + j] = a * D[j];
        }
    return 1;
}

static void* build(void* arg) {
    const int id = (int) (intptr_t) arg;
    for (uint32_t s = 1 + id; s <= gNS; s += NT) gOK[s] = (uint8_t) entry(s, gTAB + (size_t) s * gENT);
    return NULL;
}

// ---- vector helpers ----
static inline v8f vabs(v8f x) { return (v8f) ((v8i) x & 0x7FFFFFFF); }
static inline v8f vsel(v8i m, v8f a, v8f b) { return (v8f) (((v8i) a & m) | ((v8i) b & ~m)); }
static inline v8i vseli(v8i m, v8i a, v8i b) { return (a & m) | (b & ~m); }
static inline v8f vmax(v8f a, v8f b) { return vsel(a > b, a, b); }
static inline v8f vclamp(v8f r) { return vsel(r < -8.0f, (v8f) -8.0f, vsel(r > 7.0f, (v8f) 7.0f, r)); }
static inline v8f vpow2(v8i e) { return (v8f) ((e + 127) << 23); }

static const float* gX;   // [nb][8] transformed blocks x = A w
static int gNB;
static float* gBest;
static uint32_t* gBS;
static int* gBE;

static inline __attribute__((always_inline)) void search_vec(const int P, int v, uint32_t s0, uint32_t s1) {
    v8f x[8], wn = 0;
    for (int c = 0; c < 8; ++c) {
        for (int l = 0; l < 8; ++l) { const int k = 8 * v + l; x[c][l] = k < gNB ? gX[8 * k + c] : 0.0f; }
        wn = wn + x[c] * x[c];
    }
    v8f best;
    v8i bs, be;
    for (int l = 0; l < 8; ++l) {
        const int k = 8 * v + l;
        best[l] = k < gNB ? gBest[k] : 0.0f;
        bs[l] = k < gNB ? (int) gBS[k] : 0;
        be[l] = k < gNB ? gBE[k] : 0;
    }
    for (uint32_t s = s0; s <= s1; ++s) {
        if (!gOK[s]) continue;
        const float* U = gTAB + (size_t) s * gENT, *L = U + 8 * P, *D = L + P * (P + 1) / 2;
        v8f b[4] = {0, 0, 0, 0}, y[4], t[4];
        for (int c = 0; c < 8; ++c)
            for (int p = 0; p < P; ++p) b[p] = b[p] + U[P * c + p] * x[c];
        for (int i = 0; i < P; ++i) { v8f a = b[i]; for (int k = 0; k < i; ++k) a = a - L[i * (i + 1) / 2 + k] * y[k]; y[i] = a * D[i]; }
        for (int i = P - 1; i >= 0; --i) { v8f a = y[i]; for (int k = i + 1; k < P; ++k) a = a - L[k * (k + 1) / 2 + i] * t[k]; t[i] = a * D[i]; }
        v8f m = vabs(t[0]);
        for (int p = 1; p < P; ++p) m = vmax(m, vabs(t[p]));
        const v8i e0 = (((v8i) m >> 23) & 255) - 127 - 2;
        for (int ci = 0; ci < 3; ++ci) {
            const v8i e = e0 + (ci == 0 ? 0 : (ci == 1 ? -1 : 1));
            const v8f inv = vpow2(-e), sc = vpow2(e);
            v8f q[4], qb = 0, qgq = 0;
            for (int p = 0; p < P; ++p) { q[p] = vclamp((t[p] * inv + MAGIC) - MAGIC); qb = qb + q[p] * b[p]; }
            for (int j = 0; j < P; ++j) { v8f z = 0; for (int i = j; i < P; ++i) z = z + L[i * (i + 1) / 2 + j] * q[i]; qgq = qgq + z * z; }
            const v8f rec = sc * sc * qgq;
            const v8f er = vsel(rec > 4.0f * wn, (v8f) INFINITY, (wn - 2.0f * sc * qb) + rec);
            v8i better = er < best;
            if (!__builtin_reduce_or(better)) continue;
            // The expanded error loses precision when U is nearly singular along q and sc is large: confirm an
            // improvement with the direct residual |x - sc U q|^2 before it replaces the best.
            v8f er2 = 0;
            for (int c = 0; c < 8; ++c) {
                v8f r = x[c];
                for (int p = 0; p < P; ++p) r = r - (sc * q[p]) * U[P * c + p];
                er2 = er2 + r * r;
            }
            better = er2 < best;
            best = vsel(better, er2, best);
            bs = vseli(better, (v8i) (int) s, bs);
            be = vseli(better, e, be);
        }
    }
    for (int l = 0; l < 8; ++l) {
        const int k = 8 * v + l;
        if (k < gNB) { gBest[k] = best[l]; gBS[k] = (uint32_t) bs[l]; gBE[k] = be[l]; }
    }
}

static void* run(void* arg) {
    const int id = (int) (intptr_t) arg, nv = (gNB + 7) / 8;
    for (uint32_t s0 = 1; s0 <= gNS; s0 += CHUNK) {
        const uint32_t s1 = s0 + CHUNK - 1 < gNS ? s0 + CHUNK - 1 : gNS;
        for (int v = id; v < nv; v += NT) {
            if (gP == 3) search_vec(3, v, s0, s1);
            else search_vec(4, v, s0, s1);
        }
    }
    return NULL;
}

static inline float bfr(float f) { union { uint32_t u; float f; } v; v.f = f; v.u += 0x7FFFu + ((v.u >> 16) & 1u); v.u &= 0xFFFF0000u; return v.f; }
static inline float pow2f(int e) { union { uint32_t u; float f; } v; v.u = (uint32_t) (e + 127) << 23; return v.f; }

static void decode(uint32_t s, int e, const int* q, float* out) {
    const int P = gP;
    const int16_t* st = gS + (size_t) s * 8 * P;
    const float sc = R32 * pow2f(e);
    for (int c = 0; c < 8; ++c) {
        int32_t is = 0;
        for (int p = 0; p < P; ++p) is += (int32_t) st[P * c + p] * q[p];
        out[c] = bfr((float) is * sc);
    }
}
static float aerr(const float* A, const float* x, const float* wd) {   // |x - A wd|^2
    float er = 0;
    for (int c = 0; c < 8; ++c) {
        float v = 0;
        for (int k = 0; k <= c; ++k) v += A[c * 8 + k] * wd[k];
        er += (x[c] - v) * (x[c] - v);
    }
    return er;
}

// One column group: w [nb][8] (unscaled), A [64] lower triangular (row-major).  Writes the decoded weights wd [nb][8],
// seeds, exponents, and |A (w - wd)|^2 per block.
void seed_search(const float* A, const float* w, int nb, float* wd, uint32_t* seed, int* ex, float* err) {
    gA = A;
    pthread_t th[NT];
    for (int i = 0; i < NT; ++i) pthread_create(&th[i], NULL, build, (void*) (intptr_t) i);
    for (int i = 0; i < NT; ++i) pthread_join(th[i], NULL);
    float* X = malloc(sizeof(float) * 8 * nb);
    for (int k = 0; k < nb; ++k)
        for (int c = 0; c < 8; ++c) { float acc = 0; for (int j = 0; j <= c; ++j) acc += A[c * 8 + j] * w[8 * k + j]; X[8 * k + c] = acc; }
    gX = X; gNB = nb;
    gBest = malloc(4 * nb); gBS = malloc(4 * nb); gBE = malloc(4 * nb);
    for (int k = 0; k < nb; ++k) { gBest[k] = INFINITY; gBS[k] = 0; gBE[k] = 0; }
    for (int i = 0; i < NT; ++i) pthread_create(&th[i], NULL, run, (void*) (intptr_t) i);
    for (int i = 0; i < NT; ++i) pthread_join(th[i], NULL);
    const int P = gP;
    int nn = 1;
    for (int p = 0; p < P; ++p) nn *= 3;
    for (int k = 0; k < nb; ++k) {
        float* o = wd + 8 * k;
        if (!gBS[k]) { memset(o, 0, 32); seed[k] = 0; ex[k] = 0; err[k] = aerr(A, X + 8 * k, o); continue; }
        const float* U = gTAB + (size_t) gBS[k] * gENT, *L = U + 8 * P, *D = L + P * (P + 1) / 2;
        float b[4] = {0, 0, 0, 0}, y[4], t[4];
        for (int c = 0; c < 8; ++c) for (int p = 0; p < P; ++p) b[p] += U[P * c + p] * X[8 * k + c];
        for (int i = 0; i < P; ++i) { float a = b[i]; for (int j = 0; j < i; ++j) a -= L[i * (i + 1) / 2 + j] * y[j]; y[i] = a * D[i]; }
        for (int i = P - 1; i >= 0; --i) { float a = y[i]; for (int j = i + 1; j < P; ++j) a -= L[j * (j + 1) / 2 + i] * t[j]; t[i] = a * D[i]; }
        const float inv = pow2f(-gBE[k]);
        int q[4], bq[4];
        for (int p = 0; p < P; ++p) {
            float r = (t[p] * inv + MAGIC) - MAGIC;
            r = r < -8.0f ? -8.0f : (r > 7.0f ? 7.0f : r);
            bq[p] = q[p] = (int) r;
        }
        float tmp[8];
        decode(gBS[k], gBE[k], q, o);
        float be = aerr(A, X + 8 * k, o);
        for (int d = 0; d < nn; ++d) {
            int cq[4], okq = 1, dd = d;
            for (int p = 0; p < P; ++p) { cq[p] = q[p] + dd % 3 - 1; dd /= 3; okq &= cq[p] >= -8 && cq[p] <= 7; }
            if (!okq) continue;
            decode(gBS[k], gBE[k], cq, tmp);
            const float ee = aerr(A, X + 8 * k, tmp);
            if (ee < be) { be = ee; memcpy(bq, cq, sizeof bq); memcpy(o, tmp, 32); }
        }
        seed[k] = gBS[k]; ex[k] = gBE[k]; err[k] = be;
    }
    free(X); free(gBest); free(gBS); free(gBE);
}
