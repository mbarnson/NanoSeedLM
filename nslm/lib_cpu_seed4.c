// nslm/lib_cpu_seed4.c - see cpu_seed4.h.
#include "cpu_seed4.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "lfsr.h"

#define VW 16   // rows per vector step
typedef float f16v __attribute__((ext_vector_type(VW)));
typedef uint32_t u16v __attribute__((ext_vector_type(VW)));
typedef int32_t i16v __attribute__((ext_vector_type(VW)));

static uint32_t* g_table;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void build_table(void) {
    g_table = malloc(65536 * sizeof(uint32_t));
    g_table[0] = 0;
    for (int s = 1; s < 65536; ++s) g_table[s] = lfsr_stream32((uint16_t) s);
}

int cpu_seed4_on(void) { return 1; }

uint32_t* cpu_seed4_table(void) {
    pthread_once(&g_once, build_table);
    return g_table;
}

static inline f16v fma16(f16v a, f16v b, f16v c) { return __builtin_elementwise_fma(a, b, c); }

// v = 1 + state_k / 2^16 as f32 bits: 0x3F800000 | (state_k << 7), state_k = bits k .. k + 15 of the window.
#define STATE(lo, t, k) ((k) <= 16 ? (lo) >> (k) : (t) >> ((k) - 16))
#define VBITS(src) (((src) << 7 & 0x007FFF80u) | 0x3F800000u)

// One 16-row step over all kb blocks: rows r .. r + VW - 1 (all valid).
static void rows16(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias, const uint32_t* table,
                   const float* x, const float* xs, int kb, int r, float* y) {
    f16v acc = 0.0f;
    for (int b = 0; b < kb; ++b) {
        u16v s, t, q, code;
        for (int i = 0; i < VW; ++i) {
            const int g = (r + i) * kb + b;
            s[i] = seeds[g];
            t[i] = table[seeds[g]];
            q[i] = coefs[g];
            code[i] = (codes[g >> 1] >> ((g & 1) * 4)) & 15u;
        }
        const u16v lo = s | t << 16;
        const float* xb = x + 8 * b;
        f16v a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#define STEP(c)                                                                                  \
        {                                                                                        \
            const f16v xc = xb[c];                                                               \
            a0 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 1)), a0);                               \
            a1 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 2)), a1);                               \
            a2 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 3)), a2);                               \
            a3 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 4)), a3);                               \
        }
        STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
#undef STEP
        const i16v qi = (i16v) q;
        const f16v q0 = __builtin_convertvector(qi << 28 >> 28, f16v), q1 = __builtin_convertvector(qi << 24 >> 28, f16v);
        const f16v q2 = __builtin_convertvector(qi << 20 >> 28, f16v), q3 = __builtin_convertvector(qi << 16 >> 28, f16v);
        const f16v qs = 1.5f * (((q0 + q1) + q2) + q3);
        // sc = R32 * 2^(e + 16) from exponent bits
        const f16v sc = (f16v) ((u16v) (bias + 16 + 127) + code << 23) * NSLM_R32;
        const f16v d = (((a0 * q0 + a1 * q1) + a2 * q2) + a3 * q3) - qs * xs[b];
        acc = fma16(d, sc, acc);
    }
    memcpy(y + r, &acc, sizeof acc);
}

static void rows1(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias, const uint32_t* table,
                  const float* x, const float* xs, int kb, int r, float* y) {
    float acc = 0.0f;
    for (int b = 0; b < kb; ++b) {
        const int g = r * kb + b;
        const uint32_t s = seeds[g], t = table[s], lo = s | t << 16, q = coefs[g];
        const int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int c = 0; c < 8; ++c)
            for (int p = 0; p < 4; ++p) {
                const int k = 4 * c + p + 1;
                const uint32_t bits = VBITS(STATE(lo, t, k));
                float v;
                memcpy(&v, &bits, 4);
                a[p] = __builtin_fmaf(x[8 * b + c], v, a[p]);
            }
        const float q0 = (float) ((int32_t) (q << 28) >> 28), q1 = (float) ((int32_t) (q << 24) >> 28);
        const float q2 = (float) ((int32_t) (q << 20) >> 28), q3 = (float) ((int32_t) (q << 16) >> 28);
        const float d = (((a[0] * q0 + a[1] * q1) + a[2] * q2) + a[3] * q3) - 1.5f * (((q0 + q1) + q2) + q3) * xs[b];
        const uint32_t eb = (uint32_t) (bias + 16 + 127 + code) << 23;
        float p2;
        memcpy(&p2, &eb, 4);
        acc = __builtin_fmaf(d, p2 * NSLM_R32, acc);
    }
    y[r] = acc;
}

static void block_sums(const float* x, int kb, float* xs) {
    for (int b = 0; b < kb; ++b) {
        float s = 0.0f;
        for (int c = 0; c < 8; ++c) s += x[8 * b + c];
        xs[b] = s;
    }
}

static void matvec_rows(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                        const uint32_t* table, const float* x, const float* xs, int kb, int r0, int r1, float* y) {
    int r = r0;
    for (; r + VW <= r1; r += VW) rows16(seeds, coefs, codes, bias, table, x, xs, kb, r, y);
    for (; r < r1; ++r) rows1(seeds, coefs, codes, bias, table, x, xs, kb, r, y);
}

void cpu_seed4_matvec_rows(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                           const uint32_t* table, const float* x, int kb, int r0, int r1, float* y) {
    float* xs = malloc(kb * sizeof(float));
    block_sums(x, kb, xs);
    matvec_rows(seeds, coefs, codes, bias, table, x, xs, kb, r0, r1, y);
    free(xs);
}

typedef struct {
    const uint16_t *seeds, *coefs;
    const uint8_t* codes;
    int bias, kb, rows;
    const uint32_t* table;
    const float *x, *xs;
    float* y;
    atomic_int next;
} Job;

#define CHUNK 64

static void* worker(void* arg) {
    Job* j = arg;
    for (;;) {
        const int r0 = atomic_fetch_add(&j->next, CHUNK);
        if (r0 >= j->rows) break;
        const int r1 = r0 + CHUNK < j->rows ? r0 + CHUNK : j->rows;
        matvec_rows(j->seeds, j->coefs, j->codes, j->bias, j->table, j->x, j->xs, j->kb, r0, r1, j->y);
    }
    return NULL;
}

void cpu_seed4_matvec_mt(const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                         const uint32_t* table, const float* x, int rows, int kb, float* y, int threads) {
    float* xs = malloc(kb * sizeof(float));
    block_sums(x, kb, xs);
    Job j = {seeds, coefs, codes, bias, kb, rows, table, x, xs, y, 0};
    if (threads <= 1) {
        worker(&j);
    } else {
        pthread_t* th = malloc((threads - 1) * sizeof(pthread_t));
        for (int i = 0; i < threads - 1; ++i) pthread_create(&th[i], NULL, worker, &j);
        worker(&j);
        for (int i = 0; i < threads - 1; ++i) pthread_join(th[i], NULL);
        free(th);
    }
    free(xs);
}

// ---- the CPU-tiled layout ----
typedef uint16_t h16v __attribute__((ext_vector_type(CPU_SEED4_T)));
typedef uint8_t b16v __attribute__((ext_vector_type(CPU_SEED4_T)));
_Static_assert(CPU_SEED4_T == VW, "tile = vector width");

void cpu_seed4_tile(CpuSeed4Tiled* t, const uint16_t* seeds, const uint16_t* coefs, const uint8_t* codes, int bias,
                    int rows, int kb) {
    const int T = CPU_SEED4_T, tiles = (rows + T - 1) / T;
    t->seeds = malloc((size_t) tiles * kb * T * 2);
    t->coefs = calloc((size_t) tiles * kb * T, 2);
    t->codes = calloc((size_t) tiles * (kb / 2) * T, 1);
    t->bias = bias, t->rows = rows, t->kb = kb, t->tiles = tiles, t->p = 4;
    for (int tile = 0; tile < tiles; ++tile)
        for (int b = 0; b < kb; ++b)
            for (int i = 0; i < T; ++i) {
                const int r = tile * T + i;
                const size_t o = ((size_t) tile * kb + b) * T + i;
                if (r >= rows) { t->seeds[o] = 1; continue; }   // a valid seed, zero coefficients: contributes 0
                const size_t g = (size_t) r * kb + b;
                t->seeds[o] = seeds[g];
                t->coefs[o] = coefs[g];
                const int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
                t->codes[((size_t) tile * (kb / 2) + b / 2) * T + i] |= (uint8_t) (code << ((b & 1) * 4));
            }
}

void cpu_seed4_tiled_free(CpuSeed4Tiled* t) {
    free(t->seeds), free(t->coefs), free(t->codes);
    memset(t, 0, sizeof *t);
}

void cpu_seed3_tile(CpuSeed4Tiled* t, const uint16_t* seeds, const uint16_t* nibs, int bias, int rows, int kb) {
    const int T = CPU_SEED4_T, tiles = (rows + T - 1) / T;
    t->seeds = malloc((size_t) tiles * kb * T * 2);
    t->coefs = calloc((size_t) tiles * kb * T, 2);
    t->codes = NULL;
    t->bias = bias, t->rows = rows, t->kb = kb, t->tiles = tiles, t->p = 3;
    for (int tile = 0; tile < tiles; ++tile)
        for (int b = 0; b < kb; ++b)
            for (int i = 0; i < T; ++i) {
                const int r = tile * T + i;
                const size_t o = ((size_t) tile * kb + b) * T + i;
                t->seeds[o] = r < rows ? seeds[(size_t) r * kb + b] : 1;   // padding: a valid seed, zero coefficients
                if (r < rows) t->coefs[o] = nibs[(size_t) r * kb + b];
            }
}

// One tile: the arithmetic of rows16, with vector loads.
static void tile_rows(const CpuSeed4Tiled* m, int tile, const uint32_t* table, const float* x, const float* xs,
                      float* out) {
    const int kb = m->kb, T = CPU_SEED4_T;
    const uint16_t* sp = m->seeds + (size_t) tile * kb * T;
    const uint16_t* qp = m->coefs + (size_t) tile * kb * T;
    const uint8_t* cp = m->codes + (size_t) tile * (kb / 2) * T;
    f16v acc = 0.0f;
    for (int b = 0; b < kb; ++b) {
        h16v s16, q16;
        b16v c16;
        memcpy(&s16, sp + (size_t) b * T, sizeof s16);
        memcpy(&q16, qp + (size_t) b * T, sizeof q16);
        memcpy(&c16, cp + (size_t) (b / 2) * T, sizeof c16);
        const u16v s = __builtin_convertvector(s16, u16v), q = __builtin_convertvector(q16, u16v);
        const u16v code = __builtin_convertvector(c16, u16v) >> ((b & 1) * 4) & 15u;
        u16v t;
        for (int i = 0; i < VW; ++i) t[i] = table[sp[(size_t) b * T + i]];
        const u16v lo = s | t << 16;
        const float* xb = x + 8 * b;
        f16v a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
#define STEP(c)                                                                                  \
        {                                                                                        \
            const f16v xc = xb[c];                                                               \
            a0 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 1)), a0);                         \
            a1 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 2)), a1);                         \
            a2 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 3)), a2);                         \
            a3 = fma16(xc, (f16v) VBITS(STATE(lo, t, 4 * (c) + 4)), a3);                         \
        }
        STEP(0) STEP(1) STEP(2) STEP(3) STEP(4) STEP(5) STEP(6) STEP(7)
#undef STEP
        const i16v qi = (i16v) q;
        const f16v q0 = __builtin_convertvector(qi << 28 >> 28, f16v), q1 = __builtin_convertvector(qi << 24 >> 28, f16v);
        const f16v q2 = __builtin_convertvector(qi << 20 >> 28, f16v), q3 = __builtin_convertvector(qi << 16 >> 28, f16v);
        const f16v qs = 1.5f * (((q0 + q1) + q2) + q3);
        const f16v sc = (f16v) ((u16v) (m->bias + 16 + 127) + code << 23) * NSLM_R32;
        const f16v d = (((a0 * q0 + a1 * q1) + a2 * q2) + a3 * q3) - qs * xs[b];
        acc = fma16(d, sc, acc);
    }
    memcpy(out, &acc, sizeof acc);
}

typedef struct {
    const CpuSeed4Tiled* m;
    const uint32_t* table;
    const float *x, *xs;
    float* y;
    atomic_int next;
} TJob;

#define TILE_CHUNK 4

static void* tworker(void* arg) {
    TJob* j = arg;
    const int T = CPU_SEED4_T;
    float out[CPU_SEED4_T];
    for (;;) {
        const int t0 = atomic_fetch_add(&j->next, TILE_CHUNK);
        if (t0 >= j->m->tiles) break;
        const int t1 = t0 + TILE_CHUNK < j->m->tiles ? t0 + TILE_CHUNK : j->m->tiles;
        for (int tile = t0; tile < t1; ++tile) {
            tile_rows(j->m, tile, j->table, j->x, j->xs, out);
            const int r0 = tile * T, n = j->m->rows - r0 < T ? j->m->rows - r0 : T;
            memcpy(j->y + r0, out, n * sizeof(float));
        }
    }
    return NULL;
}

static void run_n(const CpuSeed4Tiled* m, const uint32_t* table, const float* x, int ntok, float* y, int threads);
void cpu_seed4_tiled_matvec_mt(const CpuSeed4Tiled* m, const uint32_t* table, const float* x, float* y, int threads) {
    if (m->p == 3) { run_n(m, table, x, 1, y, threads); return; }
    float* xs = malloc(m->kb * sizeof(float));
    block_sums(x, m->kb, xs);
    TJob j = {m, table, x, xs, y, 0};
    if (threads <= 1) {
        tworker(&j);
    } else {
        pthread_t* th = malloc((threads - 1) * sizeof(pthread_t));
        for (int i = 0; i < threads - 1; ++i) pthread_create(&th[i], NULL, tworker, &j);
        tworker(&j);
        for (int i = 0; i < threads - 1; ++i) pthread_join(th[i], NULL);
        free(th);
    }
    free(xs);
}

// tile_rows for ntok tokens: the states of each block once (v[4c + p] = state 4c + p + 1), then each token's FMAs,
// combine and accumulate in tile_rows' order.
static void tile_rows_n(const CpuSeed4Tiled* m, int tile, const uint32_t* table, const float* x, const float* xs, int ntok,
                        float* out) {
    const int kb = m->kb, T = CPU_SEED4_T;
    const uint16_t* sp = m->seeds + (size_t) tile * kb * T;
    const uint16_t* qp = m->coefs + (size_t) tile * kb * T;
    const uint8_t* cp = m->codes + (size_t) tile * (kb / 2) * T;
    f16v acc[CPU_SEED4_MAX_TOK];
    for (int k = 0; k < ntok; ++k) acc[k] = 0.0f;
    for (int b = 0; b < kb; ++b) {
        h16v s16, q16;
        b16v c16;
        memcpy(&s16, sp + (size_t) b * T, sizeof s16);
        memcpy(&q16, qp + (size_t) b * T, sizeof q16);
        memcpy(&c16, cp + (size_t) (b / 2) * T, sizeof c16);
        const u16v s = __builtin_convertvector(s16, u16v), q = __builtin_convertvector(q16, u16v);
        const u16v code = __builtin_convertvector(c16, u16v) >> ((b & 1) * 4) & 15u;
        u16v t;
        for (int i = 0; i < VW; ++i) t[i] = table[sp[(size_t) b * T + i]];
        const u16v lo = s | t << 16;
        f16v v[32];
#define GEN(c)                                                                                   \
        v[4 * (c)] = (f16v) VBITS(STATE(lo, t, 4 * (c) + 1));                                    \
        v[4 * (c) + 1] = (f16v) VBITS(STATE(lo, t, 4 * (c) + 2));                                \
        v[4 * (c) + 2] = (f16v) VBITS(STATE(lo, t, 4 * (c) + 3));                                \
        v[4 * (c) + 3] = (f16v) VBITS(STATE(lo, t, 4 * (c) + 4));
        GEN(0) GEN(1) GEN(2) GEN(3) GEN(4) GEN(5) GEN(6) GEN(7)
#undef GEN
        const i16v qi = (i16v) q;
        const f16v q0 = __builtin_convertvector(qi << 28 >> 28, f16v), q1 = __builtin_convertvector(qi << 24 >> 28, f16v);
        const f16v q2 = __builtin_convertvector(qi << 20 >> 28, f16v), q3 = __builtin_convertvector(qi << 16 >> 28, f16v);
        const f16v qs = 1.5f * (((q0 + q1) + q2) + q3);
        const f16v sc = (f16v) ((u16v) (m->bias + 16 + 127) + code << 23) * NSLM_R32;
        for (int k = 0; k < ntok; ++k) {
            const float* xb = x + (size_t) k * kb * 8 + 8 * b;
            f16v a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (int c = 0; c < 8; ++c) {
                const f16v xc = xb[c];
                a0 = fma16(xc, v[4 * c], a0);
                a1 = fma16(xc, v[4 * c + 1], a1);
                a2 = fma16(xc, v[4 * c + 2], a2);
                a3 = fma16(xc, v[4 * c + 3], a3);
            }
            const f16v d = (((a0 * q0 + a1 * q1) + a2 * q2) + a3 * q3) - qs * xs[(size_t) k * kb + b];
            acc[k] = fma16(d, sc, acc[k]);
        }
    }
    for (int k = 0; k < ntok; ++k) memcpy(out + (size_t) k * T, &acc[k], sizeof acc[k]);
}

// P = 3: states 1..24 (v[3c + p] = state 3c + p + 1), the nibble word's code and 3 coefficients; else as tile_rows_n.
static void tile_rows3_n(const CpuSeed4Tiled* m, int tile, const uint32_t* table, const float* x, const float* xs, int ntok,
                         float* out) {
    const int kb = m->kb, T = CPU_SEED4_T;
    const uint16_t* sp = m->seeds + (size_t) tile * kb * T;
    const uint16_t* qp = m->coefs + (size_t) tile * kb * T;
    f16v acc[CPU_SEED4_MAX_TOK];
    for (int k = 0; k < ntok; ++k) acc[k] = 0.0f;
    for (int b = 0; b < kb; ++b) {
        h16v s16, q16;
        memcpy(&s16, sp + (size_t) b * T, sizeof s16);
        memcpy(&q16, qp + (size_t) b * T, sizeof q16);
        const u16v s = __builtin_convertvector(s16, u16v), q = __builtin_convertvector(q16, u16v);
        u16v t;
        for (int i = 0; i < VW; ++i) t[i] = table[sp[(size_t) b * T + i]];
        const u16v lo = s | t << 16;
        f16v v[24];
#define GEN(c)                                                                                   \
        v[3 * (c)] = (f16v) VBITS(STATE(lo, t, 3 * (c) + 1));                                    \
        v[3 * (c) + 1] = (f16v) VBITS(STATE(lo, t, 3 * (c) + 2));                                \
        v[3 * (c) + 2] = (f16v) VBITS(STATE(lo, t, 3 * (c) + 3));
        GEN(0) GEN(1) GEN(2) GEN(3) GEN(4) GEN(5) GEN(6) GEN(7)
#undef GEN
        const i16v qi = (i16v) q;
        const f16v q0 = __builtin_convertvector(qi << 24 >> 28, f16v), q1 = __builtin_convertvector(qi << 20 >> 28, f16v);
        const f16v q2 = __builtin_convertvector(qi << 16 >> 28, f16v);
        const f16v qs = 1.5f * ((q0 + q1) + q2);
        const f16v sc = (f16v) ((u16v) (m->bias + 16 + 127) + (q & 15u) << 23) * NSLM_R32;
        for (int k = 0; k < ntok; ++k) {
            const float* xb = x + (size_t) k * kb * 8 + 8 * b;
            f16v a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            for (int c = 0; c < 8; ++c) {
                const f16v xc = xb[c];
                a0 = fma16(xc, v[3 * c], a0);
                a1 = fma16(xc, v[3 * c + 1], a1);
                a2 = fma16(xc, v[3 * c + 2], a2);
            }
            const f16v d = ((a0 * q0 + a1 * q1) + a2 * q2) - qs * xs[(size_t) k * kb + b];
            acc[k] = fma16(d, sc, acc[k]);
        }
    }
    for (int k = 0; k < ntok; ++k) memcpy(out + (size_t) k * T, &acc[k], sizeof acc[k]);
}

typedef struct {
    const CpuSeed4Tiled* m;
    const uint32_t* table;
    const float *x, *xs;
    float* y;
    int ntok;
    atomic_int next;
} NJob;

static void* nworker(void* arg) {
    NJob* j = arg;
    const int T = CPU_SEED4_T;
    float out[CPU_SEED4_MAX_TOK * CPU_SEED4_T];
    for (;;) {
        const int t0 = atomic_fetch_add(&j->next, TILE_CHUNK);
        if (t0 >= j->m->tiles) break;
        const int t1 = t0 + TILE_CHUNK < j->m->tiles ? t0 + TILE_CHUNK : j->m->tiles;
        for (int tile = t0; tile < t1; ++tile) {
            (j->m->p == 3 ? tile_rows3_n : tile_rows_n)(j->m, tile, j->table, j->x, j->xs, j->ntok, out);
            const int r0 = tile * T, n = j->m->rows - r0 < T ? j->m->rows - r0 : T;
            for (int k = 0; k < j->ntok; ++k) memcpy(j->y + (size_t) k * j->m->rows + r0, out + (size_t) k * T, n * sizeof(float));
        }
    }
    return NULL;
}

static void run_n(const CpuSeed4Tiled* m, const uint32_t* table, const float* x, int ntok, float* y, int threads);

void cpu_seed4_tiled_matmul_mt(const CpuSeed4Tiled* m, const uint32_t* table, const float* x, int ntok, float* y, int threads) {
    if (ntok == 1 && m->p == 4) cpu_seed4_tiled_matvec_mt(m, table, x, y, threads);   // 12% faster: no state array
    else run_n(m, table, x, ntok, y, threads);
}

static void run_n(const CpuSeed4Tiled* m, const uint32_t* table, const float* x, int ntok, float* y, int threads) {
    float* xs = malloc((size_t) ntok * m->kb * sizeof(float));
    for (int k = 0; k < ntok; ++k) block_sums(x + (size_t) k * m->kb * 8, m->kb, xs + (size_t) k * m->kb);
    NJob j = {m, table, x, xs, y, ntok, 0};
    if (threads <= 1) {
        nworker(&j);
    } else {
        pthread_t* th = malloc((threads - 1) * sizeof(pthread_t));
        for (int i = 0; i < threads - 1; ++i) pthread_create(&th[i], NULL, nworker, &j);
        nworker(&j);
        for (int i = 0; i < threads - 1; ++i) pthread_join(th[i], NULL);
        free(th);
    }
    free(xs);
}

void cpu_seed_xt(const float* x, int ntok, int kb, float* xt, float* xs) {
    for (int k = 0; k < ntok; ++k)
        for (int b = 0; b < kb; ++b) {
            float sum = 0;
            for (int c = 0; c < 8; ++c) sum += xt[((size_t) k * 8 + c) * kb + b] = x[(size_t) k * kb * 8 + 8 * b + c];
            xs[(size_t) k * kb + b] = sum;
        }
}

// One vector step of cpu_seed3_rows: blocks [b, b + NL) of the row for all tokens. NL < VW (a last step) loads zero lanes
// (q = 0, x = 0), which add nothing; NL == VW gives fixed-size loads.
#define ROWS3_STEP(NL)                                                                                      \
    {                                                                                                       \
        h16v s16 = 0, q16 = 0;                                                                              \
        memcpy(&s16, sp + b, sizeof(uint16_t) * (size_t) (NL));                                             \
        memcpy(&q16, qp + b, sizeof(uint16_t) * (size_t) (NL));                                             \
        const u16v s = __builtin_convertvector(s16, u16v), q = __builtin_convertvector(q16, u16v);          \
        u16v t;                                                                                             \
        for (int i = 0; i < VW; ++i) t[i] = table[s16[i]];                                                  \
        const u16v lo = s | t << 16;                                                                        \
        f16v v[24];                                                                                         \
        for (int c = 0; c < 24; ++c) v[c] = (f16v) VBITS(STATE(lo, t, c + 1));                              \
        const i16v qi = (i16v) q;                                                                           \
        const f16v q0 = __builtin_convertvector(qi << 24 >> 28, f16v), q1 = __builtin_convertvector(qi << 20 >> 28, f16v); \
        const f16v q2 = __builtin_convertvector(qi << 16 >> 28, f16v);                                      \
        const f16v qs = 1.5f * ((q0 + q1) + q2);                                                            \
        const f16v sc = (f16v) ((u16v) (bias + 16 + 127) + (q & 15u) << 23) * NSLM_R32;                     \
        for (int k = 0; k < ntok; ++k) {                                                                    \
            const float* xk = xt + (size_t) k * 8 * kb + b;                                                 \
            f16v a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, xb = 0.0f;                                                \
            for (int c = 0; c < 8; ++c) {                                                                   \
                f16v xc = 0.0f;                                                                             \
                memcpy(&xc, xk + (size_t) c * kb, sizeof(float) * (size_t) (NL));                           \
                a0 = fma16(xc, v[3 * c], a0);                                                               \
                a1 = fma16(xc, v[3 * c + 1], a1);                                                           \
                a2 = fma16(xc, v[3 * c + 2], a2);                                                           \
            }                                                                                               \
            memcpy(&xb, xs + (size_t) k * kb + b, sizeof(float) * (size_t) (NL));                           \
            const f16v d = ((a0 * q0 + a1 * q1) + a2 * q2) - qs * xb;                                       \
            acc[k] = fma16(d, sc, acc[k]);                                                                  \
        }                                                                                                   \
    }

void cpu_seed3_rows(const uint16_t* seeds, const uint16_t* nibs, int bias, const uint32_t* table, const float* xt,
                    const float* xs, int ntok, int kb, int r0, int r1, float* y, int ys) {
    for (int r = r0; r < r1; ++r) {
        const uint16_t* sp = seeds + (size_t) r * kb, *qp = nibs + (size_t) r * kb;
        f16v acc[CPU_SEED4_MAX_TOK];
        for (int k = 0; k < ntok; ++k) acc[k] = 0.0f;
        int b = 0;
        if (ntok == 1)   // one token: the states inside the dot products, no state array (as the tiled matvec)
            for (; b + VW <= kb; b += VW) {
                h16v s16, q16;
                memcpy(&s16, sp + b, sizeof s16);
                memcpy(&q16, qp + b, sizeof q16);
                const u16v s = __builtin_convertvector(s16, u16v), q = __builtin_convertvector(q16, u16v);
                u16v t;
                for (int i = 0; i < VW; ++i) t[i] = table[sp[b + i]];
                const u16v lo = s | t << 16;
                f16v a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, xb;
#define C1(c) { f16v xc; memcpy(&xc, xt + (size_t) (c) * kb + b, sizeof xc); \
                a0 = fma16(xc, (f16v) VBITS(STATE(lo, t, 3 * (c) + 1)), a0); a1 = fma16(xc, (f16v) VBITS(STATE(lo, t, 3 * (c) + 2)), a1); \
                a2 = fma16(xc, (f16v) VBITS(STATE(lo, t, 3 * (c) + 3)), a2); }
                C1(0) C1(1) C1(2) C1(3) C1(4) C1(5) C1(6) C1(7)
#undef C1
                const i16v qi = (i16v) q;
                const f16v q0 = __builtin_convertvector(qi << 24 >> 28, f16v), q1 = __builtin_convertvector(qi << 20 >> 28, f16v);
                const f16v q2 = __builtin_convertvector(qi << 16 >> 28, f16v);
                memcpy(&xb, xs + b, sizeof xb);
                const f16v d = ((a0 * q0 + a1 * q1) + a2 * q2) - 1.5f * ((q0 + q1) + q2) * xb;
                acc[0] = fma16(d, (f16v) ((u16v) (bias + 16 + 127) + (q & 15u) << 23) * NSLM_R32, acc[0]);
            }
        for (; b + VW <= kb; b += VW) ROWS3_STEP(VW)
        if (b < kb) ROWS3_STEP(kb - b)
        for (int k = 0; k < ntok; ++k) {
            float sum = 0;
            for (int i = 0; i < VW; ++i) sum += acc[k][i];
            y[(size_t) k * ys + r] = sum;
        }
    }
}
#undef ROWS3_STEP
