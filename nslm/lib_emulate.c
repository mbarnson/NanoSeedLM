// nslm/lib_emulate.c - see emulate.h.
#include "emulate.h"

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gguf.h"
#include "rows.h"

uint16_t ns_f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    const uint32_t s = (u >> 16) & 0x8000u, a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return (uint16_t) (s | 0x7E00u);    // NaN
    if (a >= 0x477FF000u) return (uint16_t) (s | 0x7C00u);   // 65520 and above round to infinity
    if (a >= 0x38800000u) {                                  // normal: 10 of 23 mantissa bits, rounded
        const uint32_t r = a - 0x38000000u;
        return (uint16_t) (s | ((r + 0xFFFu + ((r >> 13) & 1u)) >> 13));
    }
    const int sh = 126 - (int) (a >> 23);   // subnormal: the value in steps of 2^-24
    if (sh > 24) return (uint16_t) s;
    const uint32_t m = (a & 0x7FFFFFu) | 0x800000u, q = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    return (uint16_t) (s | (q + (rem > half || (rem == half && (q & 1)))));
}

void ns_q8_0_row(const float* x, int n, uint8_t* out) {
    for (int b = 0; b < n / 32; ++b, x += 32, out += 34) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = fabsf(x[j]) > amax ? fabsf(x[j]) : amax;
        const float d = amax / 127, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t h = ns_f32_to_f16(d);
        out[0] = (uint8_t) h;
        out[1] = (uint8_t) (h >> 8);
        for (int j = 0; j < 32; ++j) out[2 + j] = (uint8_t) (int8_t) roundf(x[j] * id);
    }
}

typedef struct { const NsTensor* t[NS_EMU_PACKS]; uint64_t off; int lp; } EmTensor;   // + Q8_0 offset, 3 L + p
typedef struct {
    const EmTensor* t;
    const uint8_t* map;
    const NsRows* nr;
    size_t len, cols;           // the largest tensor slice in bytes, the most columns
    int fd, items, next, bad;   // item k: expert k % slices of tensor k / slices (all tensors of a shard share slices)
    pthread_mutex_t mu;
} Emu;

static void* emu_worker(void* arg) {
    Emu* m = (Emu*) arg;
    uint8_t* buf = (uint8_t*) malloc(m->len);
    float* w = (float*) malloc(4 * m->cols);
    for (;;) {
        pthread_mutex_lock(&m->mu);
        const int k = m->bad ? m->items : m->next++;
        pthread_mutex_unlock(&m->mu);
        if (k >= m->items) break;
        const EmTensor* et = &m->t[k / m->t[0].t[0]->slices];
        const int e = k % et->t[0]->slices;
        const NsTensor* t = et->t[m->map ? m->map[(size_t) et->lp * (size_t) et->t[0]->slices + (size_t) e] : 0];
        const size_t rb = (size_t) t->cols / 32 * 34, len = rb * (size_t) t->rows;
        for (int r = 0; r < t->rows; ++r) {
            ns_row_f32(m->nr, t, e, r, 0, w);
            ns_q8_0_row(w, t->cols, buf + rb * (size_t) r);
        }
        size_t done = 0;
        const uint64_t at = et->off + (uint64_t) e * len;
        while (done < len) {
            const ssize_t n = pwrite(m->fd, buf + done, len - done, (off_t) (at + done));
            if (n <= 0) break;
            done += (size_t) n;
        }
        if (done < len) { pthread_mutex_lock(&m->mu); m->bad = 1; pthread_mutex_unlock(&m->mu); }
    }
    free(buf);
    free(w);
    return NULL;
}

int ns_emulate_gguf(const NsModel* const* packs, int np, const uint8_t* map, size_t map_len, const char* gguf, int workers,
                    int* n, char* err, int errlen) {
    static const char* kProj[3] = {"gate", "up", "down"};
    *n = 0;
    if (np < 1 || np > NS_EMU_PACKS) { snprintf(err, (size_t) errlen, "%d packs: 1 to %d", np, NS_EMU_PACKS); return -1; }
    Gguf g;
    if (gguf_open(&g, gguf, err, errlen)) return -1;
    EmTensor* et = (EmTensor*) malloc(sizeof *et * (size_t) (g.n ? g.n : 1));
    int nt = 0, rc = 0;
    size_t len = 0, cols = 0;
    for (int i = 0; i < g.n && !rc; ++i) {   // check every expert tensor first
        const GgufTensor* gt = &g.t[i];
        char proj[8], name[128];
        int l = 0, at = 0, p = 0;
        if (sscanf(gt->name, "blk.%d.ffn_%7[a-z]_exps.weight%n", &l, proj, &at) != 2 || gt->name[at]) continue;
        while (p < 3 && strcmp(kProj[p], proj)) ++p;
        if (p == 3) continue;
        snprintf(name, sizeof name, "language_model.model.layers.%d.mlp.switch_mlp.%s_proj.weight", l, proj);
        EmTensor x = {{NULL}, g.data + gt->offset, 3 * l + p};
        for (int k = 0; k < np && !rc; ++k) {
            const NsTensor* t = x.t[k] = ns_find(packs[k], name);
            if (!t) rc = snprintf(err, (size_t) errlen, "%s: pack %d has no %s", gguf, k, name);
            else if (gt->type != 8 || gt->ndim != 3 || gt->dims[0] != t->cols || gt->dims[1] != t->rows || gt->dims[2] != t->slices ||
                     t->cols % 32 || t->enc == NS_SEED6P8 || (nt && t->slices != et[0].t[0]->slices) ||
                     g.data + gt->offset + (uint64_t) t->slices * (uint64_t) t->rows * (uint64_t) (t->cols / 32) * 34 > g.size)
                rc = snprintf(err, (size_t) errlen, "%s: %s is not Q8_0 [%d][%d][%d] like %s of pack %d", gguf, gt->name, t->slices,
                              t->rows, t->cols, name, k);
        }
        if (rc) break;
        const NsTensor* t0 = x.t[0];
        const size_t E = (size_t) t0->slices, lp = (size_t) x.lp, rb = (size_t) t0->rows * (size_t) (t0->cols / 32) * 34;
        if (map && (lp + 1) * E > map_len) rc = snprintf(err, (size_t) errlen, "the map has no %s", gt->name);
        for (size_t e = 0; map && !rc && e < E; ++e)
            if (map[lp * E + e] >= np) rc = snprintf(err, (size_t) errlen, "the map takes %s from pack %d", gt->name, map[lp * E + e]);
        if (rc) break;
        et[nt++] = x;
        if (rb > len) len = rb;
        if ((size_t) t0->cols > cols) cols = (size_t) t0->cols;
    }
    gguf_close(&g);   // before the writes: the mapping is read-only
    if (rc || !nt) { free(et); return rc ? -1 : 0; }
    NsRows* nr = (NsRows*) malloc(sizeof *nr);
    ns_rows_init(nr);
    Emu m = {et, map, nr, len, cols, open(gguf, O_RDWR), nt * et[0].t[0]->slices, 0, 0, PTHREAD_MUTEX_INITIALIZER};
    if (m.fd < 0) rc = snprintf(err, (size_t) errlen, "%s: cannot open for writing", gguf);
    else {
        pthread_t th[64];
        const int nw = workers < 1 ? 1 : workers > 64 ? 64 : workers;
        for (int i = 0; i < nw; ++i) pthread_create(&th[i], NULL, emu_worker, &m);
        for (int i = 0; i < nw; ++i) pthread_join(th[i], NULL);
        if (close(m.fd) || m.bad) rc = snprintf(err, (size_t) errlen, "%s: a write failed", gguf);
    }
    free(nr);
    free(et);
    *n = rc ? 0 : nt;
    return rc ? -1 : 0;
}
