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

static float f16_to_f32(uint16_t h) {
    const int e = h >> 10 & 31, m = h & 0x3FF;
    const float v = e ? ldexpf(1.0f + (float) m / 1024.0f, e - 15) : ldexpf((float) m / 1024.0f, -14);
    return h & 0x8000 ? -v : v;
}

// The engine's name of the shard's dense tensor g, else 0
static int dense_name(const char* g, char* out, size_t len) {
    static const char* kTop[][2] = {
        {"output.weight", "language_model.lm_head.weight"}, {"token_embd.weight", "language_model.model.embed_tokens.weight"},
        {"output_hc_up.weight", "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight"},
        {"output_hc_down.weight", "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight"}};
    static const char* kBlk[][2] = {
        {"attn_qkv", "linear_attn.in_proj_qkv"}, {"attn_gate", "linear_attn.in_proj_z"}, {"ssm_out", "linear_attn.out_proj"},
        {"ssm_alpha", "linear_attn.in_proj_a"}, {"ssm_beta", "linear_attn.in_proj_b"}, {"attn_q", "self_attn.q_proj"},
        {"attn_k", "self_attn.k_proj"}, {"attn_v", "self_attn.v_proj"}, {"attn_output", "self_attn.o_proj"},
        {"ffn_gate_shexp", "mlp.shared_expert.gate_proj"}, {"ffn_up_shexp", "mlp.shared_expert.up_proj"},
        {"ffn_down_shexp", "mlp.shared_expert.down_proj"}, {"hc_attn_up", "attn_hyper_connection.input_mix_weight_up"},
        {"hc_attn_down", "attn_hyper_connection.input_mix_weight_down"}, {"hc_attn_inject", "attn_hyper_connection.block_inject_weight"},
        {"hc_ffn_up", "mlp_hyper_connection.input_mix_weight_up"}, {"hc_ffn_down", "mlp_hyper_connection.input_mix_weight_down"},
        {"hc_ffn_inject", "mlp_hyper_connection.block_inject_weight"}};
    for (size_t i = 0; i < sizeof kTop / sizeof kTop[0]; ++i)
        if (!strcmp(g, kTop[i][0])) return snprintf(out, len, "%s", kTop[i][1]) > 0;
    char k[32];
    int l = -1, at = 0;
    if (sscanf(g, "blk.%d.%31[a-z_].weight%n", &l, k, &at) != 2 || g[at]) return 0;
    for (size_t i = 0; i < sizeof kBlk / sizeof kBlk[0]; ++i)
        if (!strcmp(k, kBlk[i][0])) return snprintf(out, len, "language_model.model.layers.%d.%s.weight", l, kBlk[i][1]) > 0;
    return 0;
}

// A DeltaNet tensor's value-head axis: none, rows by head (a, b), rows by head block (z), the v rows of qkv, columns (out)
enum { GDN_NONE, GDN_HEADS, GDN_BLOCKS, GDN_QKV, GDN_COLS };
static int gdn_kind(const char* name) {
    static const char* kSuf[] = {"in_proj_a.weight", "in_proj_b.weight", "in_proj_z.weight", "in_proj_qkv.weight", "out_proj.weight"};
    static const int kKind[] = {GDN_HEADS, GDN_HEADS, GDN_BLOCKS, GDN_QKV, GDN_COLS};
    const char* at = strstr(name, ".linear_attn.");
    for (int i = 0; at && i < 5; ++i) if (!strcmp(at + 13, kSuf[i])) return kKind[i];
    return GDN_NONE;
}
static int vhead(const NsGdnHeads* q, int h) { return h % q->hk * (q->hv / q->hk) + h / q->hk; }   // llama.cpp's h: MLX's
static int gdn_ok(const NsGdnHeads* q, int kind, const NsTensor* t) {
    const int vd = q->hv * q->dv;
    return kind == GDN_HEADS ? t->rows == q->hv : kind == GDN_BLOCKS ? t->rows == vd : kind == GDN_QKV ? t->rows == 2 * q->hk * q->dk + vd :
           kind == GDN_COLS ? t->cols == vd : 1;
}
// Row r of the shard's tensor from the pack's t, in llama.cpp's head order (tmp: t->cols floats for the columns)
static void shard_row(const NsRows* nr, const NsTensor* t, const NsGdnHeads* q, int kind, int r, float* w, float* tmp) {
    const int v0 = kind == GDN_QKV ? 2 * q->hk * q->dk : 0, dv = q ? q->dv : 1;
    if (kind == GDN_COLS) {
        ns_row_f32(nr, t, 0, r, 0, tmp);
        for (int c = 0; c < t->cols; ++c) w[c] = tmp[vhead(q, c / dv) * dv + c % dv];
        return;
    }
    const int src = kind == GDN_HEADS ? vhead(q, r) : (kind == GDN_BLOCKS || kind == GDN_QKV) && r >= v0 ?
                    v0 + vhead(q, (r - v0) / dv) * dv + (r - v0) % dv : r;
    ns_row_f32(nr, t, 0, src, 0, w);
}

typedef struct { const NsTensor* t[NS_EMU_PACKS]; uint64_t off; int lp; } EmTensor;   // lp: 3 L + p, or a DeltaNet kind   // + Q8_0 offset, 3 L + p
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

int ns_emulate_dense(const NsModel* pack, const NsGdnHeads* gdn, const char* only, const char* gguf, int* n, char* err, int errlen) {
    *n = 0;
    Gguf g;
    if (gguf_open(&g, gguf, err, errlen)) return -1;
    EmTensor* et = (EmTensor*) malloc(sizeof *et * (size_t) (g.n ? g.n : 1));
    NsRows* nr = (NsRows*) malloc(sizeof *nr);
    ns_rows_init(nr);
    float* w = NULL, *tmp = NULL;
    uint8_t* buf = NULL;
    int nt = 0, rc = 0;
    if (gdn && (gdn->hk < 1 || gdn->hv % gdn->hk || gdn->dv < 1)) rc = snprintf(err, (size_t) errlen, "DeltaNet heads %d / %d", gdn->hk, gdn->hv);
    for (int i = 0; i < g.n && !rc; ++i) {   // check every dense tensor first
        const GgufTensor* gt = &g.t[i];
        char name[160];
        if (!dense_name(gt->name, name, sizeof name) || (only && !strstr(name, only))) continue;
        const NsTensor* t = ns_find(pack, name);
        const size_t rb = t ? (size_t) t->cols / 32 * 34 : 0;
        const int kind = gdn ? gdn_kind(name) : GDN_NONE;
        if (!t || gt->type != 8 || gt->ndim != 2 || gt->dims[0] != t->cols || gt->dims[1] != t->rows || !(gdn ? gdn_ok(gdn, kind, t) : 1) ||
            t->slices != 1 || t->cols % 32 || g.data + gt->offset + rb * (uint64_t) t->rows > g.size) {
            rc = snprintf(err, (size_t) errlen, "%s: %s is not Q8_0 like the pack's %s", gguf, gt->name, name);
            break;
        }
        w = (float*) realloc(w, sizeof(float) * (size_t) t->cols);
        tmp = (float*) realloc(tmp, sizeof(float) * (size_t) t->cols);
        double e2 = 0, w2 = 0;
        const int ns = t->rows < 16 ? t->rows : 16;
        for (int k = 0; k < ns; ++k) {   // the row order: 16 rows spread over the tensor, against the shard's own
            const int r = (int) ((int64_t) k * t->rows / ns);
            const uint8_t* q = (const uint8_t*) gguf_data(&g, gt) + rb * (size_t) r;
            shard_row(nr, t, gdn, kind, r, w, tmp);
            for (int c = 0; c < t->cols; ++c) {
                const uint8_t* b = q + (size_t) c / 32 * 34;
                const double v = (double) f16_to_f32((uint16_t) (b[0] | b[1] << 8)) * (int8_t) b[2 + c % 32];
                e2 += (w[c] - v) * (w[c] - v);
                w2 += v * v;
            }
        }
        if (e2 > 0.09 * w2) {
            rc = snprintf(err, (size_t) errlen, "%s: %s is %.2f from the pack's %s: another row order", gguf, gt->name, sqrt(e2 / (w2 > 0 ? w2 : 1)), name);
            break;
        }
        et[nt++] = (EmTensor) {{t}, g.data + gt->offset, kind};
    }
    gguf_close(&g);   // before the writes: the mapping is read-only
    const int fd = rc || !nt ? -1 : open(gguf, O_RDWR);
    if (!rc && nt && fd < 0) rc = snprintf(err, (size_t) errlen, "%s: cannot open for writing", gguf);
    for (int i = 0; fd >= 0 && !rc && i < nt; ++i) {
        const NsTensor* t = et[i].t[0];
        const size_t rb = (size_t) t->cols / 32 * 34;
        w = (float*) realloc(w, sizeof(float) * (size_t) t->cols);
        tmp = (float*) realloc(tmp, sizeof(float) * (size_t) t->cols);
        buf = (uint8_t*) realloc(buf, rb);
        for (int r = 0; !rc && r < t->rows; ++r) {
            shard_row(nr, t, gdn, et[i].lp, r, w, tmp);
            ns_q8_0_row(w, t->cols, buf);
            if (pwrite(fd, buf, rb, (off_t) (et[i].off + rb * (uint64_t) r)) != (ssize_t) rb) rc = snprintf(err, (size_t) errlen, "%s: a write failed", gguf);
        }
    }
    if (fd >= 0 && close(fd) && !rc) rc = snprintf(err, (size_t) errlen, "%s: a write failed", gguf);
    free(w); free(tmp); free(buf); free(nr); free(et);
    *n = rc ? 0 : nt;
    return rc ? -1 : 0;
}
