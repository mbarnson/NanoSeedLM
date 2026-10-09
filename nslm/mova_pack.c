// nslm/mova_pack.c - nslm-mova-pack: writes a NanoSeedLM model folder for K2-Horizon MoVA (nslm/model_st.h).
//
//   nslm-mova-pack --model DIR --config CONFIG --out DIR [--blk DIR] [--blk4 DIR] [--q4 PARTS|--rest4]
//                  [--mla bf16|q8|q4|p4 [--mla-blk4 DIR] [--mla-q8 NAMES]] [--threads 16]
//   --mla-q8: MLA tensors kept in Q8 while the rest take --mla, comma-separated, each optionally for layers A-B (v_up@0-23)
//                  [--shard-gb 4.5] [--loader tools/nanoseedlm_k2.py]
//
//   CONFIG  routed experts                                                  needs
//   q8mx    Q8
//   q4mx    Q4
//   nslmmx  gate/up/down as seeds (SEED4)                                   --blk
//   gu4d    gate/up as seeds (SEED4), down Q4                               --blk
//   gup4d   gate/up as seeds (SEED4), down BF16 decoded from P = 4 blocks   --blk, --blk4
//   p4mx    gate/up/down as 4.5-bit P = 4 seeds (SEED4P4)                   --blk4
//   p4mxbf  gate/up/down BF16 decoded from P = 4 blocks                     --blk4
// --blk DIR holds nslm-moe's L{l}_{proj}.blk files, --blk4 DIR its L{l}_{proj}.blk4 files (nslm-moe --p4).
//
// Value experts, attention, shared / dense MLPs, embedding and LM head are Q8, or Q4 for the parts named in --q4
// (v value experts, a attention, m shared / dense MLPs, h LM head; --rest4 = vamh; the embedding is always Q8);
// routers are BF16 holding their Q8 round trip (the router needs BF16 operands); norms and router biases are BF16.
// MLA models (a TransMLA conversion): the latent projections and per-head maps stay BF16, as exported, or Q8 / Q4
// with --mla (per-head maps quantized per head along their input dim; a map or projection whose input dim is not a
// multiple of 64, e.g. v_up at rank 96, stays BF16), or P = 4 seeds (--mla p4: SEED4P4 from --mla-blk4's nslm-moe
// --scope mla blocks, stored as NAME.weight); such a folder is
// for the NanoSeedLM engine (no MLX loader reads MLA yet, so config.json gets no "model_file").
// Q8/Q4 are MLX's affine g64 (affine.h).  The folder also gets config.json (with MLX's "quantization" and, for seeds,
// "model_file": the MLX loader), the tokenizer and template files of DIR, and the loader.
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

#include "affine.h"
#include "json.h"
#include "model_st.h"
#include "search4.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"

typedef struct {
    char magic[8];
    uint32_t rows, cols, n_experts;
    uint32_t n_seeds, n_exp, refit;
    int32_t exp_delta[3];
    float n0;
    double seconds;
} MoeBlkHeader;   // nslm/moe.c

static int has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) if (!strcmp(argv[i], name)) return 1;
    return 0;
}
static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}

typedef struct {
    MovaCkpt* ck;
    const MovaTensor* src;       // logical tensor per directory entry
    const char* blk;
    const char* blk4;            // --blk4: the P = 4 blocks (nslm-moe --p4)
    const char* blk4mla;         // --mla-blk4: the MLA projections' P = 4 blocks (nslm-moe --scope mla --p4)
    int threads;
    // cache for the current quantized tensor (streams 0, 1, 2 come in order)
    int cur;
    uint32_t* words;
    uint16_t *scales, *biases;
    // seed cache
    uint8_t* blkbuf;
    size_t blklen;
} Ctx;

typedef struct {
    Ctx* c;
    const MovaTensor* t;
    int bits;
    atomic_int next;
    atomic_int failed;
} QJob;

static void* qworker(void* arg) {
    QJob* j = (QJob*) arg;
    const MovaTensor* t = j->t;
    char name[128], err[256];
    const int chunk = t->slices > 1 ? 1 : 256;   // stacked: one slice per job item; else 256-row bands
    const int items = t->slices > 1 ? t->slices : (t->rows + chunk - 1) / chunk;
    for (;;) {
        const int k = atomic_fetch_add(&j->next, 1);
        if (k >= items) break;
        const int slice = t->slices > 1 ? k : 0;
        const uint16_t* w = t->kind == MOVA_K_HEADS   // MLA's per-head maps: one stacked 3-D tensor
            ? mova_ckpt_bf16_3d(j->c->ck, t->name, t->slices, t->rows, t->cols, err, sizeof err)
            : mova_ckpt_bf16(j->c->ck, mova_slice_name(t, slice, name, sizeof name), t->rows, t->cols, err, sizeof err);
        if (w && t->kind == MOVA_K_HEADS) w += (size_t) slice * t->rows * t->cols;
        if (!w) { fprintf(stderr, "%s\n", err); atomic_store(&j->failed, 1); break; }
        const int r0 = t->slices > 1 ? 0 : k * chunk, nr = t->slices > 1 ? t->rows : (t->rows - r0 < chunk ? t->rows - r0 : chunk);
        const size_t e0 = (size_t) slice * t->rows * t->cols + (size_t) r0 * t->cols;   // first element
        nslm_affine_quantize(w + (size_t) r0 * t->cols, nr, t->cols, j->bits, j->c->words + e0 * j->bits / 32,
                             j->c->scales + e0 / 64, j->c->biases + e0 / 64);
    }
    return NULL;
}

static int quantize_tensor(Ctx* c, const MovaTensor* t, int bits) {
    const size_t n = (size_t) t->slices * t->rows * t->cols;
    free(c->words); free(c->scales); free(c->biases);
    c->words = (uint32_t*) malloc(n * (size_t) bits / 8);
    c->scales = (uint16_t*) malloc(n / 64 * 2);
    c->biases = (uint16_t*) malloc(n / 64 * 2);
    QJob j = {c, t, bits, 0, 0};
    pthread_t th[64];
    for (int i = 0; i < c->threads; ++i) pthread_create(&th[i], NULL, qworker, &j);
    for (int i = 0; i < c->threads; ++i) pthread_join(th[i], NULL);
    return atomic_load(&j.failed) ? -1 : 0;
}

// gup4d / p4mxbf: the stacked [E][rows][cols] BF16 experts decoded from a .blk4 (header, int32 bias[E], float rel[E],
// float wrel[E], uint16 seed[E][nb], uint16 coef[E][nb], uint8 ecode[E][nb]).
typedef struct { const uint8_t* seed, *coef, *ec; const int32_t* bias; uint16_t* dst; size_t nb; int E, t, nt; } Dec4;
static void* dec4_worker(void* arg) {
    Dec4* d = (Dec4*) arg;
    const uint16_t* sd = (const uint16_t*) d->seed, *cf = (const uint16_t*) d->coef;
    for (int e = d->t; e < d->E; e += d->nt)
        for (size_t b = 0; b < d->nb; ++b) {
            const size_t k = (size_t) e * d->nb + b;
            nslm4_decode_block(sd[k], cf[k], d->bias[e] + d->ec[k], d->dst + k * NSLM4_C);
        }
    return NULL;
}
static int decode_blk4(Ctx* c, const MovaTensor* t, uint8_t* dst, uint64_t len) {
    char path[1200];
    snprintf(path, sizeof path, "%s/L%d_%s.blk4", c->blk4, t->layer, t->proj);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    const size_t n = (size_t) ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*) malloc(n);
    const size_t got = fread(buf, 1, n, f);
    fclose(f);
    const MoeBlkHeader* h = (const MoeBlkHeader*) buf;
    const size_t E = (size_t) t->slices, nb = (size_t) t->rows * t->cols / 8;
    if (got != n || memcmp(h->magic, "NSLMBLK4", 8) || h->rows != (uint32_t) t->rows || h->cols != (uint32_t) t->cols ||
        h->n_experts != (uint32_t) E || n != sizeof *h + 12 * E + 5 * nb * E || len != 2 * (uint64_t) E * nb * 8) {
        fprintf(stderr, "%s: not the expected P = 4 seed file\n", path);
        free(buf);
        return -1;
    }
    const uint8_t* base = buf + sizeof *h;
    Dec4 d[64];
    pthread_t th[64];
    const int nt = c->threads;
    for (int i = 0; i < nt; ++i) {
        d[i] = (Dec4){base + 12 * E, base + 12 * E + 2 * nb * E, base + 12 * E + 4 * nb * E, (const int32_t*) base, (uint16_t*) dst, nb,
                      (int) E, i, nt};
        pthread_create(&th[i], NULL, dec4_worker, &d[i]);
    }
    for (int i = 0; i < nt; ++i) pthread_join(th[i], NULL);
    free(buf);
    return 0;
}

// p4mx: SEED4P4 streams from a .blk4: s0 seeds, s1 coefficient words, s2 int32 bias[E], s3 exponent codes (2 per byte).
static int streams_blk4(Ctx* c, const MovaTensor* t, int s, uint8_t* dst, uint64_t len) {
    const size_t E = (size_t) t->slices, nb = (size_t) t->rows * t->cols / 8;
    if (s == 0) {
        char path[1200];
        const char* mla = strstr(t->name, ".mla.");   // MLA projections: --mla-blk4's L{l}_{name}.blk4 (nslm-moe --scope mla)
        if (mla) snprintf(path, sizeof path, "%s/L%d_%s.blk4", c->blk4mla, t->layer, mla + 5);
        else snprintf(path, sizeof path, "%s/L%d_%s.blk4", c->blk4, t->layer, t->proj);
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", path); return -1; }
        fseek(f, 0, SEEK_END);
        c->blklen = (size_t) ftell(f);
        fseek(f, 0, SEEK_SET);
        free(c->blkbuf);
        c->blkbuf = (uint8_t*) malloc(c->blklen);
        const size_t got = fread(c->blkbuf, 1, c->blklen, f);
        fclose(f);
        const MoeBlkHeader* h = (const MoeBlkHeader*) c->blkbuf;
        if (got != c->blklen || memcmp(h->magic, "NSLMBLK4", 8) || h->rows != (uint32_t) t->rows || h->cols != (uint32_t) t->cols ||
            h->n_experts != (uint32_t) E || c->blklen != sizeof *h + 12 * E + 5 * nb * E) {
            fprintf(stderr, "%s: not the expected P = 4 seed file\n", path);
            return -1;
        }
    }
    const uint8_t* base = c->blkbuf + sizeof(MoeBlkHeader);
    if (s == 0 || s == 1) { memcpy(dst, base + 12 * E + (s == 1 ? 2 * nb * E : 0), len); return 0; }
    if (s == 2) { memcpy(dst, base, 4 * E); return 0; }              // int32 bias per slice
    const uint8_t* ec = base + 12 * E + 4 * nb * E;
    uint8_t* nib = dst;
    for (size_t k = 0; k < nb * E; k += 2) nib[k / 2] = (uint8_t) ((ec[k] & 15) | ((k + 1 < nb * E ? ec[k + 1] & 15 : 0) << 4));
    return 0;
}

static int fill(void* ctx, int ti, int s, uint8_t* dst, uint64_t len) {
    Ctx* c = (Ctx*) ctx;
    const MovaTensor* t = &c->src[ti];
    extern int g_enc[];
    const int enc = g_enc[ti];
    char err[256], path[1200];
    if (enc == NS_Q8 || enc == NS_Q4) {
        if (s == 0 && quantize_tensor(c, t, enc == NS_Q8 ? 8 : 4)) return -1;
        memcpy(dst, s == 0 ? (void*) c->words : s == 1 ? (void*) c->scales : (void*) c->biases, len);
        return 0;
    }
    if (enc == NS_BF16 && t->kind == MOVA_K_EXPERTS) return decode_blk4(c, t, dst, len);
    if (enc == NS_SEED4P4) return streams_blk4(c, t, s, dst, len);
    if (enc == NS_BF16 && t->kind == MOVA_K_HEADS) {   // MLA's per-head maps: one stacked 3-D tensor
        const uint16_t* w = mova_ckpt_bf16_3d(c->ck, t->name, t->slices, t->rows, t->cols, err, sizeof err);
        if (!w) { fprintf(stderr, "%s\n", err); return -1; }
        memcpy(dst, w, len);
        return 0;
    }
    if (enc == NS_BF16) {   // a router: its Q8 round trip as BF16; norms, router biases and MLA projections verbatim
        const uint16_t* w = mova_ckpt_bf16(c->ck, t->name, t->rows, t->cols, err, sizeof err);
        if (!w) { fprintf(stderr, "%s\n", err); return -1; }
        if (t->kind != MOVA_K_ROUTER) { memcpy(dst, w, len); return 0; }
        const size_t n = (size_t) t->rows * t->cols;
        uint32_t* q = (uint32_t*) malloc(n);
        uint16_t* sc = (uint16_t*) malloc(n / 32), *bi = (uint16_t*) malloc(n / 32);
        nslm_affine_quantize(w, t->rows, t->cols, 8, q, sc, bi);
        nslm_affine_dequantize(q, sc, bi, t->rows, t->cols, 8, (uint16_t*) dst);
        free(q); free(sc); free(bi);
        return 0;
    }
    // SEED4: from the nslm-moe .blk of (layer, projection)
    if (s == 0) {
        snprintf(path, sizeof path, "%s/L%d_%s.blk", c->blk, t->layer, t->proj);
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "cannot read %s\n", path); return -1; }
        fseek(f, 0, SEEK_END);
        c->blklen = (size_t) ftell(f);
        fseek(f, 0, SEEK_SET);
        free(c->blkbuf);
        c->blkbuf = (uint8_t*) malloc(c->blklen);
        const size_t got = fread(c->blkbuf, 1, c->blklen, f);
        fclose(f);
        const MoeBlkHeader* h = (const MoeBlkHeader*) c->blkbuf;
        const size_t nb = (size_t) t->rows * t->cols / 8, E = (size_t) t->slices;
        if (got != c->blklen || memcmp(h->magic, "NSLMBLKM", 8) || h->rows != (uint32_t) t->rows || h->cols != (uint32_t) t->cols ||
            h->n_experts != (uint32_t) t->slices || h->n_seeds != 65535 ||
            c->blklen != sizeof *h + 12 * E + 4 * nb * E) {
            fprintf(stderr, "%s: not the expected seed file\n", path);
            return -1;
        }
    }
    const size_t E = (size_t) t->slices, nb = (size_t) t->rows * t->cols / 8;
    const uint8_t* base = c->blkbuf + sizeof(MoeBlkHeader);
    if (s == 2) memcpy(dst, base, 4 * E);                                   // int32 bias per slice
    else memcpy(dst, base + 12 * E + (s == 1 ? 2 * nb * E : 0), len);       // seeds, then nibbles
    return 0;
}

int g_enc[2048];

static int copy_file(const char* from, const char* to) {
    FILE* a = fopen(from, "rb");
    if (!a) return -1;
    FILE* b = fopen(to, "wb");
    if (!b) { fclose(a); return -1; }
    char buf[1 << 16];
    for (size_t n; (n = fread(buf, 1, sizeof buf, a)) > 0;) fwrite(buf, 1, n, b);
    fclose(a);
    return fclose(b);
}

// config.json of the source with MLX's quantization block and, for seeds, the loader.
static int write_config(const char* model, const char* out, const NsSpec* t, int n, int seeds, int mla, const char* loader_name) {
    char path[2048];
    snprintf(path, sizeof path, "%s/config.json", model);
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    const long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* s = (char*) malloc((size_t) len);
    const size_t got = fread(s, 1, (size_t) len, f);
    fclose(f);
    char err[256];
    Json* c = json_parse(s, got, err, sizeof err);
    free(s);
    if (!c || c->t != J_OBJ) return -1;
    Json* keep = json_new(J_OBJ);   // auto_map names PyTorch files this folder does not ship
    for (int i = 0; i < c->n; ++i)   // MLA: the source's MLX model file reads BF16 weights only
        if (strcmp(c->k[i], "auto_map") && !(mla && !strcmp(c->k[i], "model_file"))) json_set(keep, c->k[i], json_copy(c->v[i]));
    json_free(c);
    c = keep;
    Json* q = json_new(J_OBJ);
    Json* v = json_new(J_INT); v->i = 64; json_set(q, "group_size", v);
    v = json_new(J_INT); v->i = 8; json_set(q, "bits", v);
    json_set(q, "mode", json_str("affine"));
    for (int i = 0; i < n; ++i) {
        if (t[i].enc != NS_Q4) continue;
        char key[128];
        snprintf(key, sizeof key, "%.*s", (int) (strlen(t[i].name) - 7), t[i].name);
        Json* e = json_new(J_OBJ);
        v = json_new(J_INT); v->i = 64; json_set(e, "group_size", v);
        v = json_new(J_INT); v->i = 4; json_set(e, "bits", v);
        json_set(q, key, e);
    }
    json_set(c, "quantization", q);
    if (seeds && !mla) json_set(c, "model_file", json_str(loader_name));
    Buf b = {0};
    json_dump_indent(&b, c, 0);
    buf_puts(&b, "\n");
    json_free(c);
    snprintf(path, sizeof path, "%s/config.json", out);
    f = fopen(path, "wb");
    if (!f) { free(b.p); return -1; }
    fwrite(b.p, 1, b.n, f);
    free(b.p);
    return fclose(f);
}

// --mla-q8 LIST: comma-separated MLA tensor names, each optionally for layers A-B only (v_up@0-23); 1 when the MLA tensor
// `name` (model.layers.L.self_attn.mla.NAME[.weight]) is listed
static int mla_keep_q8(const char* list, const char* name) {
    const char* m = strstr(name, ".mla."), *ly = strstr(name, "layers.");
    if (!m) return 0;
    m += 5;
    const size_t mn = strcspn(m, ".");
    const int l = ly ? atoi(ly + 7) : -1;
    for (const char* p = list; *p;) {
        const size_t n = strcspn(p, ",@");
        int a = -1, b = 1 << 30;
        const char* q = p + n;
        if (*q == '@' && sscanf(q + 1, "%d-%d", &a, &b) != 2) { a = atoi(q + 1); b = a; }
        if (n == mn && !strncmp(p, m, n) && (a < 0 || (l >= a && l <= b))) return 1;
        p = q + strcspn(q, ",");
        if (*p == ',') ++p;
    }
    return 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // (_IOLBF with size 0 fails fast in MSVC's CRT)
    const char* model = opt(argc, argv, "--model", NULL), *config = opt(argc, argv, "--config", NULL);
    const char* out = opt(argc, argv, "--out", NULL), *blk = opt(argc, argv, "--blk", NULL);
    const char* loader = opt(argc, argv, "--loader", "tools/nanoseedlm_k2.py");
    if (!model || !config || !out) {
        fprintf(stderr, "usage: nslm-mova-pack --model DIR --config q8mx|q4mx|nslmmx|gu4d|gup4d|p4mx|p4mxbf [--blk DIR] [--blk4 DIR] "
                        "[--q4 vamh|--rest4] [--mla bf16|q8|q4|p4 [--mla-blk4 DIR] [--mla-q8 NAMES]] --out DIR [--threads N] [--shard-gb 4.5] [--loader FILE]\n");
        return 2;
    }
    const int q8mx = !strcmp(config, "q8mx"), q4mx = !strcmp(config, "q4mx"), mx = !strcmp(config, "nslmmx"),
              gu4d = !strcmp(config, "gu4d"), gup4d = !strcmp(config, "gup4d"), p4mx = !strcmp(config, "p4mx"),
              p4mxbf = !strcmp(config, "p4mxbf");
    if (!(q8mx || q4mx || mx || gu4d || gup4d || p4mx || p4mxbf)) { fprintf(stderr, "unknown --config %s\n", config); return 2; }
    const char* q4parts = has_flag(argc, argv, "--rest4") ? "vamh" : opt(argc, argv, "--q4", "");
    const char* mla = opt(argc, argv, "--mla", "bf16");   // the MLA projections (and per-head maps)
    const int mla_enc = !strcmp(mla, "q8") ? NS_Q8 : !strcmp(mla, "q4") ? NS_Q4 : !strcmp(mla, "p4") ? NS_SEED4P4
                      : !strcmp(mla, "bf16") ? NS_BF16 : -1;
    if (mla_enc < 0) { fprintf(stderr, "--mla: bf16, q8, q4 or p4\n"); return 2; }
    if (mla_enc == NS_SEED4P4 && !opt(argc, argv, "--mla-blk4", NULL)) { fprintf(stderr, "--mla p4 needs --mla-blk4 DIR\n"); return 2; }
    const char* mla_q8 = opt(argc, argv, "--mla-q8", "");   // MLA tensors kept in Q8 (mla_keep_q8)
    if ((gup4d || p4mx || p4mxbf) && !opt(argc, argv, "--blk4", NULL)) { fprintf(stderr, "--config %s needs --blk4\n", config); return 2; }
    if ((mx || gu4d || gup4d) && !blk) { fprintf(stderr, "--config %s needs --blk\n", config); return 2; }
    char err[512] = "";
    MovaCfg cfg;
    if (mova_cfg_load(&cfg, model, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 2; }
    MovaCkpt* ck = mova_ckpt_open(model, err, sizeof err);
    if (!ck) { fprintf(stderr, "%s\n", err); return 2; }
    MovaTensor* all = NULL;
    const int na = mova_tensors(&cfg, &all);
    static MovaTensor sel[2048];
    static NsSpec spec[2048];
    int n = 0, seeds = 0;
    double params = 0, bytes = 0;
    for (int i = 0; i < na; ++i) {
        const MovaTensor* t = &all[i];
        int enc = -1;
        switch (t->kind) {
        case MOVA_K_EXPERTS:
            if (q8mx) enc = NS_Q8;
            else if (p4mxbf) enc = NS_BF16;   // decoded from the P = 4 blocks
            else if (p4mx) enc = NS_SEED4P4;
            else if (q4mx || (gu4d && !strcmp(t->proj, "down_proj"))) enc = NS_Q4;
            else if (gup4d && !strcmp(t->proj, "down_proj")) enc = NS_BF16;   // decoded from the P = 4 blocks
            else if (mx || strcmp(t->proj, "down_proj")) enc = NS_SEED4;
            break;
        case MOVA_K_ROUTER: case MOVA_K_NORM: case MOVA_K_ROUTER_BIAS: enc = NS_BF16; break;
        case MOVA_K_HEADS: enc = mla_keep_q8(mla_q8, t->name) ? NS_Q8 : mla_enc; break;
        case MOVA_K_VEXPERTS: case MOVA_K_LINEAR: case MOVA_K_EMBED: case MOVA_K_HEAD: {
            if (strstr(t->name, ".mla.")) { enc = mla_keep_q8(mla_q8, t->name) ? NS_Q8 : mla_enc; break; }   // --mla
            // --q4 PARTS: v value experts, a attention, m shared / dense MLPs, h LM head (--rest4 = vamh)
            const int part = t->kind == MOVA_K_VEXPERTS ? 'v' : t->kind == MOVA_K_HEAD ? 'h' : t->kind == MOVA_K_EMBED ? 0
                           : strstr(t->name, "self_attn") ? 'a' : 'm';
            enc = part && strchr(q4parts, part) ? NS_Q4 : NS_Q8;
            break;
        }
        default: break;
        }
        if (enc < 0) { fprintf(stderr, "%s: no encoding for tensor kind %d\n", t->name, t->kind); return 2; }
        if ((enc == NS_Q8 || enc == NS_Q4) && t->cols % 64 && (t->kind == MOVA_K_HEADS || strstr(t->name, ".mla."))) {
            printf("%s: %d columns (not whole groups of 64), kept BF16\n", t->name, t->cols);   // e.g. v_up at rank 96
            enc = NS_BF16;
        }
        sel[n] = *t;
        g_enc[n] = enc;
        static char wname[2048][176];   // seed-encoded MLA projections are stored as NAME.weight (model_st.h's seed names)
        snprintf(wname[n], sizeof wname[n], "%s%s", sel[n].name, enc == NS_SEED4P4 && strstr(sel[n].name, ".mla.") ? ".weight" : "");
        spec[n] = (NsSpec) {wname[n], enc, t->slices, t->rows, t->cols};
        seeds |= enc == NS_SEED4 || enc == NS_SEED4P4;
        params += (double) t->slices * t->rows * t->cols;
        for (int s = 0; s < 4; ++s) bytes += (double) ns_stream_len(enc, t->slices, t->rows, t->cols, s);
        ++n;
    }
    printf("%s: %d tensors, %.3f B params, %.2f GB\n", config, n, params / 1e9, bytes / 1e9);
    if (mkdir(out, 0755) && errno != EEXIST) { perror(out); return 1; }
    Ctx c;
    memset(&c, 0, sizeof c);
    c.ck = ck; c.src = sel; c.blk = blk; c.blk4 = opt(argc, argv, "--blk4", NULL); c.blk4mla = opt(argc, argv, "--mla-blk4", NULL); c.threads = atoi(opt(argc, argv, "--threads", "16"));
    if (c.threads < 1 || c.threads > 64) c.threads = 16;
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    const double t0 = (double) ts0.tv_sec + 1e-9 * (double) ts0.tv_nsec;
    const uint64_t shard = (uint64_t) (atof(opt(argc, argv, "--shard-gb", "4.5")) * 1e9);
    char meta[128];
    snprintf(meta, sizeof meta, "\"nanoseedlm\": \"%s\"", config);
    if (ns_write(out, spec, n, shard, meta, fill, &c, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }

    const char* sl = strrchr(loader, '/');
    const char* loader_name = sl ? sl + 1 : loader;
    if (write_config(model, out, spec, n, seeds, cfg.mla, loader_name)) { fprintf(stderr, "%s/config.json: cannot write\n", out); return 1; }
    static const char* files[] = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                                  "generation_config.json"};
    char from[2048], to[2048];
    for (size_t i = 0; i < sizeof files / sizeof files[0]; ++i) {
        snprintf(from, sizeof from, "%s/%s", model, files[i]);
        snprintf(to, sizeof to, "%s/%s", out, files[i]);
        copy_file(from, to);   // optional files
    }
    snprintf(to, sizeof to, "%s/%s", out, loader_name);
    if (seeds && !cfg.mla && copy_file(loader, to)) { fprintf(stderr, "%s: cannot copy the MLX loader\n", loader); return 1; }
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    printf("wrote %s in %.1f s\n", out, (double) ts1.tv_sec + 1e-9 * (double) ts1.tv_nsec - t0);
    free(c.words); free(c.scales); free(c.biases); free(c.blkbuf); free(all);
    mova_ckpt_close(ck);
    return 0;
}
