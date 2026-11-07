// nslm/qwen4_pack.c - nslm-qwen4-pack: an MLX qwen4_exp base folder with each layer's switch_mlp replaced by seeds, or
// (--tq) by trellis banks beside the shards (nslm_qwen4_pack_tq; --tq-down: down_proj's banks; --dense8: the quantized
// modules of an 8-bit MLX folder; the loader when there is one, for mlx-vlm and oMLX).
//   nslm-qwen4-pack --base DIR (--blk DIR | --blk4 DIR) --out DIR [--loader tools/nanoseedlm_qwen4.py] [--shard-gb 4.5]
//   nslm-qwen4-pack --base DIR --tq DIR [--tq-down DIR] [--dense8 DIR] --out DIR [--loader FILE] [--shard-gb 4.5]
//   nslm-qwen4-pack --tcq-st DIR [--loader FILE]   a trellis folder's .blkt banks as model-tcq-*.safetensors shards, in
//                                                  place (nslm_tcq_to_st); --loader: copy the MLX loader files again
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "format.h"
#include "json.h"
#include "model_st.h"
#include "moe.h"

static const char* kProj[3] = {"gate_proj", "up_proj", "down_proj"};

typedef struct { const uint8_t* src; int layer, proj; } Item;   // src: base bytes; NULL: seeds from a .blk4
typedef struct { const Item* it; const char* dir; int p, E, D, F, cur; uint8_t* buf; size_t len; } Ctx;

static const char* opt(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static char* read_all(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t) ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = (char*) malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    else b[*n] = 0;
    fclose(f);
    return b;
}
static Json* read_json(const char* dir, const char* file) {
    char path[2048], err[256];
    size_t n = 0;
    snprintf(path, sizeof path, "%s/%s", dir, file);
    char* s = read_all(path, &n);
    Json* j = s ? json_parse(s, n, err, sizeof err) : NULL;
    if (!j) fprintf(stderr, "%s: %s\n", path, s ? err : "cannot read");
    free(s);
    return j;
}
static int blk_path(char* p, int n, const char* dir, int P, int l, int proj) {
    return snprintf(p, (size_t) n, "%s/L%d_%s.blk%s", dir, l, kProj[proj], P == 4 ? "4" : "");
}
static int exists(const char* p) { struct stat sb; return !stat(p, &sb); }

static int fill(void* ctx, int i, int s, uint8_t* dst, uint64_t len) {
    Ctx* c = (Ctx*) ctx;
    const Item* t = &c->it[i];
    if (t->src) { memcpy(dst, t->src, len); return 0; }
    if (c->cur != i) {   // the streams of a seed tensor come in a row: read its block file once
        char path[2048];
        blk_path(path, sizeof path, c->dir, c->p, t->layer, t->proj);
        free(c->buf);
        if (!(c->buf = (uint8_t*) read_all(path, &c->len))) { fprintf(stderr, "cannot read %s\n", path); return -1; }
        c->cur = i;
    }
    const int down = t->proj == 2;
    if (nslm_blk_stream(c->buf, c->len, c->p, c->E, down ? c->D : c->F, down ? c->F : c->D, s, dst)) {
        fprintf(stderr, "L%d %s: not %d x %d x %d P = %d blocks\n", t->layer, kProj[t->proj], c->E, down ? c->D : c->F, down ? c->F : c->D, c->p);
        return -1;
    }
    return 0;
}

// The layer of a seeded switch_mlp key (language_model.model.layers.L.mlp.switch_mlp.*; L: the MTP head's), else -1.
static int seeded_layer(const char* key, const int* seeded, int L) {
    int l = -1, at = 0;
    if (!strncmp(key, "mtp.layers.0.mlp.switch_mlp.", 28)) return seeded[L] ? L : -1;
    if (sscanf(key, "language_model.model.layers.%d.mlp.switch_mlp.%n", &l, &at) != 1 || !at) return -1;
    return l >= 0 && l < L && seeded[l] ? l : -1;
}

// config.json of the base, with the loader and without quantization entries for the seeded projections.
static int write_config(Json* c, const char* out, const char* loader_name, const int* seeded, int L) {
    const char* qk[2] = {"quantization", "quantization_config"};
    for (int k = 0; k < 2; ++k) {
        Json* q = json_get(c, qk[k]);
        if (!q || q->t != J_OBJ) continue;
        Json* keep = json_new(J_OBJ);
        for (int i = 0; i < q->n; ++i) if (seeded_layer(q->k[i], seeded, L) < 0) json_set(keep, q->k[i], json_copy(q->v[i]));
        json_set(c, qk[k], keep);
    }
    json_set(c, "model_file", json_str(loader_name));
    char path[2048];
    snprintf(path, sizeof path, "%s/config.json", out);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    Buf b = {0};
    json_dump_indent(&b, c, 0);
    buf_puts(&b, "\n");
    fwrite(b.p, 1, b.n, f);
    free(b.p);
    return fclose(f);
}

// The default loader: tools/ of the working directory, else of the repository that built this tool (out/bin, build/bin)
static const char* find_loader(const char* exe, char* buf, size_t n) {
    FILE* f = fopen("tools/nanoseedlm_qwen4.py", "rb");
    if (f) { fclose(f); return "tools/nanoseedlm_qwen4.py"; }
    const char* sl = strrchr(exe, '/'), *bs = strrchr(exe, '\\');
    if (bs > sl) sl = bs;
    snprintf(buf, n, "%.*s../../tools/nanoseedlm_qwen4.py", sl ? (int) (sl - exe + 1) : 0, exe);
    return buf;
}

int main(int argc, char** argv) {
    const char* base = opt(argc, argv, "--base", NULL), *out = opt(argc, argv, "--out", NULL);
    const char* blk3 = opt(argc, argv, "--blk", NULL), *blk4 = opt(argc, argv, "--blk4", NULL), *blk = blk4 ? blk4 : blk3;
    const int P = blk4 ? 4 : 3;
    char lbuf[2048];
    const char* loader = opt(argc, argv, "--loader", NULL);
    if (!loader) loader = find_loader(argv[0], lbuf, sizeof lbuf);
    const uint64_t shard = (uint64_t) (atof(opt(argc, argv, "--shard-gb", "4.5")) * 1e9);
    const char* tq = opt(argc, argv, "--tq", NULL), *tst = opt(argc, argv, "--tcq-st", NULL);
    if (tst && !base && !out && !tq && !blk) {   // plan tcq-safetensors
        char e[512] = "";
        const int rc = nslm_tcq_to_st(tst, e, sizeof e);
        if (rc < 0) { fprintf(stderr, "%s\n", e); return 1; }
        if (opt(argc, argv, "--loader", NULL) && (nslm_copy_loader(loader, tst) || nslm_copy_beside(loader, "nanoseedlm_trellis.py", tst))) {
            fprintf(stderr, "%s: cannot copy the loader, nanoseedlm_seeds.py and nanoseedlm_trellis.py to %s\n", loader, tst);
            return 1;
        }
        printf(rc ? "%s: no .blkt bank, nothing to convert\n" : "%s: the trellis banks are model-tcq-*.safetensors shards now\n", tst);
        return 0;
    }
    if (tq && base && out && !blk) {
        char e[512];
        FILE* lf = fopen(loader, "rb");
        if (lf) fclose(lf);
        else printf("%s: no MLX loader, so mlx-vlm and oMLX cannot load %s (--loader FILE)\n", loader, out);
        if (nslm_qwen4_pack_tq(base, tq, opt(argc, argv, "--tq-down", NULL), opt(argc, argv, "--dense8", NULL), lf ? loader : NULL, out, shard, e,
                               sizeof e)) {
            fprintf(stderr, "%s\n", e);
            return 1;
        }
        printf("wrote %s: %s's tensors without the routed experts, and the trellis banks of %s\n", out, base, tq);
        return 0;
    }
    if (!base || !out || !blk3 == !blk4) {
        fprintf(stderr, "usage: nslm-qwen4-pack --base DIR (--blk DIR | --blk4 DIR | --tq DIR [--tq-down DIR] [--dense8 DIR]) --out DIR [--loader FILE] [--shard-gb 4.5]\n"
                        "       nslm-qwen4-pack --tcq-st DIR [--loader FILE]\n");
        return 2;
    }
    FILE* lf = fopen(loader, "rb");   // before the pack's hours, not after
    if (!lf) { fprintf(stderr, "%s: no loader (give --loader tools/nanoseedlm_qwen4.py)\n", loader); return 2; }
    fclose(lf);
    Json* cfg = read_json(base, "config.json"), *idx = read_json(base, "model.safetensors.index.json");
    if (!cfg || !idx) return 2;
    int D = 0, F = 0, E = 0, L = 0;
    char* cs = json_dumps(cfg, 1);
    const int q = nslm_qwen_experts(cs, &D, &F, &E, &L);
    free(cs);
    if (q != 1 || L > 255) { fprintf(stderr, "%s: not a qwen4_exp config\n", base); return 2; }
    int seeded[256] = {0}, ns = 0;
    char path[2048], err[512];
    for (int l = 0; l <= L; ++l) {   // l = L: the MTP head's blocks
        int have = 0;
        for (int p = 0; p < 3; ++p) { blk_path(path, sizeof path, blk, P, l, p); have += exists(path); }
        if (have && have != 3) { fprintf(stderr, "layer %d: %d of 3 block files\n", l, have); return 2; }
        ns += seeded[l] = have == 3;
    }
    const Json* wm = json_get(idx, "weight_map");
    if (!wm || wm->t != J_OBJ || !ns) { fprintf(stderr, "%s\n", ns ? "no weight_map in the base index" : "no layer has block files"); return 2; }

    StFile* sh = (StFile*) calloc((size_t) wm->n, sizeof(StFile));   // one per distinct shard file, opened on first use
    const char** shn = (const char**) calloc((size_t) wm->n, sizeof(char*));
    int nsh = 0, n = 0;
    NsSpec* spec = (NsSpec*) calloc((size_t) wm->n + 3 * (size_t) (L + 1), sizeof(NsSpec));
    NsRaw* raw = (NsRaw*) calloc((size_t) wm->n + 3 * (size_t) (L + 1), sizeof(NsRaw));
    Item* it = (Item*) calloc((size_t) wm->n + 3 * (size_t) (L + 1), sizeof(Item));
    uint64_t copied = 0;
    for (int i = 0; i < wm->n; ++i) {
        if (seeded_layer(wm->k[i], seeded, L) >= 0) continue;
        const char* file = wm->v[i]->s;
        int k = 0;
        while (k < nsh && strcmp(shn[k], file)) ++k;
        if (k == nsh) {
            snprintf(path, sizeof path, "%s/%s", base, file);
            if (st_open_file(&sh[k], path, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 2; }
            shn[nsh++] = file;
        }
        const StEntry* e = st_find(&sh[k], wm->k[i]);
        if (!e) { fprintf(stderr, "%s: not in %s\n", wm->k[i], file); return 2; }
        spec[n] = (NsSpec) {wm->k[i], NS_RAW, 1, 1, 1};
        raw[n].dtype = e->dtype;
        raw[n].ndim = e->ndim;
        memcpy(raw[n].shape, e->shape, sizeof e->shape);
        raw[n].bytes = e->end - e->begin;
        it[n++] = (Item) {st_data(&sh[k], e), -1, -1};
        copied += e->end - e->begin;
    }
    static char names[256 * 3][128];
    for (int l = 0; l <= L; ++l)
        for (int p = 0; seeded[l] && p < 3; ++p) {
            char* nm = names[3 * l + p];
            if (l == L) snprintf(nm, 128, "mtp.layers.0.mlp.switch_mlp.%s.weight", kProj[p]);
            else snprintf(nm, 128, "language_model.model.layers.%d.mlp.switch_mlp.%s.weight", l, kProj[p]);
            spec[n] = (NsSpec) {nm, P == 4 ? NS_SEED4P4 : NS_SEED4, E, p == 2 ? D : F, p == 2 ? F : D};
            it[n++] = (Item) {NULL, l, p};
        }
    mkdir(out, 0755);
    Ctx c = {it, blk, P, E, D, F, -1, NULL, 0};
    if (ns_write_raw(out, spec, raw, n, shard, P == 4 ? "\"nanoseedlm\": \"qwen4_exp, P = 4 routed experts\"" : "\"nanoseedlm\": \"qwen4_exp, P = 3 routed experts\"", fill, &c, err, sizeof err)) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    const char* sl = strrchr(loader, '/');
    if (write_config(cfg, out, sl ? sl + 1 : loader, seeded, L)) { fprintf(stderr, "%s/config.json: cannot write\n", out); return 1; }
    nslm_copy_model_files(base, out);
    if (nslm_copy_loader(loader, out)) { fprintf(stderr, "%s: cannot copy the loader and nanoseedlm_seeds.py\n", loader); return 1; }
    printf("wrote %s: %d of %d layers (and the MTP head: %s) with P = %d seed experts, %.2f GB copied from %s\n", out, ns - seeded[L], L,
           seeded[L] ? "yes" : "no", P, (double) copied / 1e9, base);
    for (int k = 0; k < nsh; ++k) st_close_file(&sh[k]);
    free(c.buf); free(sh); free(shn); free(spec); free(raw); free(it);
    json_free(cfg); json_free(idx);
    return 0;
}
