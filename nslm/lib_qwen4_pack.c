// nslm/lib_qwen4_pack.c - see moe.h (nslm_qwen4_pack_tq).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "format.h"
#include "json.h"
#include "model_st.h"
#include "moe.h"

static Json* load_json(const char* dir, const char* file, char* err, int errlen) {
    char path[2048];
    snprintf(path, sizeof path, "%s/%s", dir, file);
    FILE* f = fopen(path, "rb");
    const long n = f && !fseek(f, 0, SEEK_END) ? ftell(f) : -1;
    char* s = n > 0 && !fseek(f, 0, SEEK_SET) ? (char*) malloc((size_t) n) : NULL;
    Json* j = s && fread(s, 1, (size_t) n, f) == (size_t) n ? json_parse(s, (size_t) n, err, errlen) : NULL;
    if (!s) snprintf(err, (size_t) errlen, "%s: cannot read", path);
    if (f) fclose(f);
    free(s);
    return j;
}
static int routed(const char* key) { return strstr(key, ".mlp.switch_mlp.") != NULL; }   // any layer's, the head's too
static int raw_fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) {
    (void) s;
    memcpy(dst, ((const uint8_t* const*) ctx)[t], len);
    return 0;
}
// The group of module m in an 8-bit MLX folder's quantization (64 or 128), else 0
static int q8_group(const Json* q, const char* m) {
    const Json* o = json_get(q, m), *b = json_get(o ? o : q, "bits"), *g = json_get(o ? o : q, "group_size");
    return b && g && b->t == J_INT && b->i == 8 && g->t == J_INT && (g->i == 64 || g->i == 128) ? (int) g->i : 0;
}
// A module name and its part (weight, scales, biases) from a tensor key, else 0
static int module_of(const char* key, char* m, size_t n) {
    static const char* kPart[3] = {".weight", ".scales", ".biases"};
    const size_t kl = strlen(key);
    for (int i = 0; i < 3; ++i)
        if (kl > 7 && !strcmp(key + kl - 7, kPart[i]) && kl - 7 < n) { memcpy(m, key, kl - 7); m[kl - 7] = 0; return i + 1; }
    return 0;
}

// The config without the routed experts' quantization entries; model_file: the loader's name, or none
static Json* config_tq(const Json* cfg, const char* loader) {
    Json* c = json_new(J_OBJ);
    for (int i = 0; i < cfg->n; ++i) {
        if (!strcmp(cfg->k[i], "model_file")) continue;
        Json* v = json_copy(cfg->v[i]);
        if (v->t == J_OBJ && (!strcmp(cfg->k[i], "quantization") || !strcmp(cfg->k[i], "quantization_config"))) {
            Json* keep = json_new(J_OBJ);
            for (int k = 0; k < v->n; ++k) if (!routed(v->k[k])) json_set(keep, v->k[k], json_copy(v->v[k]));
            json_free(v);
            v = keep;
        }
        json_set(c, cfg->k[i], v);
    }
    if (loader) json_set(c, "model_file", json_str(strrchr(loader, '/') ? strrchr(loader, '/') + 1 : loader));
    return c;
}

int nslm_qwen4_pack_tq(const char* base, const char* tq, const char* tq_down, const char* dense8, const char* loader, const char* out,
                       uint64_t shard, char* err, int errlen) {
    static const char* kp[3] = {"gate_proj", "up_proj", "down_proj"};
    Json* cfg = load_json(base, "config.json", err, errlen), *idx = cfg ? load_json(base, "model.safetensors.index.json", err, errlen) : NULL;
    Json* cfg8 = dense8 && idx ? load_json(dense8, "config.json", err, errlen) : NULL;
    Json* idx8 = cfg8 ? load_json(dense8, "model.safetensors.index.json", err, errlen) : NULL;
    const Json* wm = idx ? json_get(idx, "weight_map") : NULL, *wm8 = idx8 ? json_get(idx8, "weight_map") : NULL;
    const Json* q8 = cfg8 ? json_get(cfg8, "quantization") : NULL;
    int D = 0, F = 0, E = 0, L = 0, head = 0, rc = cfg && idx && (!dense8 || (wm8 && q8)) ? 0 : -1;
    char* cs = cfg ? json_dumps(cfg, 1) : NULL;
    if (!rc && (!wm || wm->t != J_OBJ)) rc = snprintf(err, (size_t) errlen, "%s: no weight_map", base);
    else if (!rc && (nslm_qwen_experts(cs, &D, &F, &E, &L) != 1 || L > 255)) rc = snprintf(err, (size_t) errlen, "%s: not a qwen4_exp config", base);
    free(cs);
    for (int i = 0; !rc && i < wm->n; ++i) head |= routed(wm->k[i]) && !strncmp(wm->k[i], "mtp.", 4);
    char path[2048], to[2048];
    struct stat sb;
    for (int l = 0; !rc && l < L + head; ++l)   // l = L: the MTP head's banks (the packer's rule, ns_add_banks)
        for (int p = 0; !rc && p < 3; ++p) {
            snprintf(path, sizeof path, "%s/L%d_%s.blkt", p == 2 && tq_down ? tq_down : tq, l, kp[p]);
            if (stat(path, &sb)) rc = snprintf(err, (size_t) errlen, "%s: no such bank", path);
        }
    const int nk = wm && !rc ? wm->n : 1, nk8 = wm8 ? wm8->n : 1;
    StFile* sh = (StFile*) calloc((size_t) nk + (size_t) nk8, sizeof(StFile));
    const char** shn = (const char**) calloc((size_t) nk + (size_t) nk8, sizeof(char*));
    uint8_t** owned = (uint8_t**) calloc((size_t) nk, sizeof(uint8_t*));   // scales and biases regrouped from 128 to 64
    Json* swapped = json_new(J_ARR);
    NsSpec* spec = (NsSpec*) calloc((size_t) nk, sizeof(NsSpec));
    NsRaw* raw = (NsRaw*) calloc((size_t) nk, sizeof(NsRaw));
    const uint8_t** src = (const uint8_t**) calloc((size_t) nk, sizeof(uint8_t*));
    int nsh = 0, n = 0;
    for (int i = 0; !rc && i < wm->n; ++i) {   // every tensor but the routed experts: base's, or dense8's 8-bit module
        if (routed(wm->k[i])) continue;
        char m[160], ms[176];
        const int part = module_of(wm->k[i], m, sizeof m);
        snprintf(ms, sizeof ms, "%s.scales", m);
        const int g8 = dense8 && part && !strstr(m, ".ple.") && json_get(wm, ms) && json_get(wm8, ms) ? q8_group(q8, m) : 0;
        const Json* fv = g8 ? json_get(wm8, wm->k[i]) : wm->v[i];
        const char* file = fv && fv->t == J_STR ? fv->s : NULL, *dir = g8 ? dense8 : base;
        if (!file) { rc = snprintf(err, (size_t) errlen, "%s: not in %s", wm->k[i], dir); break; }
        int k = 0;
        snprintf(path, sizeof path, "%s/%s", dir, file);
        while (k < nsh && strcmp(shn[k], path)) ++k;
        if (k == nsh) {
            if (st_open_file(&sh[k], path, err, errlen)) { rc = -1; break; }
            shn[nsh++] = strdup(path);
        }
        const StEntry* e = st_find(&sh[k], wm->k[i]);
        if (!e || (g8 && e->ndim != 2)) { rc = snprintf(err, (size_t) errlen, "%s: not in %s as a 2-D tensor", wm->k[i], path); break; }
        spec[n] = (NsSpec) {wm->k[i], NS_RAW, 1, 1, 1};
        raw[n] = (NsRaw) {e->dtype, e->ndim, {0}, e->end - e->begin};
        memcpy(raw[n].shape, e->shape, sizeof raw[n].shape);
        src[n] = st_data(&sh[k], e);
        if (g8 == 128 && part > 1) {   // a group of 128 is two groups of 64 with its scale and bias: the same weights
            const uint64_t rows = (uint64_t) e->shape[0], ng = (uint64_t) e->shape[1], es = raw[n].bytes / (rows * ng);
            uint8_t* d = owned[n] = (uint8_t*) malloc(2 * raw[n].bytes);
            for (uint64_t r = 0; r < rows * ng; ++r) { memcpy(d + 2 * r * es, src[n] + r * es, es); memcpy(d + (2 * r + 1) * es, src[n] + r * es, es); }
            raw[n].shape[1] *= 2;
            raw[n].bytes *= 2;
            src[n] = d;
        }
        if (g8 && part == 1) json_push(swapped, json_str(m));
        ++n;
    }
    if (!rc) mkdir(out, 0755);
    if (!rc && ns_write_raw(out, spec, raw, n, shard, "\"nanoseedlm\": \"qwen4_exp, trellis-coded routed experts in L*_*.blkt\"",
                            raw_fill, src, err, errlen)) rc = -1;
    if (!rc) {
        Json* c = config_tq(cfg, loader);
        static const char* kq[2] = {"quantization", "quantization_config"};
        for (int k = 0; k < 2; ++k) {   // the swapped modules: 8 bits, groups of 64
            Json* q = json_get(c, kq[k]);
            for (int i = 0; q && q->t == J_OBJ && i < swapped->n; ++i) {
                static const char* k8 = "{\"group_size\": 64, \"bits\": 8, \"mode\": \"affine\"}";
                char e8[64];
                json_set(q, swapped->v[i]->s, json_parse(k8, strlen(k8), e8, sizeof e8));
            }
        }
        Buf b = {0};
        json_dump_indent(&b, c, 0);
        buf_puts(&b, "\n");
        snprintf(path, sizeof path, "%s/config.json", out);
        FILE* f = fopen(path, "wb");
        if (!f || fwrite(b.p, 1, b.n, f) != b.n) rc = snprintf(err, (size_t) errlen, "%s: cannot write", path);
        if (f && fclose(f) && !rc) rc = snprintf(err, (size_t) errlen, "%s: cannot write", path);
        free(b.p);
        json_free(c);
    }
    if (!rc) nslm_copy_model_files(base, out);
    if (!rc && loader && (nslm_copy_loader(loader, out) || nslm_copy_beside(loader, "nanoseedlm_trellis.py", out) ||
                          nslm_copy_beside(loader, "nanoseedlm_ple.py", out)))
        rc = snprintf(err, (size_t) errlen, "%s: cannot copy the loader, nanoseedlm_seeds.py, _trellis.py and _ple.py", loader);
    for (int l = 0; !rc && l < L + head; ++l)
        for (int p = 0; !rc && p < 3; ++p) {
            snprintf(path, sizeof path, "%s/L%d_%s.blkt", p == 2 && tq_down ? tq_down : tq, l, kp[p]);
            snprintf(to, sizeof to, "%s/L%d_%s.blkt", out, l, kp[p]);
            if (nslm_copy_file(path, to)) rc = snprintf(err, (size_t) errlen, "%s: cannot copy to %s", path, to);
        }
    for (int k = 0; k < nsh; ++k) { st_close_file(&sh[k]); free((char*) shn[k]); }
    for (int k = 0; k < nk; ++k) free(owned[k]);
    free(sh); free(shn); free(spec); free(raw); free(src); free(owned);
    json_free(cfg); json_free(idx); json_free(cfg8); json_free(idx8); json_free(swapped);
    return rc ? -1 : 0;
}
