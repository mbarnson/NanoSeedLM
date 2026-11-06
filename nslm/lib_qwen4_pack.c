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
static Json* config_tq(const Json* cfg) {   // no loader, no quantization entries of the routed experts
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
    return c;
}

int nslm_qwen4_pack_tq(const char* base, const char* tq, const char* tq_down, const char* out, uint64_t shard, char* err, int errlen) {
    static const char* kp[3] = {"gate_proj", "up_proj", "down_proj"};
    Json* cfg = load_json(base, "config.json", err, errlen), *idx = cfg ? load_json(base, "model.safetensors.index.json", err, errlen) : NULL;
    const Json* wm = idx ? json_get(idx, "weight_map") : NULL;
    int D = 0, F = 0, E = 0, L = 0, head = 0, rc = cfg && idx ? 0 : -1;
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
    const int nk = wm && !rc ? wm->n : 1;
    StFile* sh = (StFile*) calloc((size_t) nk, sizeof(StFile));
    const char** shn = (const char**) calloc((size_t) nk, sizeof(char*));
    NsSpec* spec = (NsSpec*) calloc((size_t) nk, sizeof(NsSpec));
    NsRaw* raw = (NsRaw*) calloc((size_t) nk, sizeof(NsRaw));
    const uint8_t** src = (const uint8_t**) calloc((size_t) nk, sizeof(uint8_t*));
    int nsh = 0, n = 0;
    for (int i = 0; !rc && i < wm->n; ++i) {   // every tensor but the routed experts, as it is
        if (routed(wm->k[i])) continue;
        const char* file = wm->v[i]->s;
        int k = 0;
        while (k < nsh && strcmp(shn[k], file)) ++k;
        if (k == nsh) {
            snprintf(path, sizeof path, "%s/%s", base, file);
            if (st_open_file(&sh[k], path, err, errlen)) { rc = -1; break; }
            shn[nsh++] = file;
        }
        const StEntry* e = st_find(&sh[k], wm->k[i]);
        if (!e) { rc = snprintf(err, (size_t) errlen, "%s: not in %s", wm->k[i], file); break; }
        spec[n] = (NsSpec) {wm->k[i], NS_RAW, 1, 1, 1};
        raw[n] = (NsRaw) {e->dtype, e->ndim, {0}, e->end - e->begin};
        memcpy(raw[n].shape, e->shape, sizeof raw[n].shape);
        src[n++] = st_data(&sh[k], e);
    }
    if (!rc) mkdir(out, 0755);
    if (!rc && ns_write_raw(out, spec, raw, n, shard, "\"nanoseedlm\": \"qwen4_exp, trellis-coded routed experts in L*_*.blkt\"",
                            raw_fill, src, err, errlen)) rc = -1;
    if (!rc) {
        Json* c = config_tq(cfg);
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
    for (int l = 0; !rc && l < L + head; ++l)
        for (int p = 0; !rc && p < 3; ++p) {
            snprintf(path, sizeof path, "%s/L%d_%s.blkt", p == 2 && tq_down ? tq_down : tq, l, kp[p]);
            snprintf(to, sizeof to, "%s/L%d_%s.blkt", out, l, kp[p]);
            if (nslm_copy_file(path, to)) rc = snprintf(err, (size_t) errlen, "%s: cannot copy to %s", path, to);
        }
    for (int k = 0; k < nsh; ++k) st_close_file(&sh[k]);
    free(sh); free(shn); free(spec); free(raw); free(src);
    json_free(cfg); json_free(idx);
    return rc ? -1 : 0;
}
