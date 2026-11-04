// nslm/lib_moe.c - MoVA and Qwen3.8 expert helpers (nslm/moe.h).
#include "moe.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

int nslm_qwen_experts(const char* config_json, int* d, int* ff, int* n_exp, int* n_layer) {
    char err[128];
    Json* j = json_parse(config_json, strlen(config_json), err, sizeof err);
    const Json* t = json_get(j, "text_config") ? json_get(j, "text_config") : j;
    const char* mt = json_gets(t, "model_type");
    int rc = j ? 0 : -1;
    if (mt && !strncmp(mt, "qwen4_exp", 9)) {
        const char* key[4] = {"hidden_size", "moe_intermediate_size", "num_experts", "num_hidden_layers"};
        int* dst[4] = {d, ff, n_exp, n_layer};
        rc = 1;
        for (int k = 0; k < 4; ++k) {
            const Json* v = json_get(t, key[k]);
            if (!v || v->t != J_INT || v->i < 1 || v->i > 1 << 20 || (k < 2 && v->i % 8)) rc = -1;
            else *dst[k] = (int) v->i;
        }
    }
    json_free(j);
    return rc;
}

void nslm_imatrix_h(const float* sum, const float* cnt, int E, int cols, int e, double n0, float* h) {
    double n = 0, *prior = (double*) calloc((size_t) cols, sizeof(double)), *own = (double*) malloc(sizeof(double) * (size_t) cols);
    for (int k = 0; k < E; ++k) {
        n += cnt[k];
        for (int c = 0; c < cols; ++c) prior[c] += sum[(size_t) k * cols + c];
    }
    for (int c = 0; c < cols; ++c) { prior[c] /= n > 0 ? n : 1; own[c] = sum[(size_t) e * cols + c]; }
    nslm_moe_blend(own, (int64_t) cnt[e], prior, n0, cols, h);
    free(prior);
    free(own);
}

int nslm_moe_index_lookup(const char* index_json, const char* name, char* file, int filelen) {
    const size_t nl = strlen(name);
    for (const char* p = strstr(index_json, name); p; p = strstr(p + 1, name)) {
        if (p == index_json || p[-1] != '"' || p[nl] != '"') continue;   // the whole quoted key
        const char* q = p + nl + 1;
        while (*q == ' ' || *q == ':' || *q == '\t' || *q == '\n') ++q;
        if (*q != '"') continue;
        const char* end = strchr(q + 1, '"');
        if (!end || end - q - 1 >= filelen) return -1;
        memcpy(file, q + 1, (size_t) (end - q - 1));
        file[end - q - 1] = 0;
        return 0;
    }
    return -1;
}

void nslm_moe_blend(const double* sum, int64_t n, const double* prior, double n0, int dim, float* h) {
    const double den = (double) n + n0;
    for (int c = 0; c < dim; ++c) h[c] = den > 0 ? (float) ((sum[c] + n0 * prior[c]) / den) : (float) prior[c];
}

int nslm_st_write_bf16_3d(const char* path, const char* name, int d0, int d1, int d2, const uint16_t* data, char* err,
                          int errlen) {
    const uint64_t bytes = 2ull * (uint64_t) d0 * (uint64_t) d1 * (uint64_t) d2;
    char hdr[512];
    int hn = snprintf(hdr, sizeof hdr, "{\"%s\":{\"dtype\":\"BF16\",\"shape\":[%d,%d,%d],\"data_offsets\":[0,%llu]}}", name,
                      d0, d1, d2, (unsigned long long) bytes);
    while (hn % 8) hdr[hn++] = ' ';
    char tmp[2048];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    const uint64_t n = (uint64_t) hn;
    int ok = f && fwrite(&n, 8, 1, f) == 1 && fwrite(hdr, 1, (size_t) hn, f) == (size_t) hn &&
             fwrite(data, 1, (size_t) bytes, f) == (size_t) bytes;
    if (f && fclose(f)) ok = 0;
    if (ok && rename(tmp, path)) ok = 0;
    if (!ok) {
        snprintf(err, (size_t) errlen, "cannot write %s: %s", path, strerror(errno));
        remove(tmp);
        return -1;
    }
    return 0;
}
