// nslm/lib_moe.c - MoVA helpers (nslm/moe.h).
#include "moe.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

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
