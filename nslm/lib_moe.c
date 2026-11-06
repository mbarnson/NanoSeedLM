// nslm/lib_moe.c - MoVA and qwen4 expert helpers (nslm/moe.h).
#include "moe.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"
#include "trellis.h"

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

int nslm_blk_stream(const uint8_t* blk, size_t len, int p, int E, int rows, int cols, int s, uint8_t* dst) {
    const size_t e = (size_t) E, nb = (size_t) rows * (size_t) cols / 8;
    const MoeBlkHeader* h = (const MoeBlkHeader*) blk;
    if ((p != 3 && p != 4) || (p == 3 && s > 2) || len != sizeof *h + 12 * e + (p == 4 ? 5 : 4) * nb * e ||
        memcmp(h->magic, p == 4 ? "NSLMBLK4" : "NSLMBLKM", 8) || h->rows != (uint32_t) rows || h->cols != (uint32_t) cols ||
        h->n_experts != (uint32_t) E)
        return -1;
    const uint8_t* base = blk + sizeof *h;
    if (s == 0 || s == 1) memcpy(dst, base + 12 * e + (s == 1 ? 2 * nb * e : 0), 2 * nb * e);
    else if (s == 2) memcpy(dst, base, 4 * e);
    else {
        const uint8_t* ec = base + 12 * e + 4 * nb * e;
        for (size_t k = 0; k < nb * e; k += 2) dst[k / 2] = (uint8_t) ((ec[k] & 15) | ((k + 1 < nb * e ? ec[k + 1] & 15 : 0) << 4));
    }
    return 0;
}

static int bank_cmp(const void* a, const void* b) { return strcmp(((const NsTensor*) a)->name, ((const NsTensor*) b)->name); }

// One bank file mapped: S3 seeds (.blks) or trellis-coded rows (.blkt, nslm-moe --codec tq). NULL, with *len = 0 when
// there is no file
static const MoeBlkHeader* map_bank(const char* path, int tcq, size_t* len) {
    const int fd = open(path, O_RDONLY);
    struct stat sb;
    *len = 0;
    if (fd < 0) return NULL;
    *len = fstat(fd, &sb) || !sb.st_size ? 1 : (size_t) sb.st_size;   // an empty file is a wrong one
    const MoeBlkHeader* h = *len >= sizeof *h ? (const MoeBlkHeader*) mmap(NULL, *len, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
    close(fd);
    if (h == (const void*) MAP_FAILED) return NULL;
    if (h && !memcmp(h->magic, tcq ? "NSLMBLKT" : "NSLMBLKS", 8)) return h;
    if (h) munmap((void*) h, *len);
    return NULL;
}

int ns_open_bank(NsModel* m, const char* dir, char* err, int errlen) {
    static const char* kp[3] = {"gate_proj", "up_proj", "down_proj"};
    memset(m, 0, sizeof *m);
    int cap = 0;
    for (int l = 0; l < 256; ++l)
        for (int p = 0; p < 3; ++p)
            for (int tcq = 0; tcq < 2; ++tcq) {
                char path[2048];
                snprintf(path, sizeof path, "%s/L%d_%s.%s", dir, l, kp[p], tcq ? "blkt" : "blks");
                size_t len;
                const MoeBlkHeader* h = map_bank(path, tcq, &len);
                if (!h && !len) continue;
                const size_t E = h ? h->n_experts : 0, R = h ? h->rows : 0;
                const int L = h ? (int) h->n_seeds : 0, K = h ? (int) h->n_exp : 0, T = h ? (int) h->refit : 0;
                const size_t nb = h ? (tcq ? (h->cols + (uint32_t) T - 1) / (uint32_t) (T ? T : 1) : R * ((h->cols + 11) / 12)) : 0;
                const size_t nw = tcq && h && T && K > 0 && K < L && L <= NSLM_TQ_MAXL ? nslm_tq_row_words((int) h->cols, T, L, K) : 0;
                const size_t want = tcq ? sizeof *h + 8 * E + (4 * nw + 2 * nb) * R * E : sizeof *h + 12 * E + 5 * nb * E;
                if (!h || (tcq && !nw) || len != want) {
                    snprintf(err, (size_t) errlen, "%s: not a %s bank", path, tcq ? "trellis" : "S3 seed");
                    free(m->t);
                    memset(m, 0, sizeof *m);
                    return -1;
                }
                if (m->n == cap) m->t = (NsTensor*) realloc(m->t, sizeof(NsTensor) * (size_t) (cap = cap ? 2 * cap : 64));
                NsTensor* t = &m->t[m->n++];
                memset(t, 0, sizeof *t);
                snprintf(t->name, sizeof t->name, "language_model.model.layers.%d.mlp.switch_mlp.%s.weight", l, kp[p]);
                t->slices = (int) E; t->rows = (int) R; t->cols = (int) h->cols;
                const uint8_t* b = (const uint8_t*) (h + 1);
                if (tcq) {   // rel[E], wrel[E], words [E][rows][nw], BF16 scales [E][rows][nb]
                    t->enc = NS_TCQ; t->group = L | K << 8 | T << 16;
                    t->s[0] = (NsStream) {b + 8 * E, 4 * nw * R * E, 0, 0};
                    t->s[1] = (NsStream) {b + 8 * E + 4 * nw * R * E, 2 * nb * R * E, 0, 0};
                } else {     // bias[E], rel[E], wrel[E], seeds, coefficients, codes
                    t->enc = NS_SEEDS3;
                    const uint8_t* at[4] = {b + 12 * E, b + 12 * E + 2 * nb * E, b, b + 12 * E + 4 * nb * E};
                    for (int s = 0; s < 4; ++s) { t->s[s].p = at[s]; t->s[s].len = ns_stream_len(NS_SEEDS3, t->slices, t->rows, t->cols, s); }
                }
            }
    if (!m->n) { snprintf(err, (size_t) errlen, "%s: no seed or trellis banks (L*_*.blks, L*_*.blkt)", dir); return 1; }
    qsort(m->t, (size_t) m->n, sizeof(NsTensor), bank_cmp);
    return 0;
}

int ns_add_banks(NsModel* m, const char* dir, char* err, int errlen) {
    NsModel b;
    const int rc = ns_open_bank(&b, dir, err, errlen);
    if (rc) return rc > 0 ? 0 : -1;
    m->t = (NsTensor*) realloc(m->t, sizeof(NsTensor) * (size_t) (m->n + b.n));
    NsModel v = *m;   // the folder's own tensors, still sorted: bank names never repeat
    for (int i = 0; i < b.n; ++i) {
        const NsTensor* t = ns_find(&v, b.t[i].name);
        m->t[t ? (int) (t - m->t) : m->n++] = b.t[i];   // a bank replaces the folder's tensor of that name
    }
    qsort(m->t, (size_t) m->n, sizeof(NsTensor), bank_cmp);
    free(b.t);
    return 0;
}

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

int nslm_copy_loader(const char* loader, const char* dir) {
    const char* sl = strrchr(loader, '/');
    char from[2048], to[2048];
    snprintf(from, sizeof from, "%.*snanoseedlm_seeds.py", sl ? (int) (sl - loader + 1) : 0, loader);
    snprintf(to, sizeof to, "%s/%s", dir, sl ? sl + 1 : loader);
    if (copy_file(loader, to)) return -1;
    snprintf(to, sizeof to, "%s/nanoseedlm_seeds.py", dir);
    return copy_file(from, to);
}

void nslm_copy_model_files(const char* from, const char* to) {
    static const char* files[] = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                                  "generation_config.json", "preprocessor_config.json", "processor_config.json",
                                  "video_preprocessor_config.json", "vocab.json", "merges.txt"};
    char a[2048], b[2048];
    for (size_t i = 0; i < sizeof files / sizeof files[0]; ++i) {
        snprintf(a, sizeof a, "%s/%s", from, files[i]);
        snprintf(b, sizeof b, "%s/%s", to, files[i]);
        copy_file(a, b);
    }
}
