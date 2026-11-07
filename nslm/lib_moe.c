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

// The folder's tensor that bank tensor b replaces (NULL: b is added). A bank of layer n is the MTP head's (the packer's
// rule) when the folder has routers for layer n - 1 and the head but none for layer n: b becomes mtp.layers.0.mlp.switch_mlp.*
static const NsTensor* bank_target(const NsModel* v, NsTensor* b) {
    const NsTensor* t = ns_find(v, b->name);
    int l = -1, at = 0;
    if (t || sscanf(b->name, "language_model.model.layers.%d.mlp.switch_mlp.%n", &l, &at) != 1 || !at || l < 1) return t;
    char r0[96], r1[96], head[sizeof b->name];
    snprintf(r0, sizeof r0, "language_model.model.layers.%d.mlp.gate.weight", l - 1);
    snprintf(r1, sizeof r1, "language_model.model.layers.%d.mlp.gate.weight", l);
    if (!ns_find(v, r0) || ns_find(v, r1) || !ns_find(v, "mtp.layers.0.mlp.gate.weight")) return NULL;
    snprintf(head, sizeof head, "mtp.layers.0.mlp.switch_mlp.%s", b->name + at);
    memcpy(b->name, head, sizeof head);
    return ns_find(v, head);
}

int ns_add_banks(NsModel* m, const char* dir, char* err, int errlen) {
    NsModel b;
    const int rc = ns_open_bank(&b, dir, err, errlen);
    if (rc) return rc > 0 ? 0 : -1;
    m->t = (NsTensor*) realloc(m->t, sizeof(NsTensor) * (size_t) (m->n + b.n));
    NsModel v = *m;   // the folder's own tensors, still sorted: bank names never repeat
    for (int i = 0; i < b.n; ++i) {
        const NsTensor* t = bank_target(&v, &b.t[i]);
        m->t[t ? (int) (t - m->t) : m->n++] = b.t[i];   // a bank replaces the folder's tensor of that name
    }
    qsort(m->t, (size_t) m->n, sizeof(NsTensor), bank_cmp);
    free(b.t);
    return 0;
}

// ---- plan tcq-safetensors: a folder's .blkt banks as safetensors shards ----

typedef struct { const uint8_t* p[12]; uint64_t len[12]; } TcqFill;   // a layer's shard: 3 projections x 4 entries
static int tcq_fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) {
    const TcqFill* f = (const TcqFill*) ctx;
    if (s || len != f->len[t]) return -1;
    memcpy(dst, f->p[t], len);
    return 0;
}
static char* read_text(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    const long m = f && !fseek(f, 0, SEEK_END) ? ftell(f) : -1;
    char* s = m >= 0 && !fseek(f, 0, SEEK_SET) ? (char*) malloc((size_t) m + 1) : NULL;
    if (s && fread(s, 1, (size_t) m, f) != (size_t) m) { free(s); s = NULL; }
    if (f) fclose(f);
    if (s) { s[m] = 0; *n = (size_t) m; }
    return s;
}
static int same_streams(const NsTensor* a, const NsTensor* b) {
    int ok = a && b && a->enc == b->enc && a->group == b->group && a->slices == b->slices && a->rows == b->rows && a->cols == b->cols;
    for (int s = 0; ok && s < 4; ++s) ok = a->s[s].len == b->s[s].len && (!a->s[s].len || !memcmp(a->s[s].p, b->s[s].p, a->s[s].len));
    return ok;
}

static void unmap_banks(NsModel* b) {   // ns_open_bank keeps its banks mapped for the process
    for (int i = 0; i < b->n; ++i) {
        const NsTensor* t = &b->t[i];
        const uint8_t* h = t->s[0].p - 8 * (uint64_t) t->slices - sizeof(MoeBlkHeader);
        munmap((void*) h, sizeof(MoeBlkHeader) + 8 * (uint64_t) t->slices + t->s[0].len + t->s[1].len);
    }
    free(b->t);
    memset(b, 0, sizeof *b);
}

int nslm_tcq_to_st(const char* dir, char* err, int errlen) {
    static const char* kp[3] = {"gate_proj", "up_proj", "down_proj"};
    static const char* suf[4] = {".tcq_words", ".tcq_scales", ".tcq_params", ".tcq_err"};
    NsModel v, b;
    if (ns_open(&v, dir, err, errlen)) return -1;
    int rc = ns_open_bank(&b, dir, err, errlen);
    if (rc) { ns_close(&v); return rc > 0 ? 1 : -1; }
    int maxl = -1, done = 0;
    int* lay = (int*) malloc(sizeof(int) * (size_t) b.n), *proj = (int*) malloc(sizeof(int) * (size_t) b.n);
    for (int i = 0; !rc && i < b.n; ++i) {   // each bank's layer and projection; the head's gets its mtp.* name
        int l = -1, at = 0;
        sscanf(b.t[i].name, "language_model.model.layers.%d.mlp.switch_mlp.%n", &l, &at);
        proj[i] = -1;
        for (int p = 0; at && p < 3; ++p) if (!strncmp(b.t[i].name + at, kp[p], strlen(kp[p]))) proj[i] = p;
        lay[i] = l;
        if (b.t[i].enc != NS_TCQ || l < 0 || proj[i] < 0) { rc = snprintf(err, (size_t) errlen, "%s: not a trellis bank (.blkt)", b.t[i].name); break; }
        maxl = l > maxl ? l : maxl;
        const NsTensor* t = bank_target(&v, &b.t[i]);
        if (t && !same_streams(t, &b.t[i]))   // a folder tensor of this name: ns_add_banks replaces it, a shard cannot
            rc = snprintf(err, (size_t) errlen, "%s: the folder has this tensor; remove it before the conversion", b.t[i].name);
        done += t != NULL;   // converted before: the banks only go
    }
    if (!rc && done && done != b.n) rc = snprintf(err, (size_t) errlen, "%s: %d of %d banks are in the shards; restore the index", dir, done, b.n);
    int nl = 0;   // the layers that have banks, in order: shard k of nl
    for (int l = 0; !rc && !done && l <= maxl; ++l) {
        int any = 0;
        for (int i = 0; i < b.n; ++i) any |= lay[i] == l;
        nl += any;
    }
    char ipath[2048];
    snprintf(ipath, sizeof ipath, "%s/model.safetensors.index.json", dir);
    size_t itn = 0;
    char* itext = !rc && !done ? read_text(ipath, &itn) : NULL;
    Json* idx = itext ? json_parse(itext, itn, err, errlen) : NULL;
    free(itext);
    Json* wm = json_get(idx, "weight_map");   // no index: ns_open lists the folder's model*.safetensors
    if (!rc && !done && itn && !wm) rc = snprintf(err, (size_t) errlen, "%s: no weight_map", ipath);
    uint64_t added = 0;
    for (int l = 0, k = 0; !rc && !done && l <= maxl; ++l) {
        int at[3] = {-1, -1, -1}, n = 0;
        for (int i = 0; i < b.n; ++i) if (lay[i] == l) at[proj[i]] = i;
        if (at[0] < 0 && at[1] < 0 && at[2] < 0) continue;
        ++k;
        char shard[64], path[2048], tmp[2100], names[12][176];
        snprintf(shard, sizeof shard, "model-tcq-%05d-of-%05d.safetensors", k, nl);
        snprintf(path, sizeof path, "%s/%s", dir, shard);
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        NsSpec sp[12];
        NsRaw raw[12];
        TcqFill fc;
        int32_t prm[3][4];
        Buf meta = {0};
        buf_puts(&meta, "\"nanoseedlm\": \"trellis-coded routed experts (nslm/trellis.h): NAME.tcq_words, NAME.tcq_scales, "
                        "NAME.tcq_params (L, K, T, cols), NAME.tcq_err (relative errors)\"");
        for (int p = 0; p < 3; ++p) {
            if (at[p] < 0) continue;
            const NsTensor* t = &b.t[at[p]];
            const uint64_t E = (uint64_t) t->slices, R = (uint64_t) t->rows;
            const MoeBlkHeader* h = (const MoeBlkHeader*) (t->s[0].p - 8 * E) - 1;   // map_bank: header, rel[E], wrel[E], words
            const int L = t->group & 255, K = t->group >> 8 & 255, T = t->group >> 16;
            const size_t base = strlen(t->name) - 7;   // without .weight
            prm[p][0] = L; prm[p][1] = K; prm[p][2] = T; prm[p][3] = t->cols;
            const uint64_t nw = nslm_tq_row_words(t->cols, T, L, K), nb = (uint64_t) ((t->cols + T - 1) / T);
            const struct { const char* dt; int nd; int64_t sh[3]; const uint8_t* src; uint64_t len; } e[4] = {
                {"U32", 3, {(int64_t) E, (int64_t) R, (int64_t) nw}, t->s[0].p, t->s[0].len},
                {"BF16", 3, {(int64_t) E, (int64_t) R, (int64_t) nb}, t->s[1].p, t->s[1].len},
                {"I32", 1, {4}, (const uint8_t*) prm[p], sizeof prm[p]},
                {"F32", 2, {2, (int64_t) E}, (const uint8_t*) (h + 1), 8 * E}};
            for (int q = 0; q < 4; ++q, ++n) {
                snprintf(names[n], sizeof names[n], "%.*s%s", (int) base, t->name, suf[q]);
                sp[n] = (NsSpec) {names[n], NS_RAW, 1, 1, 1};
                raw[n] = (NsRaw) {e[q].dt, e[q].nd, {0}, e[q].len};
                memcpy(raw[n].shape, e[q].sh, sizeof e[q].sh);
                fc.p[n] = e[q].src; fc.len[n] = e[q].len;
                added += e[q].len;
                if (wm) json_set(wm, names[n], json_str(shard));
            }
            char m[512];   // the encoder's own numbers (the header's)
            snprintf(m, sizeof m, ", \"%.*s.tcq_n0\": \"%.9g\", \"%.*s.tcq_seconds\": \"%.17g\"", (int) base, t->name, (double) h->n0,
                     (int) base, t->name, h->seconds);
            buf_puts(&meta, m);
        }
        char why[256] = "";
        if (ns_write_shard(tmp, sp, raw, n, meta.p, tcq_fill, &fc, why, sizeof why) || rename(tmp, path))
            rc = snprintf(err, (size_t) errlen, "%s: cannot write (%s)", path, why[0] ? why : strerror(errno));
        free(meta.p);
    }
    if (!rc && !done && idx) {   // the index lists the new entries, and its total grows by their bytes
        Json* ts = json_get(json_get(idx, "metadata"), "total_size");
        if (ts && ts->t == J_INT) ts->i += (int64_t) added;
        Buf o = {0};
        json_dump_indent(&o, idx, 0);
        buf_puts(&o, "\n");
        char tmp[2100];
        snprintf(tmp, sizeof tmp, "%s.tmp", ipath);
        FILE* f = fopen(tmp, "wb");
        if (!f || fwrite(o.p, 1, o.n, f) != o.n || fclose(f) || rename(tmp, ipath)) {
            if (f) fclose(f);
            rc = snprintf(err, (size_t) errlen, "%s: cannot write", ipath);
        }
        free(o.p);
    }
    if (!rc) {   // the folder read again: each bank's streams in a shard, byte for byte; then the banks go
        NsModel c;
        const int opened = !ns_open(&c, dir, err, errlen);
        if (!opened) rc = -1;
        for (int i = 0; !rc && i < b.n; ++i)
            if (!same_streams(ns_find(&c, b.t[i].name), &b.t[i])) rc = snprintf(err, (size_t) errlen, "%s: the shard differs from the bank", b.t[i].name);
        if (opened) ns_close(&c);
        const int nb = b.n;
        if (!rc) unmap_banks(&b);   // first: Windows does not delete a mapped file
        for (int i = 0; !rc && i < nb; ++i) {
            char path[2048];
            snprintf(path, sizeof path, "%s/L%d_%s.blkt", dir, lay[i], kp[proj[i]]);
            if (unlink(path)) rc = snprintf(err, (size_t) errlen, "%s: %s", path, strerror(errno));
        }
    }
    json_free(idx);
    free(lay); free(proj);
    unmap_banks(&b);
    ns_close(&v);
    return rc ? -1 : 0;
}

int nslm_copy_file(const char* from, const char* to) {
    FILE* a = fopen(from, "rb");
    if (!a) return -1;
    FILE* b = fopen(to, "wb");
    if (!b) { fclose(a); return -1; }
    char buf[1 << 16];
    for (size_t n; (n = fread(buf, 1, sizeof buf, a)) > 0;) fwrite(buf, 1, n, b);
    fclose(a);
    return fclose(b);
}

int nslm_copy_beside(const char* loader, const char* name, const char* dir) {
    const char* sl = strrchr(loader, '/');
    char from[2048], to[2048];
    snprintf(from, sizeof from, "%.*s%s", sl ? (int) (sl - loader + 1) : 0, loader, name);
    snprintf(to, sizeof to, "%s/%s", dir, name);
    return nslm_copy_file(from, to);
}

int nslm_copy_loader(const char* loader, const char* dir) {
    const char* sl = strrchr(loader, '/');
    char to[2048];
    snprintf(to, sizeof to, "%s/%s", dir, sl ? sl + 1 : loader);
    return nslm_copy_file(loader, to) || nslm_copy_beside(loader, "nanoseedlm_seeds.py", dir) ? -1 : 0;
}

void nslm_copy_model_files(const char* from, const char* to) {
    static const char* files[] = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                                  "generation_config.json", "preprocessor_config.json", "processor_config.json",
                                  "video_preprocessor_config.json", "vocab.json", "merges.txt", "LICENSE"};
    char a[2048], b[2048];
    for (size_t i = 0; i < sizeof files / sizeof files[0]; ++i) {
        snprintf(a, sizeof a, "%s/%s", from, files[i]);
        snprintf(b, sizeof b, "%s/%s", to, files[i]);
        nslm_copy_file(a, b);
    }
}
