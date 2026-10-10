// nslm/lib_model_st.c - see model_st.h.  C99 + POSIX.
#include "model_st.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "json.h"

static const char* SUF[6][4] = {
    [NS_BF16] = {"", NULL, NULL, NULL},
    [NS_SEED4] = {".seeds", ".nibbles", ".exp_bias", NULL},
    [NS_Q8] = {".weight", ".scales", ".biases", NULL},
    [NS_Q4] = {".weight", ".scales", ".biases", NULL},
    [NS_SEED4P4] = {".seeds", ".coefs", ".exp_bias", ".codes"},
    [NS_SEED6P8] = {".seeds", ".coefs", ".exp_bias", ".codes"},
};
static const char* DT[6][4] = {
    [NS_BF16] = {"BF16", NULL, NULL, NULL},
    [NS_SEED4] = {"U16", "U16", "I32", NULL},
    [NS_Q8] = {"U32", "BF16", "BF16", NULL},
    [NS_Q4] = {"U32", "BF16", "BF16", NULL},
    [NS_SEED4P4] = {"U16", "U16", "I32", "U8"},
    [NS_SEED6P8] = {"U16", "U32", "I32", "U8"},
};

uint64_t ns_stream_len(int enc, int slices, int rows, int cols, int s) {
    const uint64_t n = (uint64_t) slices * (uint64_t) rows * (uint64_t) cols;
    switch (enc) {
    case NS_BF16: return s == 0 ? 2 * n : 0;
    case NS_Q8: return s == 0 ? n : s < 3 ? 2 * (n / 64) : 0;
    case NS_Q4: return s == 0 ? n / 2 : s < 3 ? 2 * (n / 64) : 0;
    case NS_SEED4: return s < 2 ? 2 * (n / 8) : s == 2 ? 4 * (uint64_t) slices : 0;
    case NS_SEED4P4: return s < 2 ? 2 * (n / 8) : s == 2 ? 4 * (uint64_t) slices : n / 16;
    case NS_SEED6P8: return s == 0 ? 2 * (n / 8) : s == 1 ? 4 * (n / 8) : s == 2 ? 4 * (uint64_t) slices : n / 16;
    default: return 0;
    }
}
static uint64_t pad(uint64_t n) { return (n + NS_PAGE - 1) & ~(uint64_t) (NS_PAGE - 1); }
static int ends(const char* s, const char* suf) {
    const size_t a = strlen(s), b = strlen(suf);
    return a >= b && !strcmp(s + a - b, suf);
}

// ---- reader ----

typedef struct {
    const char* name;
    int shard;
    const StEntry* e;
} Ref;
static int ref_cmp(const void* a, const void* b) { return strcmp(((const Ref*) a)->name, ((const Ref*) b)->name); }
static int ten_cmp(const void* a, const void* b) { return strcmp(((const NsTensor*) a)->name, ((const NsTensor*) b)->name); }
static const Ref* lookup(const Ref* r, int n, const char* name) {
    Ref k = {name, 0, NULL};
    return (const Ref*) bsearch(&k, r, (size_t) n, sizeof(Ref), ref_cmp);
}
static int name_cmp(const void* a, const void* b) { return strcmp(*(char* const*) a, *(char* const*) b); }

// Shard file names: from the index, else model*.safetensors in the folder.
static int shard_names(const char* dir, char*** out, char* err, int errlen) {
    char path[2048];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    int n = 0;
    char** v = NULL;
    FILE* f = fopen(path, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        const long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        char* s = (char*) malloc((size_t) len + 1);
        const size_t got = fread(s, 1, (size_t) len, f);
        fclose(f);
        Json* j = json_parse(s, got, err, errlen);
        free(s);
        const Json* wm = json_get(j, "weight_map");
        if (!wm || wm->t != J_OBJ) { json_free(j); snprintf(err, (size_t) errlen, "%s: no weight_map", path); return -1; }
        for (int i = 0; i < wm->n; ++i) {
            if (wm->v[i]->t != J_STR || strchr(wm->v[i]->s, '/')) continue;
            int seen = 0;
            for (int k = 0; k < n && !seen; ++k) seen = !strcmp(v[k], wm->v[i]->s);
            if (seen) continue;
            v = (char**) realloc(v, sizeof(char*) * (size_t) (n + 1));
            v[n++] = strdup(wm->v[i]->s);
        }
        json_free(j);
    } else {
        DIR* d = opendir(dir);
        if (!d) { snprintf(err, (size_t) errlen, "%s: %s", dir, strerror(errno)); return -1; }
        for (struct dirent* de; (de = readdir(d));)
            if (!strncmp(de->d_name, "model", 5) && ends(de->d_name, ".safetensors")) {
                v = (char**) realloc(v, sizeof(char*) * (size_t) (n + 1));
                v[n++] = strdup(de->d_name);
            }
        closedir(d);
    }
    if (!n) { snprintf(err, (size_t) errlen, "%s: no safetensors", dir); free(v); return -1; }
    qsort(v, (size_t) n, sizeof(char*), name_cmp);
    *out = v;
    return n;
}

static int dims(const StEntry* e, int want_last, int64_t* s, int64_t* r, int64_t* c) {
    if (e->ndim == 3) { *s = e->shape[0]; *r = e->shape[1]; *c = e->shape[2]; }
    else if (e->ndim == 2) { *s = 1; *r = e->shape[0]; *c = e->shape[1]; }
    else if (e->ndim == 1 && !want_last) { *s = 1; *r = 1; *c = e->shape[0]; }
    else return -1;
    return 0;
}

static int add(NsModel* m, const NsModel* src, const Ref* refs, int nr, const char* base, const char* name, int enc,
               int64_t S, int64_t R, int64_t C, char* err, int errlen) {
    NsTensor t;
    memset(&t, 0, sizeof t);
    if (strlen(name) >= sizeof t.name) { snprintf(err, (size_t) errlen, "%s: name too long", name); return -1; }
    snprintf(t.name, sizeof t.name, "%s", name);
    t.enc = enc; t.slices = (int) S; t.rows = (int) R; t.cols = (int) C;
    for (int s = 0; s < 4 && SUF[enc][s]; ++s) {
        char nm[160];
        const int bare = !s && !strcmp(name, base);   // an affine tensor not named *.weight: its codes are named X
        snprintf(nm, sizeof nm, "%s%s", base, bare ? "" : SUF[enc][s]);
        const Ref* r = lookup(refs, nr, nm);
        const uint64_t want = ns_stream_len(enc, t.slices, t.rows, t.cols, s);
        if (!r || strcmp(r->e->dtype, DT[enc][s]) || r->e->end - r->e->begin != want) {
            snprintf(err, (size_t) errlen, "%s: missing, or not %s with %llu bytes", nm, DT[enc][s], (unsigned long long) want);
            return -1;
        }
        const StFile* f = &src->sh[r->shard];
        t.s[s] = (NsStream) {f->map + f->data + r->e->begin, want, f->data + r->e->begin, r->shard};
    }
    m->t = (NsTensor*) realloc(m->t, sizeof(NsTensor) * (size_t) (m->n + 1));
    m->t[m->n++] = t;
    return 0;
}

int ns_open(NsModel* m, const char* dir, char* err, int errlen) {
    memset(m, 0, sizeof *m);
    char** names = NULL;
    const int ns = shard_names(dir, &names, err, errlen);
    if (ns < 0) return -1;
    m->sh = (StFile*) calloc((size_t) ns, sizeof(StFile));
    int total = 0, rc = -1;
    for (int i = 0; i < ns; ++i) {
        char path[2048];
        snprintf(path, sizeof path, "%s/%s", dir, names[i]);
        if (st_open_file(&m->sh[i], path, err, errlen)) goto done;
        m->nsh = i + 1;
        for (int k = 0; k < m->sh[i].n; ++k) {
            const StEntry* e = &m->sh[i].e[k];
            if (e->begin > e->end || m->sh[i].data + e->end > m->sh[i].size) {
                snprintf(err, (size_t) errlen, "%s: %s is out of bounds", path, e->name);
                goto done;
            }
        }
        total += m->sh[i].n;
    }
    Ref* refs = (Ref*) malloc(sizeof(Ref) * (size_t) total);
    int nr = 0;
    for (int i = 0; i < ns; ++i)
        for (int k = 0; k < m->sh[i].n; ++k) refs[nr++] = (Ref) {m->sh[i].e[k].name, i, &m->sh[i].e[k]};
    qsort(refs, (size_t) nr, sizeof(Ref), ref_cmp);
    for (int i = 1; i < nr; ++i)
        if (!strcmp(refs[i].name, refs[i - 1].name)) { snprintf(err, (size_t) errlen, "%s appears twice", refs[i].name); free(refs); goto done; }
    for (int i = 0; i < nr; ++i) {
        const char* nm = refs[i].name;
        const StEntry* e = refs[i].e;
        char base[160], logical[176];
        const size_t L = strlen(nm);
        int64_t S, R, C;
        if (ends(nm, ".seeds") || ends(nm, ".scales")) {
            const int seeds = ends(nm, ".seeds");
            snprintf(base, sizeof base, "%.*s", (int) (L - (seeds ? 6 : 7)), nm);
            snprintf(logical, sizeof logical, "%s.weight", base);
            if (dims(e, 1, &S, &R, &C)) { snprintf(err, (size_t) errlen, "%s: bad shape", nm); free(refs); goto done; }
            int enc;
            if (seeds) {
                char k[176];
                snprintf(k, sizeof k, "%s.codes", base);
                snprintf(k, sizeof k, "%s.coefs", base);
                const Ref* cf = lookup(refs, nr, k);
                snprintf(k, sizeof k, "%s.codes", base);
                enc = !lookup(refs, nr, k) ? NS_SEED4 : cf && !strcmp(cf->e->dtype, "U32") ? NS_SEED6P8 : NS_SEED4P4;
                C *= 8;
            } else {
                char k[176];
                snprintf(k, sizeof k, "%s.weight", base);
                const Ref* w = lookup(refs, nr, k);
                if (!w && (w = lookup(refs, nr, base))) snprintf(logical, sizeof logical, "%s", base);   // codes named X
                C *= 64;
                const int64_t bits = w && w->e->ndim >= 1 ? w->e->shape[w->e->ndim - 1] * 32 / C : 0;
                if (bits != 4 && bits != 8) { snprintf(err, (size_t) errlen, "%s: not affine Q4 or Q8", base); free(refs); goto done; }
                enc = bits == 8 ? NS_Q8 : NS_Q4;
            }
            if (add(m, m, refs, nr, base, logical, enc, S, R, C, err, errlen)) { free(refs); goto done; }
        } else if (!strcmp(e->dtype, "BF16")) {
            if (ends(nm, ".biases")) {   // a quantized tensor's biases
                char k[176];
                snprintf(k, sizeof k, "%.*s.scales", (int) (L - 7), nm);
                if (lookup(refs, nr, k)) continue;
            }
            if (dims(e, 0, &S, &R, &C)) continue;   // scalars and >3-D tensors are not model weights here
            if (add(m, m, refs, nr, nm, nm, NS_BF16, S, R, C, err, errlen)) { free(refs); goto done; }
        }
    }
    free(refs);
    qsort(m->t, (size_t) m->n, sizeof(NsTensor), ten_cmp);
    rc = 0;
done:
    for (int i = 0; i < ns; ++i) free(names[i]);
    free(names);
    if (rc) ns_close(m);
    return rc;
}

void ns_close(NsModel* m) {
    for (int i = 0; i < m->nsh; ++i) st_close_file(&m->sh[i]);
    free(m->sh);
    free(m->t);
    memset(m, 0, sizeof *m);
}
const NsTensor* ns_find(const NsModel* m, const char* name) {
    NsTensor k;
    snprintf(k.name, sizeof k.name, "%s", name);
    return m->n ? (const NsTensor*) bsearch(&k, m->t, (size_t) m->n, sizeof(NsTensor), ten_cmp) : NULL;
}
int ns_mappable(const NsModel* m, const NsStream* s) {
    return s->off % NS_PAGE == 0 && s->off + pad(s->len) <= pad(m->sh[s->shard].size);
}

// ---- writer ----

typedef struct {
    char name[176];
    const char* dtype;
    int64_t shape[3];
    int ndim;
    uint64_t len, off;   // off: relative to the data section
    int tensor, stream, shard;
} Ent;

static void ent_shape(Ent* e, const NsSpec* t, int s) {
    const int seed = t->enc == NS_SEED4 || t->enc == NS_SEED4P4 || t->enc == NS_SEED6P8;
    const int lead = t->slices > 1 || seed;
    int64_t last = t->cols;
    if (t->enc == NS_Q8 || t->enc == NS_Q4) last = s == 0 ? (int64_t) t->cols * (t->enc == NS_Q8 ? 8 : 4) / 32 : t->cols / 64;
    if (seed) last = s == 3 ? t->cols / 16 : t->cols / 8;
    e->ndim = 0;
    if (seed && s == 2) { e->shape[0] = t->slices; e->ndim = 1; return; }
    if (lead) e->shape[e->ndim++] = t->slices;
    if (t->enc != NS_BF16 || t->rows > 1 || t->slices > 1) e->shape[e->ndim++] = t->rows;
    e->shape[e->ndim++] = last;
}

static int write_index(const char* dir, const Ent* ents, int ne, char shards[][64], uint64_t total, char* err, int errlen) {
    char path[2048];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) { snprintf(err, (size_t) errlen, "%s: %s", path, strerror(errno)); return -1; }
    fprintf(f, "{\n  \"metadata\": {\n    \"total_size\": %llu\n  },\n  \"weight_map\": {", (unsigned long long) total);
    for (int i = 0; i < ne; ++i) fprintf(f, "%s\n    \"%s\": \"%s\"", i ? "," : "", ents[i].name, shards[ents[i].shard]);
    fprintf(f, "\n  }\n}\n");
    return fclose(f) ? -1 : 0;
}

int ns_write(const char* dir, const NsSpec* t, int n, uint64_t shard_bytes, const char* meta, NsFill fill, void* ctx,
             char* err, int errlen) {
    Ent* ents = (Ent*) calloc((size_t) n * 4, sizeof(Ent));
    int ne = 0, nsh = 0;
    uint64_t cur = 0, total = 0;
    for (int i = 0; i < n; ++i) {
        if ((t[i].enc == NS_Q8 || t[i].enc == NS_Q4) && t[i].cols % 64) {   // affine g64: whole groups per row
            snprintf(err, (size_t) errlen, "%s: %d columns, not a multiple of the Q8 / Q4 group of 64", t[i].name, t[i].cols);
            free(ents);
            return -1;
        }
        uint64_t tb = 0;
        for (int s = 0; s < 4; ++s) tb += ns_stream_len(t[i].enc, t[i].slices, t[i].rows, t[i].cols, s);
        if (nsh == 0 || (cur > 0 && cur + tb > shard_bytes)) { ++nsh; cur = 0; }
        cur += tb;
        total += tb;
        for (int s = 0; s < 4; ++s) {
            const uint64_t len = ns_stream_len(t[i].enc, t[i].slices, t[i].rows, t[i].cols, s);
            if (!len) continue;
            Ent* e = &ents[ne++];
            const size_t L = strlen(t[i].name);
            if (t[i].enc == NS_BF16) snprintf(e->name, sizeof e->name, "%s", t[i].name);
            else if (L > 7 && ends(t[i].name, ".weight")) snprintf(e->name, sizeof e->name, "%.*s%s", (int) (L - 7), t[i].name, SUF[t[i].enc][s]);
            else if (t[i].enc == NS_Q8 || t[i].enc == NS_Q4)   // a name without .weight (MLA projections): X, X.scales, X.biases
                snprintf(e->name, sizeof e->name, "%s%s", t[i].name, s ? SUF[t[i].enc][s] : "");
            else { snprintf(err, (size_t) errlen, "%s: a seed-encoded tensor must be named *.weight", t[i].name); free(ents); return -1; }
            e->dtype = DT[t[i].enc][s];
            ent_shape(e, &t[i], s);
            e->len = len;
            e->tensor = i;
            e->stream = s;
            e->shard = nsh - 1;
        }
    }
    char(*shards)[64] = calloc((size_t) nsh, 64);
    uint64_t* data = (uint64_t*) calloc((size_t) nsh, sizeof(uint64_t));
    int* fds = (int*) calloc((size_t) nsh, sizeof(int));
    int rc = -1;
    for (int k = 0; k < nsh; ++k) {
        snprintf(shards[k], 64, "model-%05d-of-%05d.safetensors", k + 1, nsh);
        // layout: page-multiple entries first, then the rest
        uint64_t off = 0;
        for (int pass = 0; pass < 2; ++pass)
            for (int i = 0; i < ne; ++i)
                if (ents[i].shard == k && (ents[i].len % NS_PAGE == 0) == (pass == 0)) { ents[i].off = off; off += ents[i].len; }
        Buf h = {0};
        buf_puts(&h, "{\"__metadata__\": {\"format\": \"mlx\"");
        if (meta && *meta) { buf_puts(&h, ", "); buf_puts(&h, meta); }
        buf_puts(&h, "}");
        for (int i = 0; i < ne; ++i) {
            if (ents[i].shard != k) continue;
            char tmp[512];
            int w = snprintf(tmp, sizeof tmp, ", \"%s\": {\"dtype\": \"%s\", \"shape\": [", ents[i].name, ents[i].dtype);
            for (int d = 0; d < ents[i].ndim; ++d) w += snprintf(tmp + w, sizeof tmp - (size_t) w, "%s%lld", d ? ", " : "", (long long) ents[i].shape[d]);
            snprintf(tmp + w, sizeof tmp - (size_t) w, "], \"data_offsets\": [%llu, %llu]}", (unsigned long long) ents[i].off,
                     (unsigned long long) (ents[i].off + ents[i].len));
            buf_puts(&h, tmp);
        }
        buf_puts(&h, "}");
        while ((8 + h.n) % NS_PAGE) buf_puts(&h, " ");
        data[k] = 8 + h.n;
        char path[2048];
        snprintf(path, sizeof path, "%s/%s", dir, shards[k]);
        fds[k] = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0644);
        const uint64_t hn = h.n;
        if (fds[k] < 0 || pwrite(fds[k], &hn, 8, 0) != 8 || pwrite(fds[k], h.p, h.n, 8) != (ssize_t) h.n ||
            ftruncate(fds[k], (off_t) (data[k] + off))) {
            snprintf(err, (size_t) errlen, "%s: %s", path, strerror(errno));
            free(h.p);
            goto done;
        }
        free(h.p);
    }
    for (int i = 0; i < ne; ++i) {   // ents are in tensor and stream order
        uint8_t* b = (uint8_t*) malloc(ents[i].len);
        if (!b || fill(ctx, ents[i].tensor, ents[i].stream, b, ents[i].len)) {
            snprintf(err, (size_t) errlen, "%s: cannot fill", ents[i].name);
            free(b);
            goto done;
        }
        uint64_t done_n = 0;
        while (done_n < ents[i].len) {
            const ssize_t w = pwrite(fds[ents[i].shard], b + done_n, ents[i].len - done_n, (off_t) (data[ents[i].shard] + ents[i].off + done_n));
            if (w <= 0) { snprintf(err, (size_t) errlen, "%s: write failed", ents[i].name); free(b); goto done; }
            done_n += (uint64_t) w;
        }
        free(b);
    }
    rc = write_index(dir, ents, ne, shards, total, err, errlen);
done:
    for (int k = 0; k < nsh; ++k) if (fds[k] > 0) close(fds[k]);
    free(fds); free(data); free(shards); free(ents);
    return rc;
}
