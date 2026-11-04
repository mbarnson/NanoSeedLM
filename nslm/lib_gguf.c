// nslm/lib_gguf.c - see gguf.h.  C99 + POSIX mmap.
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gguf.h"

typedef struct { const uint8_t* p; size_t n, i; int bad; } Rd;

static uint64_t rd(Rd* r, int w) {   // little-endian unsigned of w bytes
    uint64_t v = 0;
    if (r->bad || r->i + (size_t) w > r->n) { r->bad = 1; return 0; }
    for (int k = 0; k < w; ++k) v |= (uint64_t) r->p[r->i + (size_t) k] << (8 * k);
    r->i += (size_t) w;
    return v;
}
static void skip(Rd* r, uint64_t n) { if (r->bad || n > r->n - r->i) r->bad = 1; else r->i += (size_t) n; }

static const int kWidth[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};   // GGUF value types; 8 string, 9 array
static void skip_value(Rd* r, uint32_t type, int depth) {
    if (type == 8) { skip(r, rd(r, 8)); return; }
    if (type == 9 && depth < 4) {
        const uint32_t et = (uint32_t) rd(r, 4);
        const uint64_t n = rd(r, 8);
        if (n > r->n) { r->bad = 1; return; }
        if (et != 8 && et != 9 && et < 13) { skip(r, n * (uint64_t) kWidth[et]); return; }
        for (uint64_t k = 0; k < n && !r->bad; ++k) skip_value(r, et, depth + 1);
        return;
    }
    if (type < 13 && kWidth[type]) { skip(r, (uint64_t) kWidth[type]); return; }
    r->bad = 1;
}

int gguf_open(Gguf* g, const char* path, char* err, int errlen) {
    memset(g, 0, sizeof *g);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(err, (size_t) errlen, "%s: %s", path, strerror(errno)); return -1; }
    struct stat sb;
    fstat(fd, &sb);
    g->size = (size_t) sb.st_size;
    void* m = g->size ? mmap(NULL, g->size, PROT_READ, MAP_PRIVATE, fd, 0) : MAP_FAILED;
    close(fd);
    if (m == MAP_FAILED) { snprintf(err, (size_t) errlen, "%s: mmap failed", path); return -1; }
    g->map = (const uint8_t*) m;
    Rd r = {g->map, g->size, 0, 0};
    const uint64_t magic = rd(&r, 4), ver = rd(&r, 4), nt = rd(&r, 8), nkv = rd(&r, 8);
    uint64_t align = 32;
    if (r.bad || magic != 0x46554747u || (ver != 2 && ver != 3) || nt > (1u << 24)) goto bad;
    for (uint64_t k = 0; k < nkv && !r.bad; ++k) {
        const uint64_t kl = rd(&r, 8);
        const size_t k0 = r.i;
        skip(&r, kl);
        const uint32_t type = (uint32_t) rd(&r, 4);
        if (!r.bad && kl == 17 && !memcmp(r.p + k0, "general.alignment", 17) && type == 4) align = rd(&r, 4);
        else skip_value(&r, type, 0);
    }
    if (r.bad || !align || align > 4096) goto bad;
    g->t = (GgufTensor*) calloc(nt ? (size_t) nt : 1, sizeof(GgufTensor));
    for (uint64_t k = 0; k < nt && !r.bad; ++k) {
        GgufTensor* t = &g->t[g->n];
        const uint64_t nl = rd(&r, 8);
        if (nl >= sizeof t->name || nl > r.n - r.i) { r.bad = 1; break; }
        memcpy(t->name, r.p + r.i, (size_t) nl);
        r.i += (size_t) nl;
        t->ndim = (int) rd(&r, 4);
        if (t->ndim < 1 || t->ndim > 4) { r.bad = 1; break; }
        for (int d = 0; d < t->ndim; ++d) t->dims[d] = (int64_t) rd(&r, 8);
        t->type = (uint32_t) rd(&r, 4);
        t->offset = rd(&r, 8);
        ++g->n;
    }
    g->data = (r.i + align - 1) / align * align;
    for (int k = 0; k < g->n && !r.bad; ++k) if (g->data + g->t[k].offset > g->size) r.bad = 1;
    if (r.bad) goto bad;
    return 0;
bad:
    snprintf(err, (size_t) errlen, "%s: not a readable GGUF file", path);
    gguf_close(g);
    return -1;
}

void gguf_close(Gguf* g) {
    if (g->map) munmap((void*) g->map, g->size);
    free(g->t);
    memset(g, 0, sizeof *g);
}

const GgufTensor* gguf_find(const Gguf* g, const char* name) {
    for (int k = 0; k < g->n; ++k) if (!strcmp(g->t[k].name, name)) return &g->t[k];
    return NULL;
}

const float* gguf_f32(const Gguf* g, const char* name, int64_t n) {
    const GgufTensor* t = gguf_find(g, name);
    if (!t || t->type != GGUF_F32) return NULL;
    int64_t m = 1;
    for (int d = 0; d < t->ndim; ++d) m *= t->dims[d];
    if (m != n || g->data + t->offset + 4 * (uint64_t) n > g->size) return NULL;
    return (const float*) gguf_data(g, t);
}
