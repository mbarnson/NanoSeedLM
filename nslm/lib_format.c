// nslm/lib_format.c - see format.h.  C99 + POSIX mmap.
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "format.h"

static const char* skip_ws(const char* p, const char* e) { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; return p; }

static const char* parse_str(const char* p, const char* e, char* out, int cap) {
    if (p >= e || *p != '"') return NULL;
    ++p;
    int n = 0;
    while (p < e && *p != '"') {
        if (*p == '\\' && p + 1 < e) ++p;
        if (n < cap - 1) out[n++] = *p;
        ++p;
    }
    out[n] = 0;
    return p < e ? p + 1 : NULL;
}

static const char* skip_value(const char* p, const char* e) {   // any JSON value (for __metadata__)
    p = skip_ws(p, e);
    if (p >= e) return NULL;
    if (*p == '"') { char tmp[8]; return parse_str(p, e, tmp, sizeof tmp); }
    if (*p == '{' || *p == '[') {
        int depth = 0, in_str = 0;
        for (; p < e; ++p) {
            if (in_str) { if (*p == '\\') ++p; else if (*p == '"') in_str = 0; continue; }
            if (*p == '"') in_str = 1;
            else if (*p == '{' || *p == '[') ++depth;
            else if (*p == '}' || *p == ']') { if (--depth == 0) return p + 1; }
        }
        return NULL;
    }
    while (p < e && *p != ',' && *p != '}') ++p;
    return p;
}

int st_open_file(StFile* s, const char* path, char* err, int errlen) {
    memset(s, 0, sizeof *s);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) { snprintf(err, (size_t) errlen, "%s: %s", path, strerror(errno)); return -1; }
    struct stat sb;
    fstat(fd, &sb);
    s->size = (size_t) sb.st_size;
    void* m = mmap(NULL, s->size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { snprintf(err, (size_t) errlen, "%s: mmap failed", path); return -1; }
    s->map = (const uint8_t*) m;
    uint64_t hn;
    memcpy(&hn, s->map, 8);
    s->data = 8 + hn;
    const char* p = (const char*) s->map + 8, *e = p + hn;
    int cap = 512;
    s->e = (StEntry*) calloc((size_t) cap, sizeof(StEntry));
    p = skip_ws(p, e);
    if (p >= e || *p != '{') goto bad;
    ++p;
    for (;;) {
        p = skip_ws(p, e);
        if (p < e && *p == '}') break;
        char key[96];
        if (!(p = parse_str(p, e, key, sizeof key))) goto bad;
        p = skip_ws(p, e);
        if (p >= e || *p != ':') goto bad;
        p = skip_ws(p + 1, e);
        if (!strcmp(key, "__metadata__")) {
            if (!(p = skip_value(p, e))) goto bad;
        } else {
            if (s->n == cap) {
                cap *= 2;
                s->e = (StEntry*) realloc(s->e, sizeof(StEntry) * (size_t) cap);
            }
            StEntry* t = &s->e[s->n++];
            snprintf(t->name, sizeof t->name, "%s", key);
            if (*p != '{') goto bad;
            ++p;
            for (;;) {
                p = skip_ws(p, e);
                if (*p == '}') { ++p; break; }
                char f[32];
                if (!(p = parse_str(p, e, f, sizeof f))) goto bad;
                p = skip_ws(p, e);
                if (*p != ':') goto bad;
                p = skip_ws(p + 1, e);
                if (!strcmp(f, "dtype")) {
                    if (!(p = parse_str(p, e, t->dtype, sizeof t->dtype))) goto bad;
                } else if (!strcmp(f, "shape") || !strcmp(f, "data_offsets")) {
                    if (*p != '[') goto bad;
                    ++p;
                    int64_t v[4];
                    int k = 0;
                    for (;;) {
                        p = skip_ws(p, e);
                        if (*p == ']') { ++p; break; }
                        char* q;
                        const long long x = strtoll(p, &q, 10);
                        if (q == p || k >= 4) goto bad;
                        v[k++] = x;
                        p = skip_ws(q, e);
                        if (*p == ',') ++p;
                    }
                    if (f[0] == 's') { t->ndim = k; memcpy(t->shape, v, sizeof(int64_t) * (size_t) k); }
                    else { if (k != 2) goto bad; t->begin = (uint64_t) v[0]; t->end = (uint64_t) v[1]; }
                } else {
                    if (!(p = skip_value(p, e))) goto bad;
                }
                p = skip_ws(p, e);
                if (*p == ',') ++p;
            }
        }
        p = skip_ws(p, e);
        if (*p == ',') ++p;
    }
    return 0;
bad:
    snprintf(err, (size_t) errlen, "%s: cannot parse the safetensors header", path);
    st_close_file(s);
    return -1;
}

void st_close_file(StFile* s) {
    if (s->map) munmap((void*) s->map, s->size);
    free(s->e);
    memset(s, 0, sizeof *s);
}

const StEntry* st_find(const StFile* s, const char* name) {
    for (int i = 0; i < s->n; ++i) if (!strcmp(s->e[i].name, name)) return &s->e[i];
    return NULL;
}
