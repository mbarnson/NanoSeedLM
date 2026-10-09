// harness/kv_disk.c - the cold KV cache (kv_disk.h).
#include "kv_disk.h"

#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "model_st.h"
#include "platform.h"
#include "sha256.h"

#define KVD_MAGIC "NSLMKV1\n"
#define KVD_HEAD (8 + 4 + 4 + 8 + 32 + 32)   // magic, version, block, data bytes, fingerprint, hash; then the token ids

typedef struct {
    char dir[17], name[65];   // DIR/<dir>/<name>.kv
    uint64_t size;
    double t;                 // last use
} Ent;

struct KvDisk {
    char root[1024], fp_hex[17];
    uint8_t fp[32];
    uint64_t budget, block_bytes, used;
    double max_age;   // seconds; 0: none
    Ent* e;
    int n, cap;
    unsigned long tick;   // orders uses within a second, and names temporary files
    pthread_mutex_t mu;
};

static double stamp(KvDisk* d) { return (double) time(NULL) + 1e-6 * (double) (++d->tick % 1000000); }   // under mu
static void hex(const uint8_t* p, int n, char* out) {
    static const char* x = "0123456789abcdef";
    for (int i = 0; i < n; ++i) { out[2 * i] = x[p[i] >> 4]; out[2 * i + 1] = x[p[i] & 15]; }
    out[2 * n] = 0;
}

int kvd_fingerprint(const char* model_dir, const char* desc, int64_t kv_bytes, uint8_t out[32]) {
    Sha256 s;
    sha256_init(&s);
    sha256_update(&s, KVD_MAGIC, 8);
    sha256_update(&s, desc, strlen(desc) + 1);
    sha256_update(&s, &kv_bytes, sizeof kv_bytes);
    NsModel m;
    char err[256];
    if (ns_open(&m, model_dir, err, sizeof err)) return -1;
    for (int i = 0; i < m.n; ++i) {
        const NsTensor* t = &m.t[i];
        sha256_update(&s, t->name, strlen(t->name) + 1);
        const int32_t dims[4] = {t->enc, t->slices, t->rows, t->cols};
        sha256_update(&s, dims, sizeof dims);
        for (int k = 0; k < 4; ++k) {
            sha256_update(&s, &t->s[k].len, sizeof t->s[k].len);
            if (t->s[k].len) sha256_update(&s, t->s[k].p, t->s[k].len < 64 ? (size_t) t->s[k].len : 64);
        }
    }
    ns_close(&m);
    sha256_final(&s, out);
    return 0;
}

static void ent_add(KvDisk* d, const char* dir, const char* name, uint64_t size, double t) {
    if (d->n == d->cap) { d->cap = d->cap ? 2 * d->cap : 256; d->e = (Ent*) realloc(d->e, sizeof(Ent) * (size_t) d->cap); }
    Ent* x = &d->e[d->n++];
    snprintf(x->dir, sizeof x->dir, "%s", dir);
    snprintf(x->name, sizeof x->name, "%s", name);
    x->size = size;
    x->t = t;
    d->used += size;
}
static int ent_find(KvDisk* d, const char* dir, const char* name) {
    for (int i = 0; i < d->n; ++i) if (!strcmp(d->e[i].name, name) && !strcmp(d->e[i].dir, dir)) return i;
    return -1;
}
static void ent_del(KvDisk* d, int i) {
    d->used -= d->e[i].size;
    d->e[i] = d->e[--d->n];
}

KvDisk* kvd_open(const char* dir, uint64_t budget, const uint8_t fingerprint[32], uint64_t block_bytes, char* err, int errlen) {
    if (!budget) { snprintf(err, (size_t) errlen, "no budget"); return NULL; }
    KvDisk* d = (KvDisk*) calloc(1, sizeof *d);
    snprintf(d->root, sizeof d->root, "%s", dir);
    memcpy(d->fp, fingerprint, 32);
    hex(fingerprint, 8, d->fp_hex);
    d->budget = budget;
    d->block_bytes = block_bytes;
    pthread_mutex_init(&d->mu, NULL);
    char p[1200];
    for (char* q = d->root + 1; *q; ++q)   // the directory and its parents
        if (*q == '/' || *q == '\\') { const char c = *q; *q = 0; mkdir(d->root, 0755); *q = c; }
    mkdir(d->root, 0755);
    snprintf(p, sizeof p, "%s/%s", d->root, d->fp_hex);
    mkdir(p, 0755);
    struct stat st;
    if (stat(p, &st) || !(st.st_mode & S_IFDIR)) { snprintf(err, (size_t) errlen, "cannot create %.900s", p); kvd_close(d); return NULL; }
    // the index: every model's blocks (one budget); temporary files of an interrupted write are deleted
    DIR* r = opendir(d->root);
    for (struct dirent* a; r && (a = readdir(r));) {
        if (strlen(a->d_name) != 16) continue;
        snprintf(p, sizeof p, "%s/%s", d->root, a->d_name);
        DIR* s = opendir(p);
        for (struct dirent* b; s && (b = readdir(s));) {
            const size_t l = strlen(b->d_name);
            char f[1300];
            snprintf(f, sizeof f, "%s/%s", p, b->d_name);
            if (l > 4 && !strcmp(b->d_name + l - 4, ".tmp")) { unlink(f); continue; }
            if (l != 67 || strcmp(b->d_name + 64, ".kv") || stat(f, &st)) continue;
            char name[65];
            memcpy(name, b->d_name, 64);
            name[64] = 0;
            ent_add(d, a->d_name, name, (uint64_t) st.st_size, (double) st.st_mtime);
        }
        if (s) closedir(s);
    }
    if (r) closedir(r);
    return d;
}

void kvd_close(KvDisk* d) {
    if (!d) return;
    pthread_mutex_destroy(&d->mu);
    free(d->e);
    free(d);
}

void kvd_hashes(const KvDisk* d, const int32_t* ids, int nb, uint8_t (*h)[32]) {
    for (int b = 0; b < nb; ++b) {
        Sha256 s;
        sha256_init(&s);
        sha256_update(&s, b ? h[b - 1] : d->fp, 32);
        sha256_update(&s, ids + (size_t) b * KVD_BLOCK, sizeof(int32_t) * KVD_BLOCK);
        sha256_final(&s, h[b]);
    }
}

void kvd_path(const KvDisk* d, const uint8_t h[32], char* out, int cap) {
    char x[65];
    hex(h, 32, x);
    snprintf(out, (size_t) cap, "%s/%s/%s.kv", d->root, d->fp_hex, x);
}

int kvd_count(KvDisk* d, const uint8_t (*h)[32], int nb) {
    int b = 0;
    char p[1300];
    struct stat st;
    for (; b < nb; ++b) {
        kvd_path(d, h[b], p, sizeof p);
        if (stat(p, &st)) break;
    }
    return b;
}

// Under mu: deletes the blocks unused for longer than max_age
static void expire_locked(KvDisk* d) {
    if (!(d->max_age > 0)) return;
    const double cut = (double) time(NULL) - d->max_age;
    for (int i = 0; i < d->n;) {
        if (d->e[i].t >= cut) { ++i; continue; }
        char q[1300];
        snprintf(q, sizeof q, "%s/%s/%s.kv", d->root, d->e[i].dir, d->e[i].name);
        unlink(q);
        ent_del(d, i);
    }
}
void kvd_set_max_age(KvDisk* d, double seconds) {
    pthread_mutex_lock(&d->mu);
    d->max_age = seconds;
    expire_locked(d);
    pthread_mutex_unlock(&d->mu);
}
static void drop_locked(KvDisk* d, const char* dir, const char* name) {
    const int i = ent_find(d, dir, name);
    if (i >= 0) ent_del(d, i);
}
void kvd_drop(KvDisk* d, const uint8_t h[32]) {
    char p[1300], x[65];
    kvd_path(d, h, p, sizeof p);
    hex(h, 32, x);
    unlink(p);
    pthread_mutex_lock(&d->mu);
    drop_locked(d, d->fp_hex, x);
    pthread_mutex_unlock(&d->mu);
}

// A block was used: its index time and its file's (the order across restarts)
static void touch(KvDisk* d, const char* path, const char* x) {
    pthread_mutex_lock(&d->mu);
    const int i = ent_find(d, d->fp_hex, x);
    const double t = stamp(d);
    if (i >= 0) d->e[i].t = t;
    pthread_mutex_unlock(&d->mu);
    plat_set_mtime(path, t);
}
int kvd_load(KvDisk* d, const uint8_t h[32], const int32_t* ids, void* dst) {
    char p[1300];
    kvd_path(d, h, p, sizeof p);
    FILE* f = fopen(p, "rb");
    if (!f) return -1;
    uint8_t hd[KVD_HEAD];
    int32_t t[KVD_BLOCK];
    uint32_t ver, blk;
    uint64_t nb;
    int ok = fread(hd, 1, KVD_HEAD, f) == KVD_HEAD && fread(t, sizeof t, 1, f) == 1;
    if (ok) {
        memcpy(&ver, hd + 8, 4);
        memcpy(&blk, hd + 12, 4);
        memcpy(&nb, hd + 16, 8);
        ok = !memcmp(hd, KVD_MAGIC, 8) && ver == 1 && blk == KVD_BLOCK && nb == d->block_bytes && !memcmp(hd + 24, d->fp, 32) &&
             !memcmp(hd + 56, h, 32) && !memcmp(t, ids, sizeof t);
    }
    char extra;
    ok = ok && fread(dst, 1, (size_t) d->block_bytes, f) == d->block_bytes && fread(&extra, 1, 1, f) == 0;
    fclose(f);
    char x[65];
    hex(h, 32, x);
    if (!ok) {
        unlink(p);
        pthread_mutex_lock(&d->mu);
        drop_locked(d, d->fp_hex, x);
        pthread_mutex_unlock(&d->mu);
        return -1;
    }
    touch(d, p, x);
    return 0;
}

int kvd_store(KvDisk* d, const uint8_t h[32], const int32_t* ids, const void* src) {
    char p[1300], tp[1310], x[65];
    kvd_path(d, h, p, sizeof p);
    hex(h, 32, x);
    struct stat st;
    if (!stat(p, &st)) { touch(d, p, x); return 0; }   // there already: used again
    pthread_mutex_lock(&d->mu);
    snprintf(tp, sizeof tp, "%s.%d.%lu.tmp", p, (int) getpid(), ++d->tick);
    pthread_mutex_unlock(&d->mu);
    FILE* f = fopen(tp, "wb");
    if (!f) return -1;
    uint8_t hd[KVD_HEAD];
    const uint32_t ver = 1, blk = KVD_BLOCK;
    memcpy(hd, KVD_MAGIC, 8);
    memcpy(hd + 8, &ver, 4);
    memcpy(hd + 12, &blk, 4);
    memcpy(hd + 16, &d->block_bytes, 8);
    memcpy(hd + 24, d->fp, 32);
    memcpy(hd + 56, h, 32);
    int ok = fwrite(hd, 1, KVD_HEAD, f) == KVD_HEAD && fwrite(ids, sizeof(int32_t), KVD_BLOCK, f) == KVD_BLOCK &&
             fwrite(src, 1, (size_t) d->block_bytes, f) == d->block_bytes && fflush(f) == 0 && fsync(fileno(f)) == 0;
    ok = fclose(f) == 0 && ok && rename(tp, p) == 0;
    if (!ok) { unlink(tp); return -1; }
    const uint64_t size = KVD_HEAD + sizeof(int32_t) * KVD_BLOCK + d->block_bytes;
    pthread_mutex_lock(&d->mu);
    drop_locked(d, d->fp_hex, x);
    ent_add(d, d->fp_hex, x, size, stamp(d));
    expire_locked(d);
    while (d->used > d->budget && d->n > 1) {   // the least recently used other block
        int lru = -1;
        for (int i = 0; i < d->n; ++i)
            if ((strcmp(d->e[i].name, x) || strcmp(d->e[i].dir, d->fp_hex)) && (lru < 0 || d->e[i].t < d->e[lru].t)) lru = i;
        if (lru < 0) break;
        char q[1300];
        snprintf(q, sizeof q, "%s/%s/%s.kv", d->root, d->e[lru].dir, d->e[lru].name);
        unlink(q);
        ent_del(d, lru);
    }
    pthread_mutex_unlock(&d->mu);
    return 0;
}

uint64_t kvd_used(KvDisk* d) {
    pthread_mutex_lock(&d->mu);
    const uint64_t u = d->used;
    pthread_mutex_unlock(&d->mu);
    return u;
}
