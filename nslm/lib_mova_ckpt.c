// nslm/lib_mova_ckpt.c - MoVA's HF checkpoint reader (nslm/mova_ckpt.h).
#include "mova_ckpt.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "format.h"
#include "moe.h"

struct MovaCkpt {
    char dir[1024];
    char* index;
    struct { char name[128]; StFile st; } shard[64];   // the shard file name (as file below)
    int nshard;
    pthread_mutex_t mu;
};

MovaCkpt* mova_ckpt_open(const char* model_dir, char* err, int errlen) {
    MovaCkpt* k = (MovaCkpt*) calloc(1, sizeof *k);
    snprintf(k->dir, sizeof k->dir, "%s", model_dir);
    pthread_mutex_init(&k->mu, NULL);
    char path[1200];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (!f) { snprintf(err, (size_t) errlen, "cannot read %s", path); free(k); return NULL; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    k->index = (char*) malloc((size_t) n + 1);
    const size_t got = fread(k->index, 1, (size_t) n, f);
    fclose(f);
    k->index[got] = 0;
    return k;
}

void mova_ckpt_close(MovaCkpt* k) {
    if (!k) return;
    for (int i = 0; i < k->nshard; ++i) st_close_file(&k->shard[i].st);
    free(k->index);
    free(k);
}

const uint16_t* mova_ckpt_bf16(MovaCkpt* k, const char* name, int rows, int cols, char* err, int errlen) {
    char file[128], path[1200];
    if (nslm_moe_index_lookup(k->index, name, file, sizeof file)) { snprintf(err, (size_t) errlen, "%s: not in the index", name); return NULL; }
    pthread_mutex_lock(&k->mu);
    int i = 0;
    while (i < k->nshard && strcmp(k->shard[i].name, file)) ++i;
    if (i == k->nshard) {
        snprintf(path, sizeof path, "%s/%s", k->dir, file);
        if (k->nshard == 64 || st_open_file(&k->shard[i].st, path, err, errlen)) { pthread_mutex_unlock(&k->mu); return NULL; }
        snprintf(k->shard[i].name, sizeof k->shard[i].name, "%s", file);
        ++k->nshard;
    }
    const StFile* st = &k->shard[i].st;
    pthread_mutex_unlock(&k->mu);
    const StEntry* e = st_find(st, name);
    const int64_t r = e ? (e->ndim == 1 ? 1 : e->shape[0]) : 0, c = e ? (e->ndim == 1 ? e->shape[0] : e->shape[1]) : 0;
    if (!e || strcmp(e->dtype, "BF16") || e->ndim > 2 || r != rows || c != cols) {
        snprintf(err, (size_t) errlen, "%s: missing, not BF16, or not %d x %d", name, rows, cols);
        return NULL;
    }
    return (const uint16_t*) st_data(st, e);
}
