// tqlib.c - probe (ctypes): encode a matrix with the repo's trellis encoder (nslm/lib_trellis.c), rows on 4 threads.
// Importance h per column (NULL: all 1).  Writes the decoded weights.
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "trellis.h"
typedef struct { const float* w; const float* h; int rows, cols, T, L, K, id; float* out; double err; } Job;
static void* run(void* a) {
    Job* j = a;
    NsTqScratch z;
    if (nslm_tq_scratch_init(&z, j->T, j->L, j->K)) { j->err = -1; return NULL; }
    const size_t nw = nslm_tq_row_words(j->cols, j->T, j->L, j->K);
    const int nb = (j->cols + j->T - 1) / j->T;
    uint32_t* words = malloc(4 * nw);
    uint16_t* sc = malloc(2 * nb);
    for (int r = j->id; r < j->rows; r += 4) {
        memset(words, 0, 4 * nw);
        j->err += nslm_tq_encode_row(j->w + (size_t) r * j->cols, j->h, j->cols, j->T, &z, words, sc, j->out + (size_t) r * j->cols);
    }
    free(words); free(sc); nslm_tq_scratch_free(&z);
    return NULL;
}
double tq_encode(const float* w, const float* h, int rows, int cols, int T, int L, int K, float* out) {
    pthread_t th[4];
    Job jb[4];
    for (int i = 0; i < 4; ++i) { jb[i] = (Job) {w, h, rows, cols, T, L, K, i, out, 0}; pthread_create(&th[i], NULL, run, &jb[i]); }
    double e = 0;
    for (int i = 0; i < 4; ++i) { pthread_join(th[i], NULL); e += jb[i].err; }
    return e;
}
