// harness/suite.h - prompt suites from tools/mova_export.py prompts: PRE.ids (int32, every prompt in a row) and
// PRE.index (one line per prompt: "id n_tokens max_tokens [temperature seed]").  Header-only, C99.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"

typedef struct {
    char id[64];
    int n, max_tokens;
    double temperature;   // 0.7 if the line has none
    uint64_t seed;
    const int32_t* ids;   // inside Suite.all
} SuitePrompt;

typedef struct {
    int32_t* all;
    SuitePrompt* p;
    int n;
} Suite;

static inline int suite_load(Suite* s, const char* pre) {
    memset(s, 0, sizeof *s);
    char path[2048];
    size_t len = 0, ilen = 0;
    snprintf(path, sizeof path, "%s.ids", pre);
    s->all = (int32_t*) plat_slurp(path, &len);
    snprintf(path, sizeof path, "%s.index", pre);
    char* idx = plat_slurp(path, &ilen);
    if (!s->all || !idx) { free(s->all); free(idx); return -1; }
    size_t off = 0;
    for (char* line = strtok(idx, "\r\n"); line; line = strtok(NULL, "\r\n")) {
        SuitePrompt q;
        memset(&q, 0, sizeof q);
        double t = 0.7;
        unsigned long long seed = 0;
        const int k = sscanf(line, "%63s %d %d %lf %llu", q.id, &q.n, &q.max_tokens, &t, &seed);
        if (k < 3) continue;
        q.temperature = k >= 5 ? t : 0.7;
        q.seed = k >= 5 ? (uint64_t) seed : 0;
        if ((off + (size_t) q.n) * 4 > len) break;
        q.ids = s->all + off;
        off += (size_t) q.n;
        s->p = (SuitePrompt*) realloc(s->p, sizeof(SuitePrompt) * (size_t) (s->n + 1));
        s->p[s->n++] = q;
    }
    free(idx);
    return 0;
}
static inline void suite_free(Suite* s) { free(s->all); free(s->p); memset(s, 0, sizeof *s); }

// --only id,id: is this prompt listed (or no list given)?
static inline int suite_selected(const char* only, const char* id) {
    if (!only) return 1;
    const size_t l = strlen(id);
    for (const char* p = only; (p = strstr(p, id)); ++p)
        if ((p == only || p[-1] == ',') && (p[l] == ',' || p[l] == 0)) return 1;
    return 0;
}
