#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mova_cfg.h"
#include "model_st.h"
typedef struct { MovaCfg c; NsModel nm; } Eng;
#include "mla_head_groups.h"

int main(void) {
    int failed = 0, checks = 0;
    Eng e = {0};
    NsTensor tensors[2] = {0};
    e.c.mla = 1; e.c.n_head = 8; e.c.n_layer = 1;
    e.nm.t = tensors; e.nm.n = 2;
    strcpy(tensors[0].name, "model.layers.0.self_attn.mla.q_lat");
    strcpy(tensors[1].name, "model.layers.0.self_attn.mla.v_up");
#define EXPECT(value) do { ++checks; if ((e.c.mla ? mla_head_groups(&e.nm, e.c.n_head, e.c.n_layer) : 1) != (value)) { ++failed; fprintf(stderr, "check%d failed\n", checks); } } while (0)
    const int formats[] = {NS_BF16, NS_Q8, NS_Q4, NS_SEED4P4};
    for (int f = 0; f < 4; ++f) {
        for (int t = 0; t < 2; ++t) {
            tensors[t].enc = formats[f]; tensors[t].slices = 8;
            tensors[t].rows = tensors[t].cols = 128;
            for (int st = 0; st < 4; ++st) {
                const size_t step = (size_t) ns_stream_len(formats[f], 1, 128, 128, st);
                tensors[t].s[st].len = step * 8;
                unsigned char* data = step ? malloc(step * 8) : NULL;
                tensors[t].s[st].p = data;
                for (int h = 0; h < 8; ++h) for (size_t i = 0; i < step; ++i)
                    data[step * h + i] = (unsigned char) ((i + t * 7 + h / 4 * 3) % 251);
            }
        }
        EXPECT(4);
        for (int t = 0; t < 2; ++t) for (int st = 0; st < 4; ++st) {
            const size_t step = (size_t) ns_stream_len(formats[f], 1, 128, 128, st);
            if (!step) continue;
            unsigned char* data = (unsigned char*) tensors[t].s[st].p;
            data[step] ^= 1; EXPECT(1); data[step] ^= 1; EXPECT(4);
            --tensors[t].s[st].len; EXPECT(1); ++tensors[t].s[st].len;
            tensors[t].s[st].p = NULL; EXPECT(1); tensors[t].s[st].p = data;
        }
        e.c.n_layer = 2; EXPECT(1); e.c.n_layer = 1;
        e.nm.n = 1; EXPECT(1); e.nm.n = 2;
        tensors[1].slices = 4; EXPECT(1); tensors[1].slices = 8;
        tensors[1].enc = 99; EXPECT(1); tensors[1].enc = formats[f];
        for (int t = 0; t < 2; ++t) for (int st = 0; st < 4; ++st) {
            free((void*) tensors[t].s[st].p); memset(&tensors[t].s[st], 0, sizeof tensors[t].s[st]);
        }
    }
    e.c.mla = 0; EXPECT(1);
    printf("exact shared-group guard: %s (%dchecks, CPU only)\n", failed ? "FAILED" : "passed", checks);
    return failed != 0;
}
