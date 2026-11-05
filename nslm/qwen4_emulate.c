// nslm/qwen4_emulate.c - nslm-qwen4-emulate: a seed pack's routed experts written into the Q8_0 expert tensors of
// Unsloth's Q8_0 GGUF shards in place, so that llama.cpp runs them (an emulated pack, nslm/emulate.h).
//   nslm-qwen4-emulate --pack DIR [--workers 8] SHARD.gguf [SHARD.gguf ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "emulate.h"

int main(int argc, char** argv) {
    const char* dir = NULL;
    int workers = 8, nf = 0;
    const char* files[64];
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--workers") && i + 1 < argc) workers = atoi(argv[++i]);
        else if (argv[i][0] != '-' && nf < 64) files[nf++] = argv[i];
        else { nf = 0; break; }
    }
    if (!dir || !nf) { fprintf(stderr, "usage: nslm-qwen4-emulate --pack DIR [--workers 8] SHARD.gguf [SHARD.gguf ...]\n"); return 2; }
    NsModel m;
    char err[512];
    if (ns_open(&m, dir, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    int want = 0, done = 0;
    for (int i = 0; i < m.n; ++i)   // the main layers: the GGUF keeps the MTP layer in another file
        want += !strncmp(m.t[i].name, "language_model.model.layers.", 28) && strstr(m.t[i].name, ".mlp.switch_mlp.");
    for (int f = 0; f < nf; ++f) {
        const time_t t0 = time(NULL);
        int n = 0;
        if (ns_emulate_gguf(&m, files[f], workers, &n, err, sizeof err)) { fprintf(stderr, "%s\n", err); ns_close(&m); return 1; }
        printf("%s: %d expert tensors, %ld s\n", files[f], n, (long) (time(NULL) - t0));
        fflush(stdout);
        done += n;
    }
    printf("%d of the pack's %d expert tensors written\n", done, want);
    ns_close(&m);
    return done == want ? 0 : 1;
}
