// nslm/qwen4_emulate.c - nslm-qwen4-emulate: a seed pack's routed experts written into the Q8_0 expert tensors of
// Unsloth's Q8_0 GGUF shards in place, so that llama.cpp runs them (an emulated pack, nslm/emulate.h).  With several
// packs, MAP (uint8 [layer][gate, up, down][expert], tools/qwen4_tier_map.py) names the pack of each expert.  A pack is a
// model folder, or a folder of S3 seed banks or trellis banks (nslm-moe --codec s3 or tq).
//   nslm-qwen4-emulate --pack DIR [--pack DIR ...] [--map FILE] [--workers 8] SHARD.gguf [SHARD.gguf ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "emulate.h"
#include "moe.h"

static uint8_t* read_map(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    uint8_t* b = NULL;
    if (f && !fseek(f, 0, SEEK_END) && (*n = (size_t) ftell(f)) > 0 && !fseek(f, 0, SEEK_SET) && (b = (uint8_t*) malloc(*n)) &&
        fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    if (f) fclose(f);
    return b;
}

// The pack's bytes for one expert of a main-layer expert tensor; its layer and projection in *lp (3 L + p), else 0.
static double slice_bytes(const NsTensor* t, int* lp) {
    char proj[8];
    int l = 0, at = 0, p = 0;
    if (sscanf(t->name, "language_model.model.layers.%d.mlp.switch_mlp.%7[a-z]_proj.weight%n", &l, proj, &at) != 2 || t->name[at])
        return 0;
    while (p < 3 && strcmp(proj, (const char*[]) {"gate", "up", "down"}[p])) ++p;
    double b = 0;
    for (int s = 0; s < 4; ++s) b += (double) t->s[s].len;
    *lp = 3 * l + p;
    return p < 3 ? b / t->slices : 0;
}

int main(int argc, char** argv) {
    const char* dirs[NS_EMU_PACKS];
    const char* files[64];
    const char* mapf = NULL;
    int workers = 8, nf = 0, np = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc && np < NS_EMU_PACKS) dirs[np++] = argv[++i];
        else if (!strcmp(argv[i], "--map") && i + 1 < argc) mapf = argv[++i];
        else if (!strcmp(argv[i], "--workers") && i + 1 < argc) workers = atoi(argv[++i]);
        else if (argv[i][0] != '-' && nf < 64) files[nf++] = argv[i];
        else { nf = 0; break; }
    }
    if (!np || !nf || (np > 1) != (mapf != NULL)) {
        fprintf(stderr, "usage: nslm-qwen4-emulate --pack DIR [--pack DIR ... --map FILE] [--workers 8] SHARD.gguf [SHARD.gguf ...]\n");
        return 2;
    }
    size_t ml = 0;
    uint8_t* map = mapf ? read_map(mapf, &ml) : NULL;
    if (mapf && !map) { fprintf(stderr, "%s: unreadable\n", mapf); return 1; }
    NsModel m[NS_EMU_PACKS];
    const NsModel* pk[NS_EMU_PACKS];
    char err[512];
    for (int k = 0; k < np; ++k) {
        if (ns_open(&m[k], dirs[k], err, sizeof err) && ns_open_bank(&m[k], dirs[k], err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
        pk[k] = &m[k];
    }
    int want = 0, done = 0, rc = 0;
    double bytes = 0, weights = 0;
    for (int i = 0; i < m[0].n; ++i) {   // the main layers: the GGUF keeps the MTP layer in another file
        int lp = 0;
        const NsTensor* t = &m[0].t[i];
        if (slice_bytes(t, &lp) == 0) continue;
        ++want;
        weights += (double) t->slices * t->rows * t->cols;
        for (int e = 0; e < t->slices; ++e) {
            const size_t at = (size_t) lp * (size_t) t->slices + (size_t) e;
            const int k = map && at < ml && map[at] < np ? map[at] : 0;
            const NsTensor* u = ns_find(&m[k], t->name);
            bytes += u ? slice_bytes(u, &lp) : 0;
        }
    }
    printf("experts: %.2f GB, %.3f bits per weight\n", bytes / 1e9, 8 * bytes / (weights > 0 ? weights : 1));
    for (int f = 0; f < nf && !rc; ++f) {
        const time_t t0 = time(NULL);
        int n = 0;
        if ((rc = ns_emulate_gguf(pk, np, map, ml, files[f], workers, &n, err, sizeof err))) fprintf(stderr, "%s\n", err);
        else printf("%s: %d expert tensors, %ld s\n", files[f], n, (long) (time(NULL) - t0));
        fflush(stdout);
        done += n;
    }
    printf("%d of the pack's %d expert tensors written\n", done, want);
    for (int k = 0; k < np; ++k) ns_close(&m[k]);
    free(map);
    return rc || done != want;
}
