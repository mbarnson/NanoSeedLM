// tests/test_mova_cfg.c - the MoVA config block (nslm/mova_cfg.h): parsing MoVA's config.json, rejecting unsupported
// variants, and the logical tensor list (names, kinds, stacked shapes, totals) the packer and the engine share.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mova_cfg.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char** argv) {
    char def[512];
    snprintf(def, sizeof def, "%s/.cache/huggingface/hub/models--IFM--K2-Horizon-MoVA-36B-A4B/snapshots/"
             "cca48b6631d03c338a0e1d8beb0c392af8becbd3", getenv("HOME"));
    const char* dir = argc > 1 ? argv[1] : getenv("MOVA_DIR") ? getenv("MOVA_DIR") : def;
    MovaCfg c;
    char err[256] = "";
    CHECK(mova_cfg_load(&c, dir, err, sizeof err) == 0, "load: %s", err);
    CHECK(c.n_layer == 48 && c.d == 2560 && c.n_head == 32 && c.n_kv == 8 && c.head_dim == 128, "attention dims");
    CHECK(c.ff_dense == 6144 && c.ff_exp == 768 && c.n_exp == 100 && c.top_k == 8 && c.n_vexp == 64 && c.top_kv == 4, "moe dims");
    CHECK(c.vocab == 250624 && c.first_sparse == 3 && c.norm_groups == 2, "vocab, sparse, norm groups");
    CHECK(c.rope_theta == 1e7f && c.eps == 1e-6f && c.route_scale == 2.5f && c.router_parts == 2, "constants");
    // rejections
    const char* bad[] = {"\"hidden_act\": \"gelu\"", "\"router_score_func\": \"softmax\"", "\"num_shared_experts\": 2",
                         "\"query_key_norm\": true"};
    for (int i = 0; i < 4; ++i) {
        MovaCfg d;
        CHECK(mova_cfg_parse(&d, bad[i], err, sizeof err) != 0, "accepted unsupported %s", bad[i]);
    }
    // the tensor list
    MovaTensor* t = NULL;
    const int n = mova_tensors(&c, &t);
    int n_exp = 0, n_vexp = 0, n_router = 0, n_norm = 0;
    double params = 0;
    for (int i = 0; i < n; ++i) {
        params += (double) t[i].slices * t[i].rows * t[i].cols;
        n_exp += t[i].kind == MOVA_K_EXPERTS;
        n_vexp += t[i].kind == MOVA_K_VEXPERTS;
        n_router += t[i].kind == MOVA_K_ROUTER;
        n_norm += t[i].kind == MOVA_K_NORM;
    }
    CHECK(n_exp == 45 * 3 && n_vexp == 45 && n_router == 90 && n_norm == 97, "kinds: %d %d %d %d", n_exp, n_vexp, n_router, n_norm);
    CHECK(params > 37.43e9 && params < 37.45e9, "parameter count %.4g", params);
    const MovaTensor* g = mova_find(t, n, "model.layers.3.mlp.experts.gate_proj.weight");
    CHECK(g && g->slices == 100 && g->rows == 768 && g->cols == 2560, "stacked gate");
    char src[128];
    CHECK(g && !strcmp(mova_slice_name(g, 7, src, sizeof src), "model.layers.3.mlp.experts.7.gate_proj.weight"), "slice name %s", src);
    const MovaTensor* v = mova_find(t, n, "model.layers.47.self_attn.v_experts.weight");
    CHECK(v && v->slices == 64 && v->rows == 1024 && v->cols == 2560, "stacked value experts");
    CHECK(v && !strcmp(mova_slice_name(v, 63, src, sizeof src), "model.layers.47.self_attn.v_experts.63.weight"), "v slice %s", src);
    CHECK(mova_find(t, n, "model.layers.2.self_attn.v_proj.weight") && !mova_find(t, n, "model.layers.3.self_attn.v_proj.weight"), "v_proj only dense");
    free(t);
    printf("%d tensors, %.4g params: %s\n", n, params, fails ? "FAIL" : "PASS");
    return fails != 0;
}
