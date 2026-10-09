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
             "cca48b6631d03c338a0e1d8beb0c392af8becbd3", getenv("HOME") ? getenv("HOME") : getenv("USERPROFILE") ? getenv("USERPROFILE") : ".");
    {   // mova_mla_keep_q8 (nslm-mova-pack --mla-q8): names, optionally for layers A-B (or one layer A)
        const char* l3 = "model.layers.3.self_attn.mla.v_up", *l30 = "model.layers.30.self_attn.mla.v_up.weight";
        CHECK(!mova_mla_keep_q8("", l3) && mova_mla_keep_q8("v_up", l3) && mova_mla_keep_q8("q_lat,v_up", l30), "names");
        CHECK(mova_mla_keep_q8("v_up@0-23", l3) && !mova_mla_keep_q8("v_up@0-23", l30) && mova_mla_keep_q8("v_up@24-47", l30), "ranges");
        CHECK(mova_mla_keep_q8("v_up@3", l3) && !mova_mla_keep_q8("v_up@30", l3) && mova_mla_keep_q8("kv_a_x,v_up@30", l30), "one layer");
        CHECK(!mova_mla_keep_q8("v_u", l3) && !mova_mla_keep_q8("v_up", "model.layers.3.self_attn.q_proj.weight"), "exact names, MLA only");
        CHECK(!mova_mla_keep_q8("kv_a_x", "model.layers.3.self_attn.mla.kv_a_x_extra"), "no prefix match");
    }
    const char* dir = argc > 1 ? argv[1] : getenv("MOVA_DIR") ? getenv("MOVA_DIR") : def;
    char probe[1100];
    snprintf(probe, sizeof probe, "%s/config.json", dir);
    FILE* pf = fopen(probe, "rb");
    if (!pf) { printf("SKIP: no model folder (MOVA_DIR, or the Hugging Face snapshot): %s\n", dir); return 77; }
    fclose(pf);
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
    CHECK(!c.mla, "GQA config read as MLA");
    free(t);

    // MLA (TransMLA conversion): the same config.json plus mla_ranks / mla_rope_dim
    FILE* cf = fopen(probe, "rb");
    fseek(cf, 0, SEEK_END);
    const long cn = ftell(cf);
    fseek(cf, 0, SEEK_SET);
    char* js = (char*) malloc((size_t) cn + 1);
    js[fread(js, 1, (size_t) cn, cf)] = 0;
    fclose(cf);
    char ranks[48 * 8 + 64] = "";
    for (int l = 0; l < 48; ++l) snprintf(ranks + strlen(ranks), sizeof ranks - strlen(ranks), "%s%d", l ? ", " : "", l == 5 ? 512 : 768);
    char* mjs = (char*) malloc((size_t) cn + 1024);
    snprintf(mjs, (size_t) cn + 1024, "{\"mla_ranks\": [%s], \"mla_rope_dim\": 128, %s", ranks, strchr(js, '{') + 1);
    MovaCfg m;
    CHECK(mova_cfg_parse(&m, mjs, err, sizeof err) == 0, "MLA parse: %s", err);
    CHECK(m.mla && m.mla_rope == 128 && m.mla_rank[0] == 768 && m.mla_rank[5] == 512 && m.mla_rank[47] == 768, "MLA fields");
    const char* mbad[] = {"[768, 768]", "[768, 760, 768]"};    // wrong count; rank not a multiple of 32
    for (int i = 0; i < 2; ++i) {
        snprintf(mjs, (size_t) cn + 1024, "{\"mla_ranks\": %s, %s", mbad[i], strchr(js, '{') + 1);
        MovaCfg d;
        CHECK(mova_cfg_parse(&d, mjs, err, sizeof err) != 0, "accepted mla_ranks %s", mbad[i]);
    }
    snprintf(mjs, (size_t) cn + 1024, "{\"mla_ranks\": [%s], \"mla_rope_dim\": 64, %s", ranks, strchr(js, '{') + 1);
    MovaCfg d64;
    CHECK(mova_cfg_parse(&d64, mjs, err, sizeof err) != 0, "accepted mla_rope_dim 64");
    snprintf(mjs, (size_t) cn + 1024, "{\"mla_ranks\": [%s], \"mla_rope_dim\": 128, %s", ranks, strchr(js, '{') + 1);
    mova_cfg_parse(&m, mjs, err, sizeof err);
    const int nm = mova_tensors(&m, &t);
    int heads = 0;
    double mparams = 0;
    for (int i = 0; i < nm; ++i) {
        mparams += (double) t[i].slices * t[i].rows * t[i].cols;
        heads += t[i].kind == MOVA_K_HEADS;
    }
    CHECK(nm == 3 * 13 + 45 * 22 + 3, "MLA tensor count %d", nm);
    CHECK(heads == 48 * 3, "per-head tensors %d", heads);
    CHECK(!mova_find(t, nm, "model.layers.3.self_attn.k_proj.weight") && !mova_find(t, nm, "model.layers.0.self_attn.k_proj.weight"), "k_proj dropped");
    CHECK(!mova_find(t, nm, "model.layers.1.self_attn.v_proj.weight"), "dense v_proj folded into kv_a_x");
    CHECK(!mova_find(t, nm, "model.layers.2.self_attn.mla.kv_a_v") && mova_find(t, nm, "model.layers.3.self_attn.mla.kv_a_v"), "kv_a_v only on MoVA layers");
    const MovaTensor* ka = mova_find(t, nm, "model.layers.5.self_attn.mla.kv_a_x");
    CHECK(ka && ka->rows == 512 && ka->cols == 2560 && ka->kind == MOVA_K_LINEAR, "kv_a_x shape");
    const MovaTensor* kv = mova_find(t, nm, "model.layers.5.self_attn.mla.kv_a_v");
    CHECK(kv && kv->rows == 512 && kv->cols == 1024, "kv_a_v shape");
    const MovaTensor* kr = mova_find(t, nm, "model.layers.7.self_attn.mla.k_rope_proj");
    CHECK(kr && kr->rows == 128 && kr->cols == 2560, "k_rope_proj shape");
    const MovaTensor* qm = mova_find(t, nm, "model.layers.7.self_attn.mla.q_rope_mix");
    const MovaTensor* ql = mova_find(t, nm, "model.layers.7.self_attn.mla.q_lat");
    const MovaTensor* vu = mova_find(t, nm, "model.layers.7.self_attn.mla.v_up");
    CHECK(qm && qm->slices == 32 && qm->rows == 128 && qm->cols == 128, "q_rope_mix shape");
    CHECK(ql && ql->slices == 32 && ql->rows == 768 && ql->cols == 128, "q_lat shape");
    CHECK(vu && vu->slices == 32 && vu->rows == 128 && vu->cols == 768, "v_up shape");
    double want = params - 48.0 * 1024 * 2560 - 3.0 * 1024 * 2560;    // no k_proj; dense v_proj folded
    for (int l = 0; l < 48; ++l) {
        const double r = m.mla_rank[l];
        want += r * 2560 + 128.0 * 2560 + 32.0 * 128 * 128 + 2 * 32.0 * r * 128 + (l >= 3 ? r * 1024 : 0);
    }
    CHECK(mparams == want, "MLA parameter count %.6g, want %.6g", mparams, want);
    free(t);
    free(js);
    free(mjs);
    printf("%d tensors, %.4g params; MLA %d tensors, %.4g params: %s\n", n, params, nm, mparams, fails ? "FAIL" : "PASS");
    return fails != 0;
}
