// tests/test_moe.c - the MoVA helpers (nslm/moe.h): shard lookup in a safetensors index, the per-expert shrinkage
// blend of calibration statistics, and the 3-D BF16 safetensors writer (read back through st_open_file).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "format.h"
#include "moe.h"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(void) {
    // ---- index lookup ----
    const char* idx = "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {\"model.layers.3.mlp.experts.0.gate_proj.weight\": "
                      "\"model-00007-of-00048.safetensors\", \"model.layers.3.mlp.experts.10.gate_proj.weight\": \"model-00008-of-00048.safetensors\"}}";
    char f[256];
    CHECK(nslm_moe_index_lookup(idx, "model.layers.3.mlp.experts.10.gate_proj.weight", f, sizeof f) == 0 &&
          !strcmp(f, "model-00008-of-00048.safetensors"), "lookup expert 10: %s", f);
    CHECK(nslm_moe_index_lookup(idx, "model.layers.3.mlp.experts.0.gate_proj.weight", f, sizeof f) == 0 &&
          !strcmp(f, "model-00007-of-00048.safetensors"), "lookup expert 0 (prefix of 10 must not match 1x): %s", f);
    CHECK(nslm_moe_index_lookup(idx, "model.layers.3.mlp.experts.1.gate_proj.weight", f, sizeof f) != 0, "missing name found");

    // ---- blend: h = (sum + n0 * prior) / (n + n0) ----
    const double sum[3] = {10.0, 0.0, 4.0}, prior[3] = {1.0, 2.0, 0.5};
    float h[3];
    nslm_moe_blend(sum, 4, prior, 64.0, 3, h);
    for (int c = 0; c < 3; ++c) {
        const double want = (sum[c] + 64.0 * prior[c]) / 68.0;
        CHECK(fabs(h[c] - want) < 1e-6 * want, "blend c%d %g want %g", c, h[c], want);
    }
    const double zero[3] = {0, 0, 0};
    nslm_moe_blend(zero, 0, prior, 64.0, 3, h);  // no routed tokens (so a zero sum): the prior
    CHECK(h[0] == 1.0f && h[1] == 2.0f && h[2] == 0.5f, "blend n=0");
    nslm_moe_blend(sum, 4, prior, 0.0, 3, h);    // no prior: the routed mean
    CHECK(h[0] == 2.5f && h[1] == 0.0f && h[2] == 1.0f, "blend n0=0");

    // ---- qwen4_exp expert shapes from config.json ----
    int d = 0, ff = 0, ne = 0, nl = 0;
    const char* qcfg = "{\"model_type\": \"qwen4_exp\", \"text_config\": {\"model_type\": \"qwen4_exp_text\", \"hidden_size\": 2560, "
                       "\"moe_intermediate_size\": 640, \"num_experts\": 512, \"num_hidden_layers\": 48}}";
    CHECK(nslm_qwen_experts(qcfg, &d, &ff, &ne, &nl) == 1 && d == 2560 && ff == 640 && ne == 512 && nl == 48,
          "qwen4_exp shapes %d %d %d %d", d, ff, ne, nl);
    CHECK(nslm_qwen_experts("{\"model_type\": \"k2_horizon\", \"hidden_size\": 2560}", &d, &ff, &ne, &nl) == 0, "MoVA is not qwen4_exp");
    CHECK(nslm_qwen_experts("{\"text_config\": {\"model_type\": \"qwen4_exp_text\", \"hidden_size\": 2560}}", &d, &ff, &ne, &nl) == -1,
          "missing shapes accepted");
    CHECK(nslm_qwen_experts("not json", &d, &ff, &ne, &nl) == -1, "bad JSON accepted");

    // ---- imatrix h: expert 1's mean x^2, shrunk toward the mean over all experts' tokens ----
    const float isum[2 * 3] = {4, 8, 12, 30, 0, 6}, icnt[2] = {2, 3};   // [E = 2][cols = 3], tokens per expert
    nslm_imatrix_h(isum, icnt, 2, 3, 1, 5.0, h);
    for (int c = 0; c < 3; ++c) {
        const double pri = (isum[c] + isum[3 + c]) / 5.0, want = (isum[3 + c] + 5.0 * pri) / 8.0;
        CHECK(fabs(h[c] - want) < 1e-6 * (want + 1e-9), "imatrix h c%d %g want %g", c, h[c], want);
    }

    // ---- .blk4 -> SEED4P4 streams: seeds, coefficient words, int32 bias per slice, exponent codes 2 per byte ----
    {
        enum { E = 2, R = 2, C = 16, NB = R * C / 8 };
        static uint8_t img[sizeof(MoeBlkHeader) + 12 * E + 5 * NB * E];
        MoeBlkHeader* hd = (MoeBlkHeader*) img;
        memcpy(hd->magic, "NSLMBLK4", 8);
        hd->rows = R; hd->cols = C; hd->n_experts = E;
        uint8_t* b = img + sizeof *hd;
        const int32_t bias[E] = {-20, -19};
        memcpy(b, bias, sizeof bias);
        uint16_t* seed = (uint16_t*) (b + 12 * E), *nib = seed + NB * E;
        uint8_t* ec = (uint8_t*) (nib + NB * E);
        for (int k = 0; k < NB * E; ++k) { seed[k] = (uint16_t) (100 + k); nib[k] = (uint16_t) (0x1000 + k); ec[k] = (uint8_t) k; }
        uint16_t s0[NB * E], s1[NB * E];
        int32_t s2[E];
        uint8_t s3[NB * E / 2];
        CHECK(!nslm_blk_stream(img, sizeof img, 4, E, R, C, 0, (uint8_t*) s0) && s0[5] == 105, "blk4 seeds");
        CHECK(!nslm_blk_stream(img, sizeof img, 4, E, R, C, 1, (uint8_t*) s1) && s1[7] == 0x1007, "blk4 coefficient words");
        CHECK(!nslm_blk_stream(img, sizeof img, 4, E, R, C, 2, (uint8_t*) s2) && s2[1] == -19, "blk4 bias");
        CHECK(!nslm_blk_stream(img, sizeof img, 4, E, R, C, 3, s3) && s3[0] == 0x10 && s3[3] == 0x76, "blk4 codes: block 2j low nibble");
        CHECK(nslm_blk_stream(img, sizeof img, 4, E, R, 2 * C, 0, (uint8_t*) s0) != 0, "blk4 of other shape accepted");
        CHECK(nslm_blk_stream(img, sizeof img - 1, 4, E, R, C, 0, (uint8_t*) s0) != 0, "short blk4 accepted");
    }

    // ---- .blk (P = 3) -> SEED4 streams: seeds, nibble words, int32 bias per slice ----
    {
        enum { E = 2, R = 2, C = 16, NB = R * C / 8 };
        static uint8_t img[sizeof(MoeBlkHeader) + 12 * E + 4 * NB * E];
        MoeBlkHeader* hd = (MoeBlkHeader*) img;
        memcpy(hd->magic, "NSLMBLKM", 8);
        hd->rows = R; hd->cols = C; hd->n_experts = E;
        uint16_t* seed = (uint16_t*) (img + sizeof *hd + 12 * E), *nib = seed + NB * E;
        for (int k = 0; k < NB * E; ++k) { seed[k] = (uint16_t) (7 + k); nib[k] = (uint16_t) (0x2000 + k); }
        uint16_t s0[NB * E], s1[NB * E];
        CHECK(!nslm_blk_stream(img, sizeof img, 3, E, R, C, 0, (uint8_t*) s0) && s0[3] == 10, "blk seeds");
        CHECK(!nslm_blk_stream(img, sizeof img, 3, E, R, C, 1, (uint8_t*) s1) && s1[6] == 0x2006, "blk nibble words");
        CHECK(nslm_blk_stream(img, sizeof img, 4, E, R, C, 0, (uint8_t*) s0) != 0, "a .blk read as .blk4");
    }

    // ---- the MLX loader ships with nanoseedlm_seeds.py ----
    {
        mkdir("out", 0755);   // a fresh checkout has no out/
        mkdir("out/test_moe_loader", 0755);
        mkdir("out/test_moe_pack", 0755);
        remove("out/test_moe_loader/nanoseedlm_seeds.py");   // a clean start on reruns
        FILE* lf = fopen("out/test_moe_loader/nanoseedlm_x.py", "wb");
        fputs("x = 1\n", lf);
        fclose(lf);
        CHECK(nslm_copy_loader("out/test_moe_loader/nanoseedlm_x.py", "out/test_moe_pack") != 0, "loader without the seed core copied");
        lf = fopen("out/test_moe_loader/nanoseedlm_seeds.py", "wb");
        fputs("y = 2\n", lf);
        fclose(lf);
        CHECK(nslm_copy_loader("out/test_moe_loader/nanoseedlm_x.py", "out/test_moe_pack") == 0, "copy loader");
        FILE* a = fopen("out/test_moe_pack/nanoseedlm_x.py", "rb"), *b = fopen("out/test_moe_pack/nanoseedlm_seeds.py", "rb");
        CHECK(a && b, "loader and seed core in the pack");
        if (a) fclose(a);
        if (b) fclose(b);
    }

    // ---- 3-D BF16 safetensors round trip ----
    const int d0 = 3, d1 = 5, d2 = 8;
    uint16_t* w = (uint16_t*) malloc(2 * (size_t) d0 * d1 * d2);
    for (int i = 0; i < d0 * d1 * d2; ++i) w[i] = (uint16_t) (i * 2654435761u >> 16);
    const char* path = "out/test_moe.safetensors";
    char err[256] = "";
    CHECK(nslm_st_write_bf16_3d(path, "w", d0, d1, d2, w, err, sizeof err) == 0, "write: %s", err);
    StFile st;
    CHECK(st_open_file(&st, path, err, sizeof err) == 0, "read back: %s", err);
    const StEntry* e = st_find(&st, "w");
    CHECK(e && e->ndim == 3 && e->shape[0] == d0 && e->shape[1] == d1 && e->shape[2] == d2 && !strcmp(e->dtype, "BF16"),
          "header");
    if (e) CHECK(!memcmp(st_data(&st, e), w, 2 * (size_t) d0 * d1 * d2), "data");
    st_close_file(&st);
    free(w);
    printf("%s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
