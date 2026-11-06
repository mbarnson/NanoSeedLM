// tests/test_moe.c - the MoVA helpers (nslm/moe.h): shard lookup in a safetensors index, the per-expert shrinkage
// blend of calibration statistics, and the 3-D BF16 safetensors writer (read back through st_open_file).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "format.h"
#include "model_st.h"
#include "moe.h"
#include "rows.h"
#include "searchc.h"
#include "trellis.h"

static int fails = 0;
static int zero_fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) { (void) ctx; (void) t; (void) s; memset(dst, 0, len); return 0; }
static int seq_fill(void* ctx, int t, int s, uint8_t* dst, uint64_t len) { (void) ctx; (void) t; (void) s; for (uint64_t i = 0; i < len; ++i) dst[i] = (uint8_t) i; return 0; }
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

    // ---- an S3 seed bank (.blks: C = 12, P = 4, a padded last block) read as a model: rows as nslmc_decode_block ----
    {
        mkdir("out", 0755);
        mkdir("out/test_moe_bank", 0755);
        enum { E = 2, R = 3, C = 28, NBK = 3, NB = R * NBK };   // 28 columns: the last block holds 4 weights
        MoeBlkHeader h;
        memset(&h, 0, sizeof h);
        memcpy(h.magic, "NSLMBLKS", 8);
        h.rows = R; h.cols = C; h.n_experts = E;
        const int32_t bias[E] = {-20, -18};
        const float zero[2 * E] = {0};
        uint16_t sd[E * NB], cf[E * NB];
        uint8_t ec[E * NB];
        for (int k = 0; k < E * NB; ++k) { sd[k] = (uint16_t) (1000 + 37 * k); cf[k] = (uint16_t) (0x1357u * (unsigned) k); ec[k] = (uint8_t) (k % 16); }
        FILE* f = fopen("out/test_moe_bank/L1_down_proj.blks", "wb");
        fwrite(&h, sizeof h, 1, f); fwrite(bias, 4, E, f); fwrite(zero, 4, 2 * E, f);
        fwrite(sd, 2, E * NB, f); fwrite(cf, 2, E * NB, f); fwrite(ec, 1, E * NB, f);
        fclose(f);
        f = fopen("out/test_moe_bank/L2_gate_proj.blks", "wb");   // short: refused
        fwrite(&h, sizeof h, 1, f);
        fclose(f);
        NsModel m;
        char berr[256] = "";
        CHECK(ns_open_bank(&m, "out/test_moe_bank", berr, sizeof berr) != 0, "a short .blks accepted");
        remove("out/test_moe_bank/L2_gate_proj.blks");
        CHECK(ns_open_bank(&m, "out/test_moe_bank", berr, sizeof berr) == 0 && m.n == 1, "bank: %s", berr);
        const NsTensor* t = ns_find(&m, "language_model.model.layers.1.mlp.switch_mlp.down_proj.weight");
        CHECK(t && t->enc == NS_SEEDS3 && t->slices == E && t->rows == R && t->cols == C, "bank tensor");
        NsRows* nr = (NsRows*) malloc(sizeof *nr);
        ns_rows_init(nr);
        int same = 1;
        for (int e = 0; t && e < E; ++e)
            for (int r = 0; r < R; ++r) {
                float out[C];
                ns_row_f32(nr, t, e, r, 1, out);
                for (int j = 0; j < NBK; ++j) {
                    const int k = (e * R + r) * NBK + j;
                    uint16_t bf[12];
                    nslmc_decode_block(12, 4, sd[k], cf[k], bias[e] + ec[k], bf);
                    for (int c = 0; c < 12 && 12 * j + c < C; ++c) same &= out[12 * j + c] == nslm4_bf2f(bf[c]);
                }
            }
        CHECK(same, "S3 rows differ from nslmc_decode_block");
        CHECK(t && ns_stream_len(NS_SEEDS3, E, R, C, 0) == 2 * E * NB && ns_stream_len(NS_SEEDS3, E, R, C, 3) == E * NB, "S3 stream sizes");
        ns_close(&m);
        free(nr);
        CHECK(ns_open_bank(&m, "out/test_moe_pack", berr, sizeof berr) != 0, "a folder without banks opened");
    }

    // ---- a trellis bank (.blkt: nslm-moe --codec tq) read as a model: rows equal to the encoder's own output ----
    {
        mkdir("out", 0755);
        mkdir("out/test_moe_tbank", 0755);
        enum { E = 2, R = 3, C = 300, L = 8, K = 3, T = 256, NB = 2 };   // 300 columns: a short last block
        const size_t nw = nslm_tq_row_words(C, T, L, K);
        MoeBlkHeader h;
        memset(&h, 0, sizeof h);
        memcpy(h.magic, "NSLMBLKT", 8);
        h.rows = R; h.cols = C; h.n_experts = E; h.n_seeds = L; h.n_exp = K; h.refit = T;
        uint32_t* words = (uint32_t*) calloc(nw * E * R, 4);
        uint16_t sc[E * R * NB];
        float* y = (float*) malloc(sizeof(float) * E * R * C), w[C];
        const float rel[2 * E] = {0};
        NsTqScratch z;
        nslm_tq_scratch_init(&z, T, L, K);
        uint32_t x = 7;
        for (int k = 0; k < E * R; ++k) {
            for (int c = 0; c < C; ++c) { x = x * 1664525u + 1013904223u; w[c] = ((float) (x >> 8) / 16777216.0f - 0.5f) * 0.04f; }
            nslm_tq_encode_row(w, NULL, C, T, &z, words + nw * k, sc + NB * k, y + (size_t) C * k);
        }
        nslm_tq_scratch_free(&z);
        FILE* f = fopen("out/test_moe_tbank/L3_up_proj.blkt", "wb");
        fwrite(&h, sizeof h, 1, f); fwrite(rel, 4, 2 * E, f); fwrite(words, 4, nw * E * R, f); fwrite(sc, 2, E * R * NB, f);
        fclose(f);
        f = fopen("out/test_moe_tbank/L4_up_proj.blkt", "wb");   // short: refused
        fwrite(&h, sizeof h, 1, f);
        fclose(f);
        NsModel m;
        char berr[256] = "";
        CHECK(ns_open_bank(&m, "out/test_moe_tbank", berr, sizeof berr) != 0, "a short .blkt accepted");
        remove("out/test_moe_tbank/L4_up_proj.blkt");
        CHECK(ns_open_bank(&m, "out/test_moe_tbank", berr, sizeof berr) == 0 && m.n == 1, "trellis bank: %s", berr);
        const NsTensor* t = ns_find(&m, "language_model.model.layers.3.mlp.switch_mlp.up_proj.weight");
        CHECK(t && t->enc == NS_TCQ && t->slices == E && t->rows == R && t->cols == C && t->group == (L | K << 8 | T << 16),
              "trellis bank tensor");
        CHECK(t && t->s[0].len == 4 * nw * E * R && t->s[1].len == 2 * NB * E * R, "trellis stream sizes");
        NsRows* nr = (NsRows*) malloc(sizeof *nr);
        ns_rows_init(nr);
        int same = 1;
        for (int e = 0; t && e < E; ++e)
            for (int r = 0; r < R; ++r) {
                float out[C];
                ns_row_f32(nr, t, e, r, 1, out);
                same &= !memcmp(out, y + (size_t) C * (e * R + r), sizeof out);
            }
        CHECK(same, "trellis rows differ from the encoder's output");
        ns_close(&m);
        // A bank of layer n is the MTP head's when the folder has routers for layer n - 1 and the head, none for layer n
        const NsSpec sp[4] = {{"language_model.model.layers.2.mlp.switch_mlp.up_proj.weight", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.2.mlp.gate.weight", NS_RAW, 1, 1, 1},
                              {"mtp.layers.0.mlp.switch_mlp.up_proj.weight", NS_RAW, 1, 1, 1}, {"mtp.layers.0.mlp.gate.weight", NS_RAW, 1, 1, 1}};
        const NsRaw rw[4] = {{"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4}};
        mkdir("out/test_moe_thead", 0755);
        CHECK(ns_write_raw("out/test_moe_thead", sp, rw, 4, 1u << 20, NULL, zero_fill, NULL, berr, sizeof berr) == 0, "head folder: %s", berr);
        const char* bk[2] = {"out/test_moe_thead/L3_up_proj.blkt", "out/test_moe_thead/L1_up_proj.blkt"};   // L1: no layer 0
        for (int i = 0; i < 2; ++i) {
            f = fopen(bk[i], "wb");
            fwrite(&h, sizeof h, 1, f); fwrite(rel, 4, 2 * E, f); fwrite(words, 4, nw * E * R, f); fwrite(sc, 2, E * R * NB, f);
            fclose(f);
        }
        CHECK(ns_open(&m, "out/test_moe_thead", berr, sizeof berr) == 0 && ns_add_banks(&m, "out/test_moe_thead", berr, sizeof berr) == 0,
              "head banks: %s", berr);
        t = ns_find(&m, "mtp.layers.0.mlp.switch_mlp.up_proj.weight");
        CHECK(t && t->enc == NS_TCQ && !ns_find(&m, "language_model.model.layers.3.mlp.switch_mlp.up_proj.weight"),
              "bank L3 is not the MTP head's");
        t = ns_find(&m, "language_model.model.layers.1.mlp.switch_mlp.up_proj.weight");
        CHECK(t && t->enc == NS_TCQ && m.n == 5, "bank L1 (no layer 0) not added as layer 1 (%d tensors)", m.n);
        ns_close(&m);
        // nslm-qwen4-pack --tq: the base without its routed experts (a layer's, the head's), plus the banks of layers 0 and 1
        const NsSpec bs[8] = {{"language_model.model.layers.0.linear_attn.out_proj.weight", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.0.mlp.switch_mlp.up_proj.weight", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.0.mlp.gate.weight", NS_RAW, 1, 1, 1},
                              {"mtp.layers.0.mlp.switch_mlp.up_proj.weight", NS_RAW, 1, 1, 1}, {"mtp.layers.0.mlp.gate.weight", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.0.linear_attn.in_proj_z.weight", NS_RAW, 1, 1, 1},   // 4-bit, groups of 32
                              {"language_model.model.layers.0.linear_attn.in_proj_z.scales", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.0.linear_attn.in_proj_z.biases", NS_RAW, 1, 1, 1}};
        const NsRaw br[8] = {{"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4}, {"BF16", 2, {1, 2}, 4},
                             {"U32", 2, {1, 16}, 64}, {"BF16", 2, {1, 4}, 8}, {"BF16", 2, {1, 4}, 8}};
        mkdir("out/test_moe_qbase", 0755);
        mkdir("out/test_moe_qbanks", 0755);
        CHECK(ns_write_raw("out/test_moe_qbase", bs, br, 8, 1u << 20, NULL, seq_fill, NULL, berr, sizeof berr) == 0, "base: %s", berr);
        f = fopen("out/test_moe_qbase/config.json", "wb");
        fputs("{\"model_type\": \"qwen4_exp\", \"hidden_size\": 8, \"moe_intermediate_size\": 8, \"num_experts\": 2, \"num_hidden_layers\": 1, "
              "\"model_file\": \"nanoseedlm_qwen4.py\", \"quantization\": {\"group_size\": 32, "
              "\"language_model.model.layers.0.mlp.switch_mlp.up_proj\": {\"bits\": 4}}}", f);
        fclose(f);
        f = fopen("out/test_moe_qbase/LICENSE", "wb");
        fputs("license", f);
        fclose(f);
        static const char* kq[3] = {"gate_proj", "up_proj", "down_proj"};
        char bp[256];
        for (int l = 0; l < 2; ++l)
            for (int q = 0; q < 3; ++q) {
                snprintf(bp, sizeof bp, "out/test_moe_qbanks/L%d_%s.blkt", l, kq[q]);
                f = fopen(bp, "wb");
                fwrite(&h, sizeof h, 1, f); fwrite(rel, 4, 2 * E, f); fwrite(words, 4, nw * E * R, f); fwrite(sc, 2, E * R * NB, f);
                fclose(f);
            }
        CHECK(nslm_qwen4_pack_tq("out/test_moe_qbase", "out/test_moe_qbanks", NULL, NULL, "out/test_moe_qpack", 1u << 20, berr, sizeof berr) == 0,
              "pack: %s", berr);
        CHECK(ns_open(&m, "out/test_moe_qpack", berr, sizeof berr) == 0 && m.n == 4 && ns_add_banks(&m, "out/test_moe_qpack", berr, sizeof berr) == 0,
              "packed folder: %s", berr);
        t = ns_find(&m, "language_model.model.layers.0.linear_attn.out_proj.weight");
        const NsTensor* hu = ns_find(&m, "mtp.layers.0.mlp.switch_mlp.up_proj.weight");
        CHECK(t && t->s[0].len == 4 && t->s[0].p[0] == 0 && t->s[0].p[3] == 3 && hu && hu->enc == NS_TCQ && m.n == 10,
              "packed tensors: %d, the head's up_proj %s", m.n, hu ? "trellis" : "missing");
        ns_close(&m);
        FILE* cf = fopen("out/test_moe_qpack/config.json", "rb");
        char cb[1024] = "";
        if (cf) { cb[fread(cb, 1, sizeof cb - 1, cf)] = 0; fclose(cf); }
        CHECK(!strstr(cb, "model_file") && !strstr(cb, "switch_mlp") && strstr(cb, "group_size") && strstr(cb, "qwen4_exp"), "config: %s", cb);
        cf = fopen("out/test_moe_qpack/LICENSE", "rb");
        CHECK(cf != NULL, "LICENSE not copied");
        if (cf) fclose(cf);
        // --dense8: in_proj_z from an 8-bit folder in one group of 128, packed as two groups of 64 with the same weights
        const NsSpec ds[3] = {{"language_model.model.layers.0.linear_attn.in_proj_z.weight", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.0.linear_attn.in_proj_z.scales", NS_RAW, 1, 1, 1},
                              {"language_model.model.layers.0.linear_attn.in_proj_z.biases", NS_RAW, 1, 1, 1}};
        const NsRaw dr[3] = {{"U32", 2, {1, 32}, 128}, {"BF16", 2, {1, 1}, 2}, {"BF16", 2, {1, 1}, 2}};
        mkdir("out/test_moe_q8", 0755);
        CHECK(ns_write_raw("out/test_moe_q8", ds, dr, 3, 1u << 20, NULL, seq_fill, NULL, berr, sizeof berr) == 0, "8-bit folder: %s", berr);
        f = fopen("out/test_moe_q8/config.json", "wb");
        fputs("{\"quantization\": {\"group_size\": 64, \"bits\": 8, \"language_model.model.layers.0.linear_attn.in_proj_z\": "
              "{\"group_size\": 128, \"bits\": 8}}}", f);
        fclose(f);
        CHECK(nslm_qwen4_pack_tq("out/test_moe_qbase", "out/test_moe_qbanks", NULL, "out/test_moe_q8", "out/test_moe_qpack8", 1u << 20, berr,
                                 sizeof berr) == 0, "pack with --dense8: %s", berr);
        CHECK(ns_open(&m, "out/test_moe_qpack8", berr, sizeof berr) == 0, "packed --dense8 folder: %s", berr);
        t = ns_find(&m, "language_model.model.layers.0.linear_attn.in_proj_z.weight");
        CHECK(t && t->enc == NS_Q8 && t->group == 64 && t->rows == 1 && t->cols == 128, "in_proj_z: enc %d group %d cols %d", t ? t->enc : -1,
              t ? t->group : 0, t ? t->cols : 0);
        if (t && t->enc == NS_Q8) {   // seq_fill: weight bytes 0..127, the scale's bytes 0, 1 and the bias's 0, 1 (BF16 0x0100)
            NsRows* r8 = (NsRows*) malloc(sizeof *r8);
            ns_rows_init(r8);
            float w8[128];
            ns_row_f32(r8, t, 0, 0, 0, w8);
            const float sb = nslm_bf2f(0x0100);
            int ok = 1;
            for (int c = 0; c < 128; ++c) ok &= w8[c] == sb * (float) c + sb;
            CHECK(ok, "in_proj_z's weights differ from the 8-bit folder's");
            free(r8);
        }
        ns_close(&m);
        cf = fopen("out/test_moe_qpack8/config.json", "rb");
        cb[0] = 0;
        if (cf) { cb[fread(cb, 1, sizeof cb - 1, cf)] = 0; fclose(cf); }
        CHECK(strstr(cb, "\"language_model.model.layers.0.linear_attn.in_proj_z\": {") && strstr(cb, "\"bits\": 8"), "config: %s", cb);
        remove("out/test_moe_qbanks/L1_down_proj.blkt");   // a missing bank: refused
        CHECK(nslm_qwen4_pack_tq("out/test_moe_qbase", "out/test_moe_qbanks", NULL, NULL, "out/test_moe_qpack2", 1u << 20, berr, sizeof berr) != 0 &&
              strstr(berr, "L1_down_proj"), "a missing head bank: %s", berr);
        free(nr); free(words); free(y);
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
