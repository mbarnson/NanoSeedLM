// CUDA model loading: unsupported encodings must fail before format / kernel table indexing.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "engine_api.h"
#include "model_st.h"
#include "mova_cfg.h"
#include "mova_ckpt.h"

static const char* CONFIG =
    "{\"model_type\":\"k2_horizon\",\"head_dim\":128,\"hidden_size\":256,"
    "\"intermediate_size\":512,\"layernorm_num_groups\":2,\"mlp_only_layers\":[0],"
    "\"moe_intermediate_size\":128,\"mova_num_experts\":4,\"mova_num_experts_per_tok\":2,"
    "\"num_attention_heads\":4,\"num_key_value_heads\":1,\"num_experts\":4,\"num_experts_per_tok\":2,"
    "\"num_hidden_layers\":2,\"num_shared_experts\":1,\"vocab_size\":64,"
    "\"rms_norm_eps\":1e-6,\"router_scaling_factor\":1,"
    "\"attention_gate_func\":\"softplus\",\"moe_gate_bias\":true,\"query_key_norm\":false,"
    "\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":10000}}";

static int fill(void* ctx, int ti, int s, uint8_t* dst, uint64_t len) {
    (void) ctx; (void) ti; (void) s;
    memset(dst, 0, (size_t) len);
    return 0;
}

static int write_model(const char* dir, const char* p8, char* err, int errlen) {
    mkdir("out", 0755);
    mkdir("out/test", 0755);
    mkdir(dir, 0755);
    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    fputs(CONFIG, f);
    fclose(f);
    MovaCfg cfg;
    if (mova_cfg_parse(&cfg, CONFIG, err, errlen)) return -1;
    MovaTensor* mt = NULL;
    const int n = mova_tensors(&cfg, &mt);
    NsSpec* sp = calloc((size_t) n, sizeof *sp);
    for (int i = 0; i < n; ++i)
        sp[i] = (NsSpec) {mt[i].name, p8 && !strcmp(mt[i].name, p8) ? NS_SEED6P8 : NS_BF16,
                          mt[i].slices, mt[i].rows, mt[i].cols};
    const int rc = ns_write(dir, sp, n, 64ull << 20, NULL, fill, NULL, err, errlen);
    free(sp); free(mt);
    return rc;
}

int main(void) {
    char err[512] = "";
    const char* dir = "out/test/cuda_formats";
    EngOpts o = {0};
    o.model_dir = dir; o.max_seqs = 1; o.kv_tokens = 64;
    if (write_model(dir, NULL, err, sizeof err)) { printf("FAIL: fixture: %s\n", err); return 1; }
    Eng* e = eng_open(&o, err, sizeof err);
    if (!e && strstr(err, "no CUDA device")) { printf("SKIP: %s\n", err); return 77; }
    if (!e) { printf("FAIL: BF16 control: %s\n", err); return 1; }
    eng_close(e);
    const char* p8[] = {"model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.experts.gate_proj.weight"};
    for (int i = 0; i < 2; ++i) {
        if (write_model(dir, p8[i], err, sizeof err)) { printf("FAIL: P8 fixture: %s\n", err); return 1; }
        err[0] = 0;
        e = eng_open(&o, err, sizeof err);
        if (e || !strstr(err, p8[i]) || !strstr(err, "encoding 5 not supported by the CUDA engine")) {
            printf("FAIL: P8 load: %s\n", e ? "opened unsupported format" : err);
            eng_close(e);
            return 1;
        }
        printf("rejected: %s\n", err);
    }
    printf("test_engine_formats: PASS\n");
    return 0;
}
