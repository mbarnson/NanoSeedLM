// nslm/lib_mova_cfg.c - the MoVA config block and logical tensor list (nslm/mova_cfg.h).
#include "mova_cfg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The value of "key": in a flat JSON text (numbers, true/false, strings); NULL if absent.
static const char* jval(const char* js, const char* key) {
    char pat[96];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(js, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ':') ++p;
    return p;
}
static int jint(const char* js, const char* key, int* v) {
    const char* p = jval(js, key);
    if (!p) return -1;
    *v = atoi(p);
    return 0;
}
static int jflt(const char* js, const char* key, float* v) {
    const char* p = jval(js, key);
    if (!p) return -1;
    *v = (float) strtod(p, NULL);
    return 0;
}
static int jstr_is(const char* js, const char* key, const char* want) {
    const char* p = jval(js, key);
    if (!p || *p != '"') return 0;
    return !strncmp(p + 1, want, strlen(want)) && p[1 + strlen(want)] == '"';
}
static int jbool(const char* js, const char* key) {
    const char* p = jval(js, key);
    return p && !strncmp(p, "true", 4);
}

int mova_cfg_parse(MovaCfg* c, const char* js, char* err, int errlen) {
    memset(c, 0, sizeof *c);
#define NEED(ok, ...) do { if (!(ok)) { snprintf(err, (size_t) errlen, __VA_ARGS__); return -1; } } while (0)
    NEED(!jval(js, "hidden_act") || jstr_is(js, "hidden_act", "silu"), "unsupported hidden_act");
    NEED(!jval(js, "router_score_func") || jstr_is(js, "router_score_func", "sigmoid"), "unsupported router_score_func");
    NEED(!jbool(js, "query_key_norm"), "query_key_norm is not supported");
    NEED(!jbool(js, "tie_word_embeddings"), "tied embeddings are not supported");
    NEED(!jbool(js, "use_sliding_window"), "sliding window is not supported");
    NEED(!jval(js, "attention_gate_func") || jstr_is(js, "attention_gate_func", "softplus"), "unsupported attention gate");
    int shared = 1;
    jint(js, "num_shared_experts", &shared);
    NEED(shared == 1, "num_shared_experts must be 1");
    NEED(!jval(js, "norm_topk_prob") || jbool(js, "norm_topk_prob"), "norm_topk_prob must be true");
    NEED(!jval(js, "moe_gate_bias") || jbool(js, "moe_gate_bias"), "moe_gate_bias must be true");
    NEED(!jval(js, "rope_type") || jstr_is(js, "rope_type", "default"), "only default RoPE is supported");
    int ok = jint(js, "num_hidden_layers", &c->n_layer) | jint(js, "hidden_size", &c->d) |
             jint(js, "num_attention_heads", &c->n_head) | jint(js, "num_key_value_heads", &c->n_kv) |
             jint(js, "head_dim", &c->head_dim) | jint(js, "intermediate_size", &c->ff_dense) |
             jint(js, "moe_intermediate_size", &c->ff_exp) | jint(js, "num_experts", &c->n_exp) |
             jint(js, "num_experts_per_tok", &c->top_k) | jint(js, "mova_num_experts", &c->n_vexp) |
             jint(js, "mova_num_experts_per_tok", &c->top_kv) | jint(js, "vocab_size", &c->vocab) |
             jint(js, "layernorm_num_groups", &c->norm_groups) | jflt(js, "rms_norm_eps", &c->eps) |
             jflt(js, "router_scaling_factor", &c->route_scale) | jflt(js, "rope_theta", &c->rope_theta);
    NEED(!ok, "config.json: a required field is missing");
    int rope_dim = c->head_dim;
    jint(js, "rope_head_dim", &rope_dim);
    NEED(rope_dim == c->head_dim, "partial RoPE is not supported");
    // mlp_only_layers must be 0 .. first_sparse - 1
    const char* m = jval(js, "mlp_only_layers");
    NEED(m && *m == '[', "mlp_only_layers missing");
    c->first_sparse = 0;
    for (const char* p = m + 1; *p && *p != ']'; ++p)
        if (*p >= '0' && *p <= '9') {
            NEED(atoi(p) == c->first_sparse, "mlp_only_layers must be a prefix 0..k-1");
            ++c->first_sparse;
            while (*p >= '0' && *p <= '9') ++p;
            --p;
        }
    int step = 1;
    jint(js, "decoder_sparse_step", &step);
    NEED(step == 1, "decoder_sparse_step must be 1");
    NEED(c->n_vexp > 0 && c->top_kv > 0, "not a MoVA config (no value experts)");
    NEED(c->d % c->norm_groups == 0 && c->d % (64 * c->norm_groups) == 0, "norm groups");
    NEED(c->n_head % c->n_kv == 0 && c->head_dim == 128, "attention layout (head_dim 128)");
    const char* mr = jval(js, "mla_ranks");
    if (mr) {
        NEED(*mr == '[', "mla_ranks must be an array");
        NEED(c->n_layer <= MOVA_MAX_LAYERS, "too many layers for MLA");
        int nr = 0;
        for (const char* p = mr + 1; *p && *p != ']'; ++p)
            if (*p >= '0' && *p <= '9') {
                NEED(nr < c->n_layer, "mla_ranks has more entries than layers");
                c->mla_rank[nr] = atoi(p);
                NEED(c->mla_rank[nr] >= 32 && c->mla_rank[nr] <= 1024 && c->mla_rank[nr] % 32 == 0,
                     "mla_ranks: %d is not a multiple of 32 in [32, 1024]", c->mla_rank[nr]);
                ++nr;
                while (*p >= '0' && *p <= '9') ++p;
                --p;
            }
        NEED(nr == c->n_layer, "mla_ranks has %d entries for %d layers", nr, c->n_layer);
        c->mla_rope = c->head_dim;
        jint(js, "mla_rope_dim", &c->mla_rope);
        NEED(c->mla_rope == c->head_dim, "mla_rope_dim must equal head_dim (128)");
        NEED(c->n_head <= 32, "MLA attention runs one simdgroup per query head (at most 32)");
        c->mla = 1;
    }
    c->router_parts = 2;   // router GEMM partitions: the MLX reference implementation's rounding contract
#undef NEED
    return 0;
}

int mova_cfg_load(MovaCfg* c, const char* model_dir, char* err, int errlen) {
    char path[2048];
    snprintf(path, sizeof path, "%s/config.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (!f) { snprintf(err, (size_t) errlen, "cannot read %s", path); return -1; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* js = (char*) malloc((size_t) n + 1);
    const size_t got = fread(js, 1, (size_t) n, f);
    fclose(f);
    js[got] = 0;
    const int r = mova_cfg_parse(c, js, err, errlen);
    free(js);
    return r;
}

static void add(MovaTensor* t, int* n, int kind, int layer, int slices, int rows, int cols, const char* proj, const char* fmt, ...) {
    MovaTensor* x = &t[(*n)++];
    memset(x, 0, sizeof *x);
    x->kind = kind; x->layer = layer; x->slices = slices; x->rows = rows; x->cols = cols;
    if (proj) snprintf(x->proj, sizeof x->proj, "%s", proj);
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);
    vsnprintf(x->name, sizeof x->name, fmt, ap);
    __builtin_va_end(ap);
}

int mova_tensors(const MovaCfg* c, MovaTensor** out) {
    MovaTensor* t = (MovaTensor*) calloc((size_t) c->n_layer * 24 + 8, sizeof *t);
    int n = 0;
    const int d = c->d, q = c->n_head * c->head_dim, kv = c->n_kv * c->head_dim;
    add(t, &n, MOVA_K_EMBED, -1, 1, c->vocab, d, NULL, "model.embed_tokens.weight");
    for (int l = 0; l < c->n_layer; ++l) {
        const int sparse = l >= c->first_sparse;
        add(t, &n, MOVA_K_NORM, l, 1, 1, d, NULL, "model.layers.%d.input_layernorm.weight", l);
        add(t, &n, MOVA_K_NORM, l, 1, 1, d, NULL, "model.layers.%d.post_attention_layernorm.weight", l);
        add(t, &n, MOVA_K_LINEAR, l, 1, q, d, NULL, "model.layers.%d.self_attn.q_proj.weight", l);
        if (!c->mla) add(t, &n, MOVA_K_LINEAR, l, 1, kv, d, NULL, "model.layers.%d.self_attn.k_proj.weight", l);
        add(t, &n, MOVA_K_LINEAR, l, 1, d, q, NULL, "model.layers.%d.self_attn.o_proj.weight", l);
        add(t, &n, MOVA_K_LINEAR, l, 1, q, d, NULL, "model.layers.%d.self_attn.gate_proj.weight", l);
        if (c->mla) {
            const int r = c->mla_rank[l], hd = c->head_dim, rd = c->mla_rope;
            add(t, &n, MOVA_K_LINEAR, l, 1, r, d, NULL, "model.layers.%d.self_attn.mla.kv_a_x", l);
            if (sparse) add(t, &n, MOVA_K_LINEAR, l, 1, r, kv, NULL, "model.layers.%d.self_attn.mla.kv_a_v", l);
            add(t, &n, MOVA_K_LINEAR, l, 1, rd, d, NULL, "model.layers.%d.self_attn.mla.k_rope_proj", l);
            add(t, &n, MOVA_K_HEADS, l, c->n_head, rd, hd, NULL, "model.layers.%d.self_attn.mla.q_rope_mix", l);
            add(t, &n, MOVA_K_HEADS, l, c->n_head, r, hd, NULL, "model.layers.%d.self_attn.mla.q_lat", l);
            add(t, &n, MOVA_K_HEADS, l, c->n_head, hd, r, NULL, "model.layers.%d.self_attn.mla.v_up", l);
        }
        if (!sparse) {
            if (!c->mla) add(t, &n, MOVA_K_LINEAR, l, 1, kv, d, NULL, "model.layers.%d.self_attn.v_proj.weight", l);
            add(t, &n, MOVA_K_LINEAR, l, 1, c->ff_dense, d, NULL, "model.layers.%d.mlp.gate_proj.weight", l);
            add(t, &n, MOVA_K_LINEAR, l, 1, c->ff_dense, d, NULL, "model.layers.%d.mlp.up_proj.weight", l);
            add(t, &n, MOVA_K_LINEAR, l, 1, d, c->ff_dense, NULL, "model.layers.%d.mlp.down_proj.weight", l);
            continue;
        }
        add(t, &n, MOVA_K_ROUTER, l, 1, c->n_vexp, d, NULL, "model.layers.%d.self_attn.v_router.weight", l);
        add(t, &n, MOVA_K_ROUTER_BIAS, l, 1, 1, c->n_vexp, NULL, "model.layers.%d.self_attn.v_router.bias", l);
        add(t, &n, MOVA_K_VEXPERTS, l, c->n_vexp, kv, d, NULL, "model.layers.%d.self_attn.v_experts.weight", l);
        add(t, &n, MOVA_K_ROUTER, l, 1, c->n_exp, d, NULL, "model.layers.%d.mlp.gate.weight", l);
        add(t, &n, MOVA_K_ROUTER_BIAS, l, 1, 1, c->n_exp, NULL, "model.layers.%d.mlp.gate.bias", l);
        add(t, &n, MOVA_K_EXPERTS, l, c->n_exp, c->ff_exp, d, "gate_proj", "model.layers.%d.mlp.experts.gate_proj.weight", l);
        add(t, &n, MOVA_K_EXPERTS, l, c->n_exp, c->ff_exp, d, "up_proj", "model.layers.%d.mlp.experts.up_proj.weight", l);
        add(t, &n, MOVA_K_EXPERTS, l, c->n_exp, d, c->ff_exp, "down_proj", "model.layers.%d.mlp.experts.down_proj.weight", l);
        add(t, &n, MOVA_K_LINEAR, l, 1, c->ff_exp, d, NULL, "model.layers.%d.mlp.shared_experts.gate_proj.weight", l);
        add(t, &n, MOVA_K_LINEAR, l, 1, c->ff_exp, d, NULL, "model.layers.%d.mlp.shared_experts.up_proj.weight", l);
        add(t, &n, MOVA_K_LINEAR, l, 1, d, c->ff_exp, NULL, "model.layers.%d.mlp.shared_experts.down_proj.weight", l);
    }
    add(t, &n, MOVA_K_NORM, -1, 1, 1, d, NULL, "model.norm.weight");
    add(t, &n, MOVA_K_HEAD, -1, 1, c->vocab, d, NULL, "lm_head.weight");
    *out = t;
    return n;
}

const MovaTensor* mova_find(const MovaTensor* t, int n, const char* name) {
    for (int i = 0; i < n; ++i) if (!strcmp(t[i].name, name)) return &t[i];
    return NULL;
}

const char* mova_slice_name(const MovaTensor* t, int s, char* buf, int len) {
    if (t->kind == MOVA_K_EXPERTS) snprintf(buf, (size_t) len, "model.layers.%d.mlp.experts.%d.%s.weight", t->layer, s, t->proj);
    else if (t->kind == MOVA_K_VEXPERTS) snprintf(buf, (size_t) len, "model.layers.%d.self_attn.v_experts.%d.weight", t->layer, s);
    else snprintf(buf, (size_t) len, "%s", t->name);
    return buf;
}
