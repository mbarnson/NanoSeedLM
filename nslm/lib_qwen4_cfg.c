// nslm/lib_qwen4_cfg.c - the qwen4_exp text configuration and the PLE n-gram rows (nslm/qwen4_cfg.h).
#include "qwen4_cfg.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int64_t num(const Json* o, const char* key, int64_t dflt) {
    const Json* v = json_get(o, key);
    return v && v->t == J_INT ? v->i : v && v->t == J_NUM ? (int64_t) v->d : dflt;
}
static double flt(const Json* o, const char* key, double dflt) {
    const Json* v = json_get(o, key);
    return v && v->t == J_NUM ? v->d : v && v->t == J_INT ? (double) v->i : dflt;
}
static int str_is(const Json* o, const char* key, const char* dflt, const char* want) {
    const char* s = json_gets(o, key);
    return !strcmp(s ? s : dflt, want);
}

int qwen4_cfg_parse(Qwen4Cfg* c, const char* json, char* err, int errlen) {
    memset(c, 0, sizeof *c);
    Json* j = json_parse(json, strlen(json), err, errlen);
    if (!j) return -1;
    const Json* t = json_get(j, "text_config") ? json_get(j, "text_config") : j;
    const Json* rope = json_get(t, "rope_parameters");
    int rc = -1;
#define NEED(ok, ...) do { if (!(ok)) { snprintf(err, (size_t) errlen, __VA_ARGS__); goto done; } } while (0)
    NEED(json_gets(t, "model_type") && !strncmp(json_gets(t, "model_type"), "qwen4_exp", 9), "not a qwen4_exp configuration");
    NEED(str_is(t, "hidden_act", "silu", "silu") && str_is(t, "output_gate_type", "sigmoid", "sigmoid"), "unsupported activation");
    NEED(!json_truthy(json_get(t, "tie_word_embeddings")) && !json_truthy(json_get(t, "attention_bias")), "tied head or attention bias");
    NEED(!json_get(t, "norm_topk_prob") || json_truthy(json_get(t, "norm_topk_prob")), "norm_topk_prob must be true");
    NEED(!rope || str_is(rope, "rope_type", str_is(rope, "type", "default", "default") ? "default" : "x", "default"),
         "only default (M)RoPE is supported");
    c->n_layer = (int) num(t, "num_hidden_layers", 0); c->d = (int) num(t, "hidden_size", 0); c->vocab = (int) num(t, "vocab_size", 0);
    const Json* eos = json_get(t, "eos_token_id");
    c->eos = (int) (eos && eos->t == J_ARR && eos->n ? eos->v[0]->i : num(t, "eos_token_id", -1));
    c->hc = (int) num(t, "hc_count", 4); c->hc_rank = (int) num(t, "hc_lowrank", 320);
    c->n_head = (int) num(t, "num_attention_heads", 0); c->n_kv = (int) num(t, "num_key_value_heads", 0);
    c->head_dim = (int) num(t, "head_dim", c->n_head ? c->d / c->n_head : 0);
    const double prf = flt(rope ? rope : t, "partial_rotary_factor", flt(t, "partial_rotary_factor", 0.25));
    c->rot_dim = (int) (c->head_dim * prf);
    c->rope_theta = (float) flt(rope ? rope : t, "rope_theta", 10000000.0); c->eps = (float) flt(t, "rms_norm_eps", 1e-6);
    c->ix_heads = (int) num(t, "indexer_n_heads", 4); c->ix_dim = (int) num(t, "indexer_head_dim", 128);
    c->ix_budget = (int) num(t, "indexer_budget", 2048); c->ix_ratio = (int) num(t, "indexer_compress_ratio", 4);
    c->lin_hk = (int) num(t, "linear_num_key_heads", 0); c->lin_hv = (int) num(t, "linear_num_value_heads", 0);
    c->lin_dk = (int) num(t, "linear_key_head_dim", 0); c->lin_dv = (int) num(t, "linear_value_head_dim", 0);
    c->lin_conv = (int) num(t, "linear_conv_kernel_dim", 4);
    c->n_exp = (int) num(t, "num_experts", 0); c->top_k = (int) num(t, "num_experts_per_tok", 0);
    c->ff_exp = (int) num(t, "moe_intermediate_size", 0); c->ff_shared = (int) num(t, "shared_expert_intermediate_size", 0);
    c->ple_dim = (int) num(t, "ple_embed_dim", c->d); c->ple_conv = (int) num(t, "ple_conv_kernel_size", 4);
    c->ngram = (int) num(t, "ngram_size", 3); c->heads_per_ngram = (int) num(t, "heads_per_ngram", 8);
    c->ngram_parts = (int) num(t, "split_ngram_parts", 128);
    c->ngram_base = num(t, "ngram_vocab_size_base", 20000000); c->ngram_div = num(t, "make_ngram_vocab_size_divisible_by", 128);
    c->seed = num(t, "seed", 1234);
    NEED(c->n_layer > 0 && c->n_layer <= QWEN4_MAX_LAYERS && c->d > 0 && c->vocab > 0 && c->n_head > 0 && c->n_kv > 0 &&
         c->n_head % c->n_kv == 0 && c->head_dim > 0 && c->rot_dim % 2 == 0 && c->rot_dim <= c->head_dim, "bad attention geometry");
    NEED(c->hc > 1 && c->hc_rank > 0 && num(t, "indexer_kv_heads", 1) == 1 && c->ix_ratio > 0 && c->ix_budget % c->ix_ratio == 0 &&
         c->ix_dim >= c->rot_dim, "bad hyper-connection or indexer geometry");
    NEED(c->lin_hk > 0 && c->lin_hv % c->lin_hk == 0 && c->lin_dk > 0 && c->lin_dv > 0 && c->lin_conv > 0, "bad Gated DeltaNet geometry");
    NEED(c->n_exp > 0 && c->top_k > 0 && c->top_k <= c->n_exp && c->ff_exp > 0 && c->ff_shared > 0, "bad MoE geometry");
    const int heads = (c->ngram - 1) * c->heads_per_ngram;
    NEED(c->ngram >= 2 && c->ngram <= QWEN4_MAX_NGRAM && heads <= QWEN4_MAX_PLE_HEADS && c->ple_dim % heads == 0 && c->ple_conv > 0 &&
         c->ngram_div > 0 && c->ngram_parts > 0 && c->ngram_base > 2, "bad PLE n-gram geometry");
    const Json* lt = json_get(t, "layer_types");
    const int interval = (int) num(t, "full_attention_interval", 4);
    NEED(!lt || (lt->t == J_ARR && lt->n == c->n_layer), "layer_types must list every layer");
    for (int l = 0; l < c->n_layer; ++l) {
        const char* k = lt ? lt->v[l]->s : NULL;
        NEED(!lt || (k && (!strcmp(k, "linear_attention") || !strcmp(k, "full_attention") || !strcmp(k, "qwen_sparse_attention"))),
             "layer %d: unsupported layer type", l);
        c->attn[l] = lt ? strcmp(k, "linear_attention") != 0 : (l + 1) % interval == 0;
        c->ple[l] = -1;
    }
    const Json* pl = json_get(t, "ple_layer_ids");
    for (int i = 0, n = 0; pl && pl->t == J_ARR && i < pl->n; ++i) {   // one-indexed; sorted and unique, as config.py makes them
        const int64_t id = pl->v[i]->i;
        NEED(pl->v[i]->t == J_INT && id >= 1 && id <= c->n_layer && !c->attn[id - 1] && c->ple[id - 1] < 0 && (i == 0 || id > pl->v[i - 1]->i),
             "bad ple_layer_ids");
        c->ple[id - 1] = (int8_t) n++;
    }
    NEED(c->eos >= 0 || !pl || !pl->n, "eos_token_id is required with PLE");
    rc = 0;
done:
#undef NEED
    json_free(j);
    return rc;
}

int qwen4_cfg_load(Qwen4Cfg* c, const char* model_dir, char* err, int errlen) {
    char path[2048];
    snprintf(path, sizeof path, "%s/config.json", model_dir);
    FILE* f = fopen(path, "rb");
    if (!f) { snprintf(err, (size_t) errlen, "%s: cannot open", path); return -1; }
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* s = (char*) malloc((size_t) n + 1);
    s[fread(s, 1, (size_t) n, f)] = 0;
    fclose(f);
    const int rc = qwen4_cfg_parse(c, s, err, errlen);
    free(s);
    return rc;
}

// ---- PLE n-grams -------------------------------------------------------------------------------------------------------

#define GAMMA 0x9E3779B97F4A7C15ull
static uint64_t splitmix64(uint64_t v) {
    v += GAMMA;
    v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
    v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
    return v ^ (v >> 31);
}
static int is_prime(int64_t v) {
    if (v < 2) return 0;
    if (v % 2 == 0) return v == 2;
    for (int64_t d = 3; d * d <= v; d += 2) if (v % d == 0) return 0;
    return 1;
}

void qwen4_ngram_init(const Qwen4Cfg* c, int ple_index, Qwen4Ngram* g) {
    memset(g, 0, sizeof *g);
    g->heads = (c->ngram - 1) * c->heads_per_ngram;
    const uint64_t half = (uint64_t) ((INT64_MAX / (c->vocab > 0 ? c->vocab : 1)) / 2);
    const uint64_t base = (uint64_t) c->seed + 10007ull * (uint64_t) ple_index;
    for (int p = 0; p < c->ngram; ++p) g->mult[p] = (int64_t) (2 * (splitmix64(base + GAMMA * (uint64_t) (p + 1)) % (half ? half : 1)) + 1);
    int64_t prime = c->ngram_base - 1, total = 0;
    for (int k = 0; k < (ple_index + 1) * g->heads; ++k) {   // head h of PLE layer i: the (i * heads + h + 1)-th prime >= base
        for (++prime; !is_prime(prime); ++prime) {}
        if (k < ple_index * g->heads) continue;
        const int h = k - ple_index * g->heads;
        g->size[h] = prime;
        g->off[h] = total;
        total += prime;
    }
    g->rows = (total + c->ngram_div - 1) / c->ngram_div * c->ngram_div;
}

void qwen4_ngram_rows(const Qwen4Cfg* c, const Qwen4Ngram* g, const int32_t* ids, int t, int64_t* rows) {
    int64_t tok[QWEN4_MAX_NGRAM];
    tok[0] = ids[t];   // the token itself, EOS or not; further back, everything at or past an EOS is EOS
    for (int p = 1; p < c->ngram; ++p) tok[p] = t - p < 0 || (p > 1 && tok[p - 1] == c->eos) ? c->eos : ids[t - p];
    for (int n = 2; n <= c->ngram; ++n) {
        uint64_t mixed = (uint64_t) (tok[0] * g->mult[0]);
        for (int p = 1; p < n; ++p) mixed ^= (uint64_t) (tok[p] * g->mult[p]);
        for (int j = 0; j < c->heads_per_ngram; ++j) {
            const int h = (n - 2) * c->heads_per_ngram + j;
            rows[h] = g->off[h] + (int64_t) (mixed % (uint64_t) g->size[h]);
        }
    }
}
