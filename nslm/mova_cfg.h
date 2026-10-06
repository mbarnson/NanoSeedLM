// nslm/mova_cfg.h - K2-Horizon MoVA: the validated config block and the logical tensor list.
//
// mova_cfg_load accepts only the architecture the engine implements (silu, no q/k norm, default RoPE over the full
// head, softplus attention gate, sigmoid routers with bias and top-k renormalisation, one shared expert, MoVA value
// experts, untied head).
//
// Logical tensors (shared by the packer and the engine) use HF names, except that routed and value experts are stacked
// [E][rows][cols] under one name ("model.layers.L.mlp.experts.gate_proj.weight",
// "model.layers.L.self_attn.v_experts.weight"); mova_slice_name gives each slice's checkpoint name.
#pragma once
#include <stdint.h>

typedef struct {
    int n_layer, d, n_head, n_kv, head_dim, ff_dense, ff_exp, n_exp, top_k, n_vexp, top_kv, vocab, first_sparse;
    int norm_groups, router_parts;
    float rope_theta, eps, route_scale;
} MovaCfg;

int mova_cfg_load(MovaCfg* c, const char* model_dir, char* err, int errlen);
int mova_cfg_parse(MovaCfg* c, const char* json, char* err, int errlen);

enum {
    MOVA_K_NORM,         // RMSNorm weight [1][d]
    MOVA_K_LINEAR,       // a dense projection [rows][cols] (attention, dense MLP, shared expert)
    MOVA_K_ROUTER,       // mlp.gate / self_attn.v_router [n][d]
    MOVA_K_ROUTER_BIAS,  // mlp.gate.bias / self_attn.v_router.bias [1][n] (f32 in use; BF16 in the checkpoint)
    MOVA_K_EXPERTS,      // routed MLP experts, stacked [n_exp][rows][cols]
    MOVA_K_VEXPERTS,     // MoVA value experts, stacked [n_vexp][kv_dim][d]
    MOVA_K_EMBED,        // [vocab][d]
    MOVA_K_HEAD,         // lm_head [vocab][d]
};

typedef struct {
    char name[96];
    int kind, layer, slices, rows, cols;
    char proj[16];       // gate_proj / up_proj / down_proj for experts
} MovaTensor;

// Allocates *out (free it); returns the count.
int mova_tensors(const MovaCfg* c, MovaTensor** out);
const MovaTensor* mova_find(const MovaTensor* t, int n, const char* name);
// The checkpoint name of slice s (for stacked tensors), or the tensor's own name.
const char* mova_slice_name(const MovaTensor* t, int s, char* buf, int len);
