// engine/mova_ext.h - MoVA-engine extensions to engine_api.h: router-selection capture and per-kernel GPU timing.
#pragma once
#include <stdint.h>

#include "engine_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// While on, every forward row records its MLP and value-expert selections and router scores per sparse layer.
// eng_mova_routes_read copies, for the last `rows` rows forwarded, [rows][n_sparse][top_k] MLP expert ids,
// [rows][n_sparse][top_kv] value expert ids, and the selection scores (sigmoid + bias) of all experts
// [rows][n_sparse][n_exp] and [rows][n_sparse][n_vexp] (any pointer may be NULL).
int eng_mova_routes(Eng* e, int on, int max_rows);
int eng_mova_routes_read(Eng* e, int rows, int32_t* mlp, int32_t* val, float* mlp_sel, float* val_sel);

// MLA calibration capture (the weighting of the seed search over the MLA projections, nslm-moe --scope mla): while on,
// every forward adds, per layer, each column's sum of squares over its rows of the projections' inputs, and prompt
// rows attend in latent space (so the latent outputs exist).  eng_mla_capture_read copies layer l's sums: x [d] (the
// attention input: kv_a_x, k_rope_proj), v [n_kv * head_dim] (the value experts' output on MoVA layers: kv_a_v; zeros
// on dense layers), q [n_head * head_dim] (the heads' queries: q_rope_mix, q_lat), o [n_head * r] (the latent
// outputs: v_up), and the rows summed.  0, or -1 (not an MLA model, or an engine without it).
// eng_mla_capture(e, 2) also sums each input's X^T X (the GPTQ seed search; on the GPU, 4 (d^2 + kvd^2 + n_head
// (head_dim^2 + r^2)) bytes a layer); eng_mla_capture_xtx copies layer l's, symmetric, f32: site 0 x [d][d], 1 v [kvd][kvd]
// (zeros on dense layers), 2 q [n_head][head_dim][head_dim] (each head's query), 3 o [n_head][r][r] (each head's latent
// output).  -1 when not captured (or an engine without it).
int eng_mla_capture(Eng* e, int on);
int eng_mla_capture_read(Eng* e, int l, double* x, double* v, double* q, double* o, int64_t* rows);
int eng_mla_capture_xtx(Eng* e, int l, int site, float* h);

// Per-kernel GPU timing: while on, forwards run one command buffer per kernel group
// and accumulate GPU seconds per group; eng_mova_timing_read copies MOVA_TG_N doubles.
enum { MOVA_TG_EMBED_NORM, MOVA_TG_ATTN_PROJ, MOVA_TG_VALUES, MOVA_TG_ATTN, MOVA_TG_ROUTER, MOVA_TG_EXPERTS, MOVA_TG_SHARED,
       MOVA_TG_HEAD, MOVA_TG_N };
int eng_mova_timing(Eng* e, int on);
int eng_mova_timing_read(Eng* e, double* seconds);

#ifdef __cplusplus
}
#endif
