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

// Per-kernel GPU timing: while on, forwards run one command buffer per kernel group
// and accumulate GPU seconds per group; eng_mova_timing_read copies MOVA_TG_N doubles.
enum { MOVA_TG_EMBED_NORM, MOVA_TG_ATTN_PROJ, MOVA_TG_VALUES, MOVA_TG_ATTN, MOVA_TG_ROUTER, MOVA_TG_EXPERTS, MOVA_TG_SHARED,
       MOVA_TG_HEAD, MOVA_TG_N };
int eng_mova_timing(Eng* e, int on);
int eng_mova_timing_read(Eng* e, double* seconds);

#ifdef __cplusplus
}
#endif
