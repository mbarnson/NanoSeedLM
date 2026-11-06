// engine/qwen4_ext.h - qwen4-engine extensions to engine_api.h: decode with the MTP head's drafts and any sampler.
#pragma once
#include <stdint.h>

#include "engine_api.h"

// The next token from a logits row: draft < 0, a free pick; else the draft, or another token in its place (a rejection).
// Greedy: the arg max.  Speculative sampling: harness/sample.h.  Each token then has the pick's own distribution.
typedef int32_t (*EngPick)(void* ctx, const float* logits, int vocab, int32_t draft);
// eng_generate for one sequence, each token by pick: with Q4E_DRAFTS = K, each forward verifies up to K MTP drafts,
// drafted on the pending token.  -1 for the MoVA engine.
int eng_generate_pick(struct Eng* e, int seq, int n_new, EngPick pick, void* ctx, int32_t* out, EngStats* st);
