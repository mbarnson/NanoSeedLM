// engine/qwen4_ext.h - qwen4-engine extensions to engine_api.h: decode with the MTP head's drafts and any sampler.
#pragma once
#include <stdint.h>

#include "engine_api.h"

// The distribution q that a draft came from: ids and probabilities (the head's top ENG_DRAFT_Q, at most)
#define ENG_DRAFT_Q 20
typedef struct { int n; int32_t id[ENG_DRAFT_Q]; float p[ENG_DRAFT_Q]; } EngDraftQ;
// The next token from a logits row: draft < 0, a free pick; else the draft, or another token in its place (a rejection).
// q: the draft's distribution (NULL: one point at the draft).  Greedy: the arg max.  Speculative sampling:
// harness/sample.h.  Each token then has the pick's own distribution.  -1: stop (not committed).
typedef int32_t (*EngPick)(void* ctx, const float* logits, int vocab, int32_t draft, const EngDraftQ* q);
#define ENG_MAX_DRAFTS 3
// eng_generate for one sequence, each token by pick: with Q4E_DRAFTS = K (<= ENG_MAX_DRAFTS), each forward verifies up
// to K MTP drafts, drafted on the pending token.  Returns the tokens committed (< n_new after a stop), or -1 (also for
// the MoVA engine).
int eng_generate_pick(struct Eng* e, int seq, int n_new, EngPick pick, void* ctx, int32_t* out, EngStats* st);
