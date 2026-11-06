// harness/sample.h - one token from a logits row, for nslm-chat and nslm-serve.  temp <= 0: the arg max (ties: the
// lowest id).  mlx (nslm-chat): p ~ exp((l - max) / temp) over the vocabulary in id order, as MLX samples.  Else
// (nslm-serve): the candidates by probability, cut by top_p, top_k and min_p (x the top p).
#pragma once
#include <stdint.h>

#include "qwen4_ext.h"

typedef struct { double temp, top_p, min_p; int top_k, mlx; uint64_t rng; } Sampler;

uint64_t sample_rand(uint64_t* s);   // splitmix64
// draft < 0: a sample of p.  draft >= 0, drawn from q (speculative sampling, Leviathan et al. 2023): the draft with
// probability min(1, p(draft) / q(draft)), else a sample of max(0, p - q).  So each token has the distribution p, for
// any q.  q NULL: one point at the draft (the draft with probability p(draft), else a sample of p without it).
int32_t sample_pick(Sampler* s, const float* l, int V, int32_t draft, const EngDraftQ* q);
int32_t sample_engpick(void* s, const float* l, int V, int32_t draft, const EngDraftQ* q);   // as an EngPick
