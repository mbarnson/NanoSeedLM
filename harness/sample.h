// harness/sample.h - one token from a logits row, for nslm-chat and nslm-serve.  temp <= 0: the arg max (ties: the
// lowest id).  mlx (nslm-chat): p ~ exp((l - max) / temp) over the vocabulary in id order, as MLX samples.  Else
// (nslm-serve): the candidates by probability, cut by top_p, top_k and min_p (x the top p).
#pragma once
#include <stdint.h>

typedef struct { double temp, top_p, min_p; int top_k, mlx; uint64_t rng; } Sampler;

uint64_t sample_rand(uint64_t* s);   // splitmix64
// draft < 0: a sample of p.  draft >= 0 (speculative sampling with one draft token, Leviathan et al. 2023): the draft
// with probability p(draft), else a sample of p without it.  So each token has the distribution p, whatever the draft.
int32_t sample_pick(Sampler* s, const float* l, int V, int32_t draft);
