# Trellis-coded experts

A trellis code stores the experts of Qwen3.8-Flash-Next as a bit stream instead of one code for each weight. Each
weight reads a short window of the stream, and a fixed function turns the window into a value. Neighbouring weights
share most of their window bits, so the encoder must choose all the bits of a block together. A Viterbi search does
that. At each size from 3 to 4 bits for each weight, trellis-coded experts come closer to the original model than
Unsloth's packs and our seeds. The GPU computes each weight from its window, as it does for seeds. On an RTX 4080, a
trellis mix with P3's quality decodes 27% faster than P3, because its experts are smaller.

## Key numbers

KL against BF16 in llama.cpp, with the experts emulated in Unsloth's Q8_0 shards. The last column interpolates
Unsloth's two nearest packs in log KL. Read [Trellis packs and Unsloth's curve](../results/2026-11-05-trellis-mixes.md).

| Experts | Bits for each expert weight | Expert GB | Mean KLD | Unsloth's curve at those bits |
|---|---|---|---|---|
| Trellis K = 4, L = 12 | 4.125 | 62.3 | 0.0398 | 0.0480 |
| P4 seeds | 4.50 | 68.0 | 0.0408 | 0.0403 |
| P3 seeds | 4.00 | 60.4 | 0.0600 | 0.0509 |
| Trellis K = 3 + 4 (`down_proj` at K = 4) | 3.458 | 52.2 | 0.0619 | 0.0685 |
| Trellis K = 3, L = 12 | 3.125 | 47.2 | 0.0765 | 0.0916 |
| Trellis K = 2 + 3 (used experts at K = 3) | 2.465 | 37.2 | 0.164 | (no Unsloth pack this small) |
| Trellis K = 2, L = 12 | 2.133 | 32.2 | 0.204 | (no Unsloth pack this small) |

Decode and prompt in the engine on the RTX 4080, with the automatic expert cache. Read
[Trellis experts on the RTX 4080](../results/2026-11-05-trellis-speed-4080.md).

| Experts | Decode, tokens per second | Prompt of 2,994 tokens, tokens per second |
|---|---|---|
| P3 seeds | 39.1 | 642 |
| Trellis K = 4 | 35.7 (-8%) | 608 |
| Trellis K = 3 + 4 (`down_proj` at K = 4) | 49.7 (+27%) | 781 (+22%) |
| Trellis K = 3 | 56.0 (+43%) | 1,036 (+61%) |

On the M4 Max all experts stay in unified memory. There, every trellis pack decodes about 2% faster than P3 and reads
prompts at P3's speed. Read [Trellis experts on the M4 Max](../results/2026-11-05-trellis-speed-m4max.md).

Speed of one expert matvec call (gate and up, one token, 10 experts). These are kernel benches, not decode in the
engine.

| Machine | P3 seeds | Trellis K = 3 | Trellis K = 4 |
|---|---|---|---|
| RTX 4080, weights in VRAM | 49 us | 29 us | 33 us |
| M4 Max | 69.4 us | 66.0 us | 64.3 us |

The CUDA encoder on 8 RTX PRO 6000s encodes all 121 G expert weights in about 5 minutes for each K at L = 12.

## How it works

### The bit stream and the code

A row of weights is cut into blocks of T weights: 256 for `gate_proj` and `up_proj`, 320 for `down_proj`. A block
is a stream of K T + L - K bits. Weight t of the block reads the L-bit window

```
s_t = stream bits [K t, K t + L)            (bit j of s_t is stream bit K t + j)
w_t = scale * q(s_t)
q(s) = (sum of the 4 bytes of x) - 510,   x = 34038481 s + 76625530 (mod 2^32)
```

Each weight adds K new bits and shares L - K bits with its neighbour. So the cost is K bits for each weight, plus
the first window's extra bits and a BF16 scale for each block: K + (L - K + 16) / T bits. The function q is QTIP's
computed "1MAD" code. The multiply and add scramble s like a pseudo-random generator. The sum of 4 nearly uniform
bytes is nearly Gaussian, with mean 0 and standard deviation 147.8. Expert weights are nearly Gaussian too, so no
rotation is needed. A code from our LFSR stream (the byte sum of 32 stream bits) gives 0.6% to 1.7% more error at
L = 16. So the format uses 1MAD, which needs no table.

### The search

Consecutive windows overlap: s_t = (s_{t-1} >> K) | (b_t << (L - K)), with b_t the K new bits. So the windows form a
trellis of 2^L states with 2^K branches for each state. For a block, the encoder finds the bits with the least
imatrix-weighted error, sum_t h_t (w_t - a q(s_t))^2, by a Viterbi search over the trellis:

1. Set the scale a from the block's weighted root mean square.
2. Run the Viterbi search for that scale.
3. Refit a by weighted least squares on the chosen path.
4. Repeat steps 2 and 3 twice more. Round the last scale to BF16.

The search costs about 6 times 2^L operations for each weight. The C encoder (`nslm/lib_trellis.c`) and the CUDA
encoder (`nslm/trellis.cu`) write the same words, bit for bit. They use the same float operations without fused
multiply-adds and the same ties: the lowest dropped bits j, then the lowest last state.

### Decode

A weight costs a 64-bit window read and shift, one integer multiply-add, a byte sum and one multiply-add with the
input. On CUDA the byte sum is one `__dp4a`, which is exact. The CUDA prompt path decodes a tile of a block row into
shared memory, then uses the same tensor-core GEMM as the seeds. The Metal kernels are not merged yet.

### Files

`nslm-moe --codec tq --tq-k K --tq-l L` writes one bank file for each layer and projection
(`L{layer}_{proj}.blkt`). A trellis pack is a model folder with these files beside its shards. Read
[The model folder format](../reference/model-folder.md).

## Measurements

- [Trellis-coded experts: weight error](../results/2026-11-05-trellis-probe.md): one layer, against ggml's types.
- [Trellis-coded experts against BF16](../results/2026-11-05-trellis-kld.md): the whole model in llama.cpp.
- [Trellis packs and Unsloth's curve](../results/2026-11-05-trellis-mixes.md): 2 to 4 bits, mixes, the corrected curve.
- [Trellis experts on the RTX 4080](../results/2026-11-05-trellis-speed-4080.md): decode and prompt against P3.
- [Trellis experts on the M4 Max](../results/2026-11-05-trellis-speed-m4max.md): decode, prompt and the files' sizes.

## See also

- [The trellis plan](../plans/2026-11-05-trellis-experts.md): steps, gates and the family of packs.
- [SeedLM](seedlm.md): the seed format, which the trellis replaces for experts.
- [Related work](../reference/related-work.md): QTIP, trellis-coded quantization and its history.
