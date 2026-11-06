# Plan: trellis-coded dense weights for Qwen3.8-Flash-Next (2026-11-05)

**Goal.** Code the dense weights with the trellis of the experts. Give a tensor kind more bits only where its KL needs
them, as Unsloth's dynamic packs do. Then a decode token reads fewer bytes than with the 4-bit MLX dense weights, and
every pack has a KL measured as it is shipped.

**Status.** Draft. Matt asked for it on 2026-11-05 and approved the HF jobs within the weekly cap.

## Why

- Every KL of [the trellis packs](../results/2026-11-05-trellis-mixes.md) uses Unsloth's Q8_0 dense weights (5.5 GB).
  The engine runs the 4-bit MLX dense weights (affine, group 32: 5.0 bits for each weight). Their KL is not measured.
- On the M4 Max the dense matvecs take 8.77 ms of an 18.7 ms decode token. They read 2.64 GB at 301 GB/s.
- On the RTX 4080 the dense weights stay in VRAM. Fewer bytes leave more VRAM for the expert cache.

## Key numbers

The dense tensors that a text decode token reads (mac-metal's count from the P3 headers):

| Kind | GB | Tensors (rows x cols, count) |
|---|---|---|
| DeltaNet projections | 1.306 | `in_proj_qkv` 10240 x 2560, `in_proj_z` 6144 x 2560, `out_proj` 2560 x 6144, `in_proj_a` and `in_proj_b` 48 x 2560; x 36 |
| Hyper-connections | 0.402 | `input_mix_weight_up` 10240 x 320, `_down` 320 x 10240, `block_inject` 4 x 10240; x 96, and the mixer's |
| `lm_head` | 0.397 | 248320 x 2560 |
| Attention projections | 0.386 | `q_proj` 12288 x 2560, `o_proj` 2560 x 6144, `index_qk_proj` 640 x 2560, `k_proj` and `v_proj` 512 x 2560; x 12 |
| Shared experts | 0.147 | gate and up 640 x 2560, down 2560 x 640; x 48 |
| Routers (Q8, group 64) | 0.067 | `mlp.gate` 512 x 5120, `shared_expert_gate` 1 x 5120; x 48 |

A text decode token does not read the vision tower (0.90 GB, BF16) or the MTP head (1.73 GB, only with drafts). It
reads one row of the embedding (0.40 GB).

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| D0 | The KL of the mix as shipped: its 4-bit MLX dense weights in place of Unsloth's Q8_0 dense weights | cloud (code), HF (KL) | Report only |
| D1 | `nslm-dense --codec tq`: the trellis for a dense matrix, with the imatrix | cloud | DA |
| D2 | The sweep: one kind at K = 3 and at K = 4, the other kinds at Q8_0, with the mix's experts | pc-cuda (encode), HF (KL) | Report only |
| D3 | The allocation: the K of each kind from D2, then the whole pack's KL | cloud, HF (KL) | DB |
| D4 | Dense trellis matvecs and GEMMs on CUDA and Metal, and decode speed | pc-cuda, mac-metal, cloud | DC |

- **DA.** The C, CUDA and Metal encoders write the same bytes, as for the expert banks. At K = 4, the weighted error of
  each kind is below that of its 4-bit MLX weights.
- **DB.** The pack's KL is at most D0's, and a decode token reads at most 2.2 GB of dense weights.
- **DC.** Decode is at least 5% faster on the M4 Max and not slower on the RTX 4080. The text is the same from run to
  run.

## How the emulation works

`nslm-qwen4-emulate` writes the experts into Unsloth's Q8_0 shards now. D0 and D2 add the dense tensors by name:

| Engine (MLX) | llama.cpp |
|---|---|
| `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` | `attn_qkv`, `attn_gate`, `ssm_out` |
| `linear_attn.in_proj_a`, `in_proj_b` | `ssm_alpha`, `ssm_beta` |
| `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` | `attn_q`, `attn_k`, `attn_v`, `attn_output` |
| `mlp.shared_expert.gate_proj`, `up_proj`, `down_proj` | `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` |
| `attn_hc` and `mlp_hc` `input_mix_weight_up`, `_down`, `block_inject` | `hc_attn_up`, `_down`, `_inject` and `hc_ffn_*` |
| `lm_head`, `embed_tokens` | `output`, `token_embd` |

- The emulator decodes each tensor and writes it as Q8_0. It compares each tensor with Unsloth's Q8_0 tensor of that
  name. A relative error above 0.3 means that the rows have another order, and the emulator stops.
- The routers (F32 in Unsloth's Q8_0), the norms, `conv1d` and the indexer stay as they are.

## Hypotheses

| ID | Hypothesis | Gate |
|---|---|---|
| HD1 | The 4-bit MLX dense weights raise the mix's KL by at least 5% against Q8_0 dense weights | D0 |
| HD2 | At K = 4, the dense weights give a KL at most that of the 4-bit MLX ones, for 17.5% fewer bytes | D2, DB |
| HD3 | Decode on the M4 Max is 5% to 8% faster | DC |

## Costs

- D0: one KL job of two packs with the saved BF16 reference, `rtx-pro-6000x2`, about 15 minutes, about $1.50.
- D2: about 10 packs (5 kinds, K = 3 and 4), about 40 minutes on `rtx-pro-6000x2`, about $4. The encode of about
  5 G dense weights takes a few minutes on the RTX 4080.
- D3: one KL job, about $1.
- D4: kernel work on both engines. The gathers of the experts may serve a dense matvec with one expert.

## Risks

- A tensor's rows may have another order in llama.cpp than in MLX. The emulator's check finds it.
- The trellis needs the BF16 weights of the original model, as for the experts. The 4-bit MLX weights are not a source.
- The hyper-connection tensors are 320 columns wide or 4 rows high. Their matvecs wait on latency, not bytes, so the
  trellis may not make them faster.

## See also

- [Trellis-coded experts](2026-11-05-trellis-experts.md), the plan for the experts.
- [Trellis-coded experts](../concepts/trellis.md), the concept page.
