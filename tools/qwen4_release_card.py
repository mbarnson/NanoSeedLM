#!/usr/bin/env python3
"""tools/qwen4_release_card.py - the model cards of the trellis release (docs/plans/2026-11-05-release-family.md), one
README per pack, with the KL as shipped read from tools/hf_release_pack.sh's llama-perplexity logs.

  python3 tools/qwen4_release_card.py KLD_DIR OUT_DIR     (KLD_DIR: the job's OUT/kld, with _m_rel_NAME.log)
"""
import os, re, sys

REPO = "txgsync/Qwen3.8-Flash-Next-NanoSeedLM-"
# name: experts, expert bits, expert GB, resident GB (PLE on disk), KL with Q8_0 dense (trellis mixes page), machines,
#       RTX 4080 decode and M4 Max decode with 4-bit dense (tok/s), the K of each projection
PACKS = {
    "TQ4": ("trellis K = 4", 4.125, 62.3, 70, 0.0398, "A 128 GB Mac. A GPU of 24 GB or more with 96 to 128 GB of RAM",
            "35.7 (at the old 2,560 MB reserve)", "48.9", "K = 4 for `gate_proj`, `up_proj` and `down_proj`"),
    "TQ3-D4": ("trellis K = 3, `down_proj` at K = 4", 3.458, 52.2, 60, 0.0619,
               "A 16 GB GPU with 96 to 128 GB of RAM. A 96 GB Mac", "54.4", "48.8",
               "K = 3 for `gate_proj` and `up_proj`, K = 4 for `down_proj`"),
    "TQ3": ("trellis K = 3", 3.125, 47.2, 55, 0.0765, "A 16 GB GPU with 64 GB of RAM. A 64 GB Mac", "61.3", "48.8",
            "K = 3 for `gate_proj`, `up_proj` and `down_proj`"),
}
UNSLOTH = [("UD-Q4_K_XL", 5.101, 0.0304), ("UD-IQ4_XS", 3.999, 0.0509), ("UD-Q3_K_XL", 3.720, 0.0593),
           ("UD-IQ3_XXS", 3.331, 0.0735), ("UD-Q2_K_XL", 3.049, 0.0994)]


def kld(path):
    """Mean KLD, its error, median, 99th percentile, top-1 agreement and perplexity from a llama-perplexity log."""
    t = open(path).read()
    g = lambda p: re.search(p, t).group(1)
    return dict(mean=float(g(r"Mean    KLD:\s+([\d.]+)")), err=float(g(r"Mean    KLD:\s+[\d.]+ ±\s+([\d.]+)")),
                med=float(g(r"Median  KLD:\s+([\d.]+)")), p99=float(g(r"99.0%   KLD:\s+([\d.]+)")),
                top1=float(g(r"Same top p:\s+([\d.]+)")), ppl=float(g(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")))


def card(name, k, ks):
    experts, bits, egb, mem, kq8, who, s4080, smac, kdesc = PACKS[name]
    fam = "\n".join(f"| [{n}](https://huggingface.co/{REPO}{n}) | {PACKS[n][0]} | {PACKS[n][1]:.3f} | {PACKS[n][2]} | "
                    f"about {PACKS[n][3]} GB | {ks[n]['mean']:.4f} | {PACKS[n][5]} |" for n in PACKS if n in ks)
    uns = "\n".join(f"| Unsloth {n} | {b:.3f} | {v:.4f} |" for n, b, v in UNSLOTH)
    return f"""---
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- qwen4_exp
- qwen3.8-flash-next
- moe
- trellis
- nanoseedlm
- nslm
- cuda
- metal
- research-preview
---

# Qwen3.8-Flash-Next-NanoSeedLM-{name}

> **Research preview.** The routed experts of this model are trellis-coded: {kdesc}, with 12-bit states. It runs in the
> [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM) engines (CUDA on Linux and Windows, Metal on macOS). mlx-vlm
> and oMLX cannot load it yet.

Qwen3.8-Flash-Next-NanoSeedLM-{name} is [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
with its 120.8 B routed expert weights trellis-coded at {bits:.3f} bits for each weight ({egb} GB). Each weight is a
code of a 12-bit window of a bit stream, and neighbouring weights share bits. No codebook is stored: a fixed
multiply-add of the window gives the value. The dense weights are 8-bit MLX, from
[mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp). The
PLE n-gram table (32 GB, 4-bit MLX) stays on the disk, and the page cache keeps the rows that a run reads. Read
[Trellis-coded experts](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/concepts/trellis.md).

## The family

Memory counts the weights that stay resident. The PLE table is on the disk.

| Pack | Experts | Bits for each expert weight | Expert GB | Memory | KL against BF16 | For |
|---|---|---|---|---|---|---|
{fam}

## Quality

KL divergence against BF16 in llama.cpp, on 64 windows of 2,048 tokens of wikitext-2. The pack's experts and dense
weights are written into Unsloth's Q8_0 GGUF, so llama.cpp runs this pack's weights. The PLE table is Unsloth's Q8_0.

| | Mean KLD | Median KLD | 99th percentile KLD | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|
| This pack | {k['mean']:.4f} ± {k['err']:.4f} | {k['med']:.4f} | {k['p99']:.3f} | {k['top1']:.2f}% | {k['ppl']:.3f} |
| BF16 | 0 | 0 | 0 | 100% | 3.802 |

Unsloth's packs measured in the same way (their own dense weights):

| Pack | Bits for each expert weight | Mean KLD |
|---|---|---|
| This pack | {bits:.3f} | {k['mean']:.4f} |
{uns}

With Unsloth's Q8_0 dense weights, these experts give {kq8:.4f}. Read
[the measurements](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/results/2026-11-05-trellis-mixes.md).

## Speed

Decode in tokens per second, the code prompt, 256 tokens, with the 4-bit MLX dense weights of the first builds. The
8-bit dense weights of this release read 1.7 times the dense bytes, so decode may be slower. We will add the numbers
of this release.

| Machine | Decode |
|---|---|
| RTX 4080 (16 GB), automatic expert cache | {s4080} |
| M4 Max (128 GB) | {smac} |

## How to run

Build [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM) ([Build](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/build.md)),
then download this folder:

```bash
hf download {REPO}{name} --local-dir Qwen3.8-Flash-Next-NanoSeedLM-{name}
```

Chat, or serve an OpenAI-compatible API (CUDA: `build/bin`, macOS: `out/bin`):

```bash
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-{name} "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-{name} --ctx 32768 --port 8080
```

On a GPU, the routed experts stay in host memory, and a cache keeps the used ones in VRAM. Read
[Run and serve](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/guides/run-and-serve.md).

## Limits

- This is a research preview. Its output is not identical to the output of the original model.
- mlx-vlm and oMLX cannot load the trellis experts yet. The MLX loader has no trellis kernels.
- We did not test image or video input.
- The PLE table's 4-bit MLX format is not measured against BF16. The KL above uses Unsloth's Q8_0 table.
- The numbers come from one PC (RTX 4080) and one Mac (M4 Max).

## Files

- `model-*.safetensors`: the dense weights (8-bit MLX), the PLE table (4-bit MLX), the vision tower and the MTP head.
- `L{{layer}}_{{gate_proj,up_proj,down_proj}}.blkt`: the trellis-coded routed experts of layers 0 to 47 and of the MTP
  head (layer 48). Read [the format](https://github.com/mbarnson/NanoSeedLM/blob/main/docs/reference/model-folder.md).
- `config.json`, the tokenizer and the chat template of the 4-bit MLX model.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
Dense weights: [mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp)
and [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
Activation statistics: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) imatrix. Trellis code:
QTIP ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). Code: [NanoSeedLM](https://github.com/mbarnson/NanoSeedLM).
"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    kdir, out = sys.argv[1], sys.argv[2]
    ks = {n: kld(os.path.join(kdir, f"_m_rel_{n}.log")) for n in PACKS if os.path.exists(os.path.join(kdir, f"_m_rel_{n}.log"))}
    os.makedirs(out, exist_ok=True)
    for n, k in ks.items():
        open(os.path.join(out, f"{n}.md"), "w").write(card(n, k, ks))
        print(f"{n}: KL {k['mean']:.4f}, top-1 {k['top1']:.2f}% -> {out}/{n}.md")


if __name__ == "__main__":
    main()
