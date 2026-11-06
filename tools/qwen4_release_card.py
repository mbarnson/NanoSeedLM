#!/usr/bin/env python3
"""tools/qwen4_release_card.py - the model cards of the trellis release (docs/plans/2026-11-05-release-family.md), one
README per pack, with the KL as shipped read from tools/hf_release_pack.sh's llama-perplexity logs.

  python3 tools/qwen4_release_card.py KLD_DIR OUT_DIR     (KLD_DIR: the job's OUT/kld, with _m_rel_NAME.log)
"""
import os, re, sys

REPO = "txgsync/Qwen3.8-Flash-Next-NanoSeedLM-"
GITHUB = "https://github.com/mbarnson/NanoSeedLM"
# name: experts, expert bits, expert GB, resident GB (PLE on disk), machines (the lead's title), the K of each projection
PACKS = {
    "TCQ4": ("trellis K = 4", 4.125, 62.3, 70, "a 128 GB Mac, or a 24 GB GPU with 96 to 128 GB of RAM",
             "K = 4 for `gate_proj`, `up_proj` and `down_proj`"),
    "TCQ3-D4": ("trellis K = 3, `down_proj` at K = 4", 3.458, 52.2, 60, "a 16 GB GPU with 96 to 128 GB of RAM, or a 96 GB Mac",
                "K = 3 for `gate_proj` and `up_proj`, K = 4 for `down_proj`"),
    "TCQ3": ("trellis K = 3", 3.125, 47.2, 55, "a 16 GB GPU with 64 GB of RAM",
             "K = 3 for `gate_proj`, `up_proj` and `down_proj`"),
}
# Long runs on the stock path (v0.1.0 nslm-serve, --ctx 32768, Unsloth's Q8_0 PLE table, thinking on): the lead's name,
# machine, decode of 4,096 tokens, decode after 30K tokens, prompt tok/s at 30K. The first row is the lead's
# (docs/results/2026-11-06-long-bench-4080.md)
RTX = "an RTX 4080", "RTX 4080 (16 GB), CUDA"
SPEED = {
    "TCQ4": [(*RTX, "31.4", "30.2", "1,725")],
    "TCQ3-D4": [(*RTX, "38.0", "36.0", "1,824")],
    "TCQ3": [(*RTX, "44.0", "43.1", "2,004")],
}
UNSLOTH = [("UD-Q4_K_XL", 5.101, 0.0304), ("UD-IQ4_XS", 3.999, 0.0509), ("UD-Q3_K_XL", 3.720, 0.0593),
           ("UD-IQ3_XXS", 3.331, 0.0735), ("UD-Q2_K_XL", 3.049, 0.0994)]
PLE_FILE = "engram/Qwen3.8-Flash-Next-engram-Q8_0.gguf_file"


def kld(path):
    """Mean KLD, its error, median, 99th percentile, top-1 agreement and perplexity from a llama-perplexity log."""
    t = open(path).read()
    g = lambda p: re.search(p, t).group(1)
    return dict(mean=float(g(r"Mean    KLD:\s+([\d.]+)")), err=float(g(r"Mean    KLD:\s+[\d.]+ ±\s+([\d.]+)")),
                med=float(g(r"Median  KLD:\s+([\d.]+)")), p99=float(g(r"99.0%   KLD:\s+([\d.]+)")),
                top1=float(g(r"Same top p:\s+([\d.]+)")), ppl=float(g(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")))


def curve(bits):
    """Unsloth's KL at these bits: the straight line between their two nearest packs."""
    pts = sorted((b, v) for _, b, v in UNSLOTH)
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if b0 <= bits <= b1:
            return v0 + (bits - b0) / (b1 - b0) * (v1 - v0)
    return None


def card(name, k, ks):
    experts, bits, egb, mem, who, kdesc = PACKS[name]
    below = 100 * (1 - k["mean"] / curve(bits))
    fam = "\n".join(f"| [{n}](https://huggingface.co/{REPO}{n}) | {PACKS[n][0]} | {PACKS[n][1]:.3f} | {PACKS[n][2]} | "
                    f"about {PACKS[n][3]} GB | {ks[n]['mean']:.4f} | {PACKS[n][4][0].upper()}{PACKS[n][4][1:]} |" for n in PACKS if n in ks)
    uns = "\n".join(f"| Unsloth {n} | {b:.3f} | {v:.4f} |" for n, b, v in UNSLOTH)
    spd = "\n".join(f"| {m} | {d} | {d30} | {p30} |" for _, m, d, d30, p30 in SPEED[name])
    lead = SPEED[name][0]
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
- tcq
- nanoseedlm
- nslm
- cuda
- metal
- research-preview
---

# Qwen3.8-Flash-Next-NanoSeedLM-{name}

**Qwen3.8-Flash-Next on {who}.** Its 120.8 B routed expert weights take {egb} GB, at {bits:.3f} bits each. Its KL
divergence against BF16 is {k['mean']:.4f}. That is {below:.0f}% below Unsloth's packs of the same size. On {lead[0]} it
decodes at {lead[2]} tokens per second, and at {lead[3]} after a prompt of 30,000 tokens.

**How.** The experts are trellis-coded ({kdesc}, 12-bit states, as in
[QTIP](https://arxiv.org/abs/2406.11235)). Neighbouring weights share the bits of one stream. Each weight is a fixed
multiply-add of a 12-bit window of it, so there is no codebook to store or look up. The dense weights are 8-bit MLX.
The 32 GB PLE n-gram table stays on the disk, and the page cache holds the rows that a session reads.

**Start.** One command. Unpack the [engine release]({GITHUB}/releases) for your machine, then run `START-HERE.bat {name}`
(Windows) or `./start-here.sh {name}` (Linux, macOS). The script checks the machine, downloads and checks this pack and
Unsloth's Q8_0 PLE table, and starts an OpenAI-compatible server on port 8080.

> **Research preview.** It runs in the [NanoSeedLM]({GITHUB}) engines: CUDA on Linux and Windows, Metal on macOS. The
> MLX loader for the trellis experts is not in this repository yet.

## The family

Memory counts the weights that stay resident. The PLE table is on the disk.

| Pack | Experts | Bits for each expert weight | Expert GB | Memory | KL against BF16 | For |
|---|---|---|---|---|---|---|
{fam}

## Quality

KL divergence against BF16 in llama.cpp, on 64 windows of 2,048 tokens of wikitext-2. This pack's experts and dense
weights are written into Unsloth's Q8_0 GGUF, so llama.cpp runs this pack's weights. The PLE table is Unsloth's Q8_0.

| | Mean KLD | Median KLD | 99th percentile KLD | Top-1 agreement | Perplexity |
|---|---|---|---|---|---|
| This pack | {k['mean']:.4f} ± {k['err']:.4f} | {k['med']:.4f} | {k['p99']:.3f} | {k['top1']:.2f}% | {k['ppl']:.3f} |
| BF16 | 0 | 0 | 0 | 100% | 3.802 |

Unsloth's packs, measured in the same way (with their own dense weights):

| Pack | Bits for each expert weight | Mean KLD |
|---|---|---|
| This pack | {bits:.3f} | {k['mean']:.4f} |
{uns}

"Below Unsloth's packs of the same size" compares with the straight line between Unsloth's two packs nearest in bits:
{curve(bits):.4f} at {bits:.3f} bits. Read [the measurements]({GITHUB}/blob/main/docs/results/2026-11-05-release-as-shipped.md).

**The PLE table.** Use Unsloth's Q8_0 table (54.4 GB): the KL above is with it. BF16's table gives the same KL within
0.00005. The 4-bit table in this folder (32 GB) adds about 0.006 of KL
([measured]({GITHUB}/blob/main/docs/results/2026-11-06-ple-table-kld.md)). The start scripts download the Q8_0 table.

## Speed

Long runs on the stock path: `nslm-serve --ctx 32768` with the Q8_0 PLE table, thinking on, in tokens per second.
Decode is the mean of two runs, and the prompt is the second run ([the runs]({GITHUB}/blob/main/docs/results/2026-11-06-long-bench-4080.md)).

| Machine, engine | Decode, 4,096 tokens | Decode after 30K tokens | Prompt of 30K tokens |
|---|---|---|---|
{spd}

## How to run, by hand

Download the engine ([releases]({GITHUB}/releases), or [build it]({GITHUB}/blob/main/docs/guides/build.md)), this
folder, and Unsloth's Q8_0 PLE table:

```bash
hf download {REPO}{name} --local-dir Qwen3.8-Flash-Next-NanoSeedLM-{name}
hf download unsloth/Qwen3.8-Flash-Next-GGUF {PLE_FILE} --local-dir unsloth-engram
```

Then chat, or serve an OpenAI-compatible API with the Q8_0 table. On macOS, use `out/bin` in place of `build/bin`. On
Windows, use `set Q4E_PLE_GGUF=...` in place of `export`.

```bash
export Q4E_PLE_GGUF=unsloth-engram/{PLE_FILE}
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-{name} "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-{name} --ctx 32768 --port 8080
```

The engines memory-map both, so the PLE table stays on the disk. On a GPU, the routed experts stay in host memory, and
a cache keeps the used ones in VRAM. Read [Run and serve]({GITHUB}/blob/main/docs/guides/run-and-serve.md).

## Limits

- This is a research preview. Its output is not identical to the output of the original model.
- The MLX loader for the trellis experts is not in this repository yet, so mlx-vlm and oMLX cannot load this pack.
- We did not test image or video input.

## Files

- `model-*.safetensors`: the dense weights (8-bit MLX), the PLE table (4-bit MLX), the vision tower and the MTP head.
- `L{{layer}}_{{gate_proj,up_proj,down_proj}}.blkt`: the trellis-coded routed experts of layers 0 to 47 and of the MTP
  head (layer 48). Read [the format]({GITHUB}/blob/main/docs/reference/model-folder.md).
- `config.json`, the tokenizer and the chat template of the 4-bit MLX model.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
Dense weights: [mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp)
and [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
PLE table: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF). Activation statistics: Unsloth's imatrix.
Trellis code: QTIP ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). Code: [NanoSeedLM]({GITHUB}).
"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    kdir, out = sys.argv[1], sys.argv[2]
    ks = {n: kld(os.path.join(kdir, f"_m_rel_{n}.log")) for n in PACKS if os.path.exists(os.path.join(kdir, f"_m_rel_{n}.log"))}
    os.makedirs(out, exist_ok=True)
    for n, k in ks.items():
        open(os.path.join(out, f"{n}.md"), "w").write(card(n, k, ks))
        print(f"{n}: KL {k['mean']:.4f}, {100 * (1 - k['mean'] / curve(PACKS[n][1])):.0f}% below Unsloth's curve -> {out}/{n}.md")


if __name__ == "__main__":
    main()
