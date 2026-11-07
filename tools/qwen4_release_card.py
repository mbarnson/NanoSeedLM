#!/usr/bin/env python3
"""tools/qwen4_release_card.py - the model cards of the Qwen3.8-Flash-Next NanoSeedLM family (P3, P4, TCQ4, TCQ3-D4,
TCQ3): one README per pack, with the same family sections in each. The TCQ KL comes from the llama-perplexity logs.

  python3 tools/qwen4_release_card.py KLD_DIR OUT_DIR     (KLD_DIR: _m_rel_TCQ*.log; the agents branch's kld/)
"""
import os, re, sys

HF = "https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-"
GITHUB = "https://github.com/mbarnson/NanoSeedLM"
DOCS = GITHUB + "/blob/main/docs/results/"
OMLX_PR = "https://github.com/jundot/omlx/pull/5112"
OMLX_FORK = "https://github.com/mbarnson/omlx/tree/fix/qwen4-exp-custom-model-file"
BLOG = "https://barnson.org/2026/10/07/nanoseedlm-llm-inference-using-pseudo-random-numbers/"
ORDER = ["TCQ4", "TCQ3-D4", "TCQ3", "P4", "P3"]

# name: expert code, bits for each expert weight, expert GB, dense weights, download GB, resident GB (PLE on disk),
# KL against BF16 as shipped (TCQ: from the logs; None: not measured), the experts' KL with Q8_0 dense, runtimes
PACKS = {
    "TCQ4": ("Trellis K4", 4.125, 62.3, "8-bit", 127.5, 70, None, 0.0398, "Engines, mlx-vlm, oMLX"),
    "TCQ3-D4": ("Trellis K3, down K4", 3.458, 52.2, "8-bit", 117.2, 60, None, None, "Engines, mlx-vlm, oMLX"),
    "TCQ3": ("Trellis K3", 3.125, 47.2, "8-bit", 112.1, 55, None, 0.0765, "Engines, mlx-vlm, oMLX"),
    "P4": ("Seeds P4", 4.5, 68.0, "4-bit", 131.3, 74, 0.169, 0.0408, "Engines, mlx-vlm, oMLX"),
    "P3": ("Seeds P3", 4.0, 60.4, "4-bit", 123.8, 66, 0.182, 0.0600, "Engines, mlx-vlm, oMLX"),
}
# Decode of 4,096 tokens and after 30K, prompt tok/s at 30K, in nslm-serve's long runs (absent: not measured)
M4 = "M4 Max, Metal"
RTX = "RTX 4080, CUDA"
R12 = "RTX 4080 as a 12 GB GPU, CUDA"   # NSLM_VRAM_RESERVE_MB=5120: 4 GB more kept free
SPEED = {
    "TCQ4": {M4: ("39.0", "38.8", "475"), RTX: ("31.4", "30.2", "1,725")},
    "TCQ3-D4": {RTX: ("38.0", "36.0", "1,824")},
    "TCQ3": {RTX: ("44.0", "43.1", "2,004"), R12: ("27.4", "25.9", "1,237")},
    "P4": {},
    "P3": {M4: ("46.9", "45.1", "462")},
}
# oMLX 29ee79c on the M4 Max with the 8-bit PLE table, SSD PLE on, temperature 0 (gate PB of plan ple-8bit)
OMLX_NOTE = {n: f"We tested this pack in oMLX with SSD PLE on an M4 Max: MTP on gives the same text as MTP off, at {on} "
                f"tokens per second against {off}." for n, on, off in (("TCQ4", "about 84", "49"), ("P3", "about 89", "52"))}
OMLX_SAME = "Its PLE table is the same file as TCQ4's and P3's, which we tested in oMLX with SSD PLE."
UNSLOTH = [("UD-Q4_K_XL", 5.101, 0.0304), ("UD-IQ4_XS", 3.999, 0.0509), ("UD-Q3_K_XL", 3.720, 0.0593),
           ("UD-IQ3_XXS", 3.331, 0.0735), ("UD-Q2_K_XL", 3.049, 0.0994)]


def kld(path):
    """Mean KLD, its error, median, 99th percentile, top-1 agreement and perplexity from a llama-perplexity log."""
    t = open(path).read()
    g = lambda p: re.search(p, t).group(1)
    return dict(mean=float(g(r"Mean    KLD:\s+([\d.]+)")), err=float(g(r"Mean    KLD:\s+[\d.]+ ±\s+([\d.]+)")),
                med=float(g(r"Median  KLD:\s+([\d.]+)")), p99=float(g(r"99.0%   KLD:\s+([\d.]+)")),
                top1=float(g(r"Same top p:\s+([\d.]+)")), ppl=float(g(r"Mean PPL\(Q\)\s+:\s+([\d.]+)")))


def curve(bits):
    """Unsloth's KL at these bits: the straight line between their two packs nearest in bits."""
    pts = sorted((b, v) for _, b, v in UNSLOTH)
    for (b0, v0), (b1, v1) in zip(pts, pts[1:]):
        if b0 <= bits <= b1:
            return v0 + (bits - b0) / (b1 - b0) * (v1 - v0)
    return None


def kl(v):
    return "not measured" if v is None else f"{v:.4f}" if v < 0.1 else f"{v:.3f}"


def family(here):
    """The sections that each card shares: which pack, the packs side by side, the trade-offs, decode, the background."""
    L = lambda n: f"**{n}**" if n == here else f"[{n}]({HF}{n})"
    t1 = "\n".join(f"| {L(n)} | {p[0]} | {p[1]:.3f} | {p[5]} | {kl(p[6])} |" for n in ORDER for p in [PACKS[n]])
    t2 = "\n".join(f"| {L(n)} | {p[2]} | {p[3]} | {p[4]} | {p[8]} |" for n in ORDER for p in [PACKS[n]])
    dec = "\n".join(f"| {L(n)} | {SPEED[n].get(M4, ('not measured',))[0]} | {SPEED[n].get(RTX, ('not measured',))[0]} |"
                    for n in ORDER)
    return f"""## Which pack

The five packs are the same model, Qwen3.8-Flash-Next. They differ in three things: how they store the 120.8 B routed
expert weights, their dense weights, and the runtimes that load them.

| Your machine | Pack | Why |
|---|---|---|
| PC: a GPU of 24 GB or more, 96 to 128 GB of RAM | {L("TCQ4")} | The lowest KL |
| PC: a 16 GB GPU, 96 to 128 GB of RAM | {L("TCQ3-D4")} | The middle KL, 10 GB smaller |
| PC: a 12 or 16 GB GPU, 64 GB of RAM | {L("TCQ3")} | The only pack for 64 GB of RAM |
| Mac: 128 GB | {L("TCQ4")} | The lowest KL |
| Mac: 128 GB, for speed | {L("P3")} | About 20% faster, 4 times the KL |
| Mac: 96 GB | {L("TCQ3-D4")} | Room for long contexts |
| Research on seeds | {L("P4")}, {L("P3")} | SeedLM seeds |
| Less than 64 GB of RAM | None | The experts must fit in RAM |

- On a PC, the experts stay in RAM, and the GPU caches the experts in use. A smaller pack lets the GPU cache more of
  its experts.
- A 12 GB GPU caches fewer experts than a 16 GB GPU. On an RTX 4080 limited to 12 GB, TCQ3 decodes at 27 tokens per
  second, against 44 with all 16 GB. We did not test a PC with 64 GB of RAM yet.
- P3 is faster on a Mac because its dense weights are 4-bit. They are also why its KL is 4 times that of TCQ4.
- The start script of the NanoSeedLM [engine release]({GITHUB}/releases) checks the machine and picks a TCQ pack.

## The family

| Pack | Experts | Bits | Memory GB | KL as shipped |
|---|---|---|---|---|
{t1}

| Pack | Expert GB | Dense weights | Download GB | Runtimes |
|---|---|---|---|---|
{t2}

- **Bits** are the bits for each expert weight. **Memory** counts the weights that stay resident, about. The PLE
  n-gram table stays on the disk, and the page cache holds the rows that a session reads.
- **KL as shipped** is against BF16 in llama.cpp, on 64 windows of 2,048 tokens of wikitext-2, with Unsloth's Q8_0 PLE
  table. Lower is closer to the original model.
- **Engines** are the NanoSeedLM engines: CUDA on Linux and Windows, Metal on macOS. oMLX needs the fix in
  [jundot/omlx#5112]({OMLX_PR}). We tested TCQ4 and P3 in oMLX with the 8-bit PLE table, and each pack with the
  earlier 4-bit table.

**The trade-offs.**

- **Trellis or seeds.** Both compute each expert weight from a few stored bits with integer arithmetic, and neither
  stores a codebook. On this model the trellis code (TCQ) is closer to BF16 for its bits. Compare the experts with the
  same 8-bit dense weights: TCQ4 gives 0.0398 at 4.125 bits, P4 0.0408 at 4.5 bits and P3 0.0600 at 4.0 bits.
- **8-bit or 4-bit dense weights.** The dense weights are a small part of the bytes, but in P3 and P4 they cause most
  of the KL. With 8-bit dense weights, P3 gives 0.0600 and P4 0.0408. With the 4-bit ones that they ship, P3 gives 0.182
  and P4 0.169. So the TCQ packs ship 8-bit dense weights. The cost is speed: they read 1.84 GB more for each token. On the M4 Max, TCQ4 decodes about 15% slower than
  P3.
- **Runtimes.** Each pack carries its MLX model files, so mlx-vlm runs it with Trust Remote Code. oMLX runs it with
  the fix in [jundot/omlx#5112]({OMLX_PR}). The NanoSeedLM engines run each pack too. The start scripts of the engine
  release prepare the TCQ packs.
- **The PLE table.** Each pack carries the PLE n-gram table at 8 bits (57.6 GB, MLX's form, group 32), made from Qwen's
  BF16 table. Every runtime reads it. Its row error against BF16 is 0.88 times that of Unsloth's Q8_0 table
  ([the encoding]({DOCS}2026-11-07-ple8-encoding.md)). A 4-bit table would add about 0.006 to the KL.

Decode in tokens per second: `nslm-serve --ctx 32768` with Unsloth's Q8_0 PLE table, 4,096 tokens, thinking on, temperature
0, without MTP drafts, the mean of two runs. The machines are an M4 Max with 128 GB and an RTX 4080 (16 GB) with
128 GB of RAM. Read the runs on [the M4 Max]({DOCS}2026-11-06-long-bench-m4max.md) and
[the RTX 4080]({DOCS}2026-11-06-long-bench-4080.md). The 12 GB GPU is [the RTX 4080 with 4 GB kept free]({DOCS}2026-11-06-tcq3-12gb.md).

| Pack | {M4} | {RTX} |
|---|---|---|
{dec}

**MTP drafts** (engine v0.1.1). Set `Q4E_DRAFTS=2` and `Q4E_DRAFT_MIN_P=0.3`. Then the M4 Max decodes TCQ4 32% faster
at temperature 0 and 16% faster at 1.0. At temperature 0 the text does not change. On the RTX 4080, drafts break even,
so they stay off there. The start script turns them on for a Mac. Read the gates on
[the M4 Max]({DOCS}2026-11-06-draft-gates-m4max.md) and [the RTX 4080]({DOCS}2026-11-06-draft-gates-4080.md).

## Why NanoSeedLM

NanoSeedLM began when Matt Barnson heard about SeedLM ([Shafipour et al., 2024](https://arxiv.org/abs/2410.10714)) at
Apple's machine learning summit in the spring of 2025. SeedLM stores each block of 8 weights as the seed of a linear
feedback shift register (LFSR), plus a few small coefficients. An LFSR is the cheapest pseudo-random generator in
hardware. The paper rebuilt the weights on an FPGA, where an LFSR is a few flip-flops.

The idea: a chip that can calculate a weight from a few bits does not have to store the weight or wait for it.
Decode on home hardware waits mostly for memory. A gaming PC keeps most experts in system RAM, behind PCIe, and a Mac
waits for its memory bandwidth. Cheap arithmetic on the GPU, the CPU or an FPGA can take the place of some memory
reads. Then a model that was state of the art a year ago can run on that hardware at a usable speed.

The first NanoSeedLM model was [K2-Horizon-MoVA-36B-A4B in 22.76 GB](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v),
for a 32 GB Mac. This family takes the idea to Qwen3.8-Flash-Next. On this model, the trellis code of
[QTIP](https://arxiv.org/abs/2406.11235) gives the best quality for its bits. QTIP is the closest working relative of
SeedLM: its weights also come from cheap integer arithmetic, on a bit stream in place of a seed. So the TCQ packs are
the ones to use, and P3 and P4 keep the SeedLM seeds. A planned next step is a CPU kernel for the trellis experts. Then
the CPU of a PC can compute the experts that are not in VRAM. Read the story:
[NanoSeedLM: LLM inference using pseudo-random numbers]({BLOG}).
"""


def yaml(name):
    tcq = name.startswith("TCQ")
    tags = ["qwen4_exp", "qwen3.8-flash-next", "moe", "nanoseedlm", "nslm", "research-preview"] + \
        (["trellis", "tcq"] if tcq else ["seedlm", "lfsr", "seeds"]) + \
        ["mlx", "mlx-vlm", "omlx", "apple-silicon", "custom_code", "cuda", "metal"]
    lib = "library_name: mlx\n"
    return ("---\nbase_model:\n- Qwen/Qwen3.8-Flash-Next\nbase_model_relation: quantized\nlicense: other\n"
            "license_name: qwen-community-1.0\nlicense_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE\n"
            f"pipeline_tag: image-text-to-text\n{lib}tags:\n" + "".join(f"- {t}\n" for t in tags) + "---\n")


def speed(name):
    rows = "\n".join(f"| {m} | {d} | {d30} | {p30} |" for m, (d, d30, p30) in SPEED[name].items())
    if not rows:
        return "We did not measure this pack over long runs yet.\n"
    return f"""This pack's long runs, as in the family table. The prompt column is the second run.

| Machine, engine | Decode, 4,096 tokens | Decode after 30K tokens | Prompt of 30K tokens |
|---|---|---|---|
{rows}
"""


def mlx(name, note):
    """The mlx-vlm and oMLX sections of a card (each pack carries its model files)."""
    dl = PACKS[name][4]
    return f"""### mlx-vlm

```bash
python -m mlx_vlm.generate --model txgsync/Qwen3.8-Flash-Next-NanoSeedLM-{name} --trust-remote-code \\
    --prompt "Why do tide pools matter?" --max-tokens 256
```

mlx-vlm loads the whole pack, {dl} GB, and macOS gives the GPU about three quarters of its memory. So mlx-vlm needs a Mac
with about {dl / 0.75:.0f} GB. On a 128 GB Mac, use oMLX with SSD PLE (below) or the NanoSeedLM engines: both keep the PLE
table on the SSD.

### oMLX

Stock oMLX removes `model_file` from each `qwen4_exp` configuration, so it does not load this pack's model files. The
fix in [jundot/omlx#5112]({OMLX_PR}) keeps `model_file` when it is not the stock `qwen4_exp.py`. Until it merges, use
the branch [fix/qwen4-exp-custom-model-file]({OMLX_FORK}). Then enable Trust Remote Code for this model. {note}
"""


def tcq(name, k):
    code, bits, egb, _, _, mem, _, _, _ = PACKS[name]
    below = 100 * (1 - k["mean"] / curve(bits))
    who = {"TCQ4": "a 128 GB Mac, or a PC with a GPU of 24 GB or more and 96 to 128 GB of RAM",
           "TCQ3-D4": "a PC with a 16 GB GPU and 96 to 128 GB of RAM, or a 96 GB Mac",
           "TCQ3": "a PC with a 12 or 16 GB GPU and 64 GB of RAM"}[name]
    kdesc = {"TCQ4": "K = 4 for `gate_proj`, `up_proj` and `down_proj`",
             "TCQ3-D4": "K = 3 for `gate_proj` and `up_proj`, K = 4 for `down_proj`",
             "TCQ3": "K = 3 for `gate_proj`, `up_proj` and `down_proj`"}[name]
    m, (d, d30, _) = next(iter(SPEED[name].items()))
    uns = "\n".join(f"| Unsloth {n} | {b:.3f} | {v:.4f} |" for n, b, v in UNSLOTH)
    omlx_note = OMLX_NOTE.get(name, OMLX_SAME)
    tested = ("- We emulated a 12 GB GPU on the RTX 4080, with 4 GB more of VRAM kept free. We did not test a PC with 64 GB\n"
              "  of RAM yet. There, close other large programs: the experts and the page cache need most of the RAM.\n") if name == "TCQ3" else ""
    return f"""{yaml(name)}
# Qwen3.8-Flash-Next-NanoSeedLM-{name}

**Qwen3.8-Flash-Next on {who}.** Its 120.8 B routed expert weights take {egb} GB, at {bits:.3f} bits each. Its KL
divergence against BF16 is {k['mean']:.4f}, {below:.0f}% below Unsloth's packs of the same size. On an {m.split(' (')[0]} it
decodes at {d} tokens per second, and at {d30} after a prompt of 30,000 tokens.

> **Research preview.** It runs in the [NanoSeedLM]({GITHUB}) engines: CUDA on Linux and Windows, Metal on macOS. It
> also runs in mlx-vlm with Trust Remote Code, and in oMLX with the fix in [jundot/omlx#5112]({OMLX_PR}).

**Start.** Unpack the [engine release]({GITHUB}/releases) for your machine. Then run `START-HERE.bat {name}` (Windows) or
`./start-here.sh {name}` (Linux, macOS). The script checks the machine, downloads and checks this pack, and starts an
OpenAI-compatible server on port 8080.

{family(name)}
## How it works

The experts are trellis-coded ({kdesc}, 12-bit states, as in [QTIP](https://arxiv.org/abs/2406.11235)). Neighbouring
weights share the bits of one stream. Each weight is a fixed multiply-add of a 12-bit window of the stream, so there is
no codebook to store or look up. The dense weights are 8-bit MLX (oQ8e). On a GPU, the routed experts stay in host
memory, and a cache keeps the experts in use in VRAM.

## Quality

This pack's experts and dense weights are written into Unsloth's Q8_0 GGUF, so llama.cpp runs this pack's weights. The
PLE table in these runs is Unsloth's Q8_0. The pack's own 8-bit table has a lower row error against BF16, so its KL
should be the same within the error. That is not yet measured in llama.cpp.

| | Mean KLD | Median KLD | 99% KLD | Top-1 | Perplexity |
|---|---|---|---|---|---|
| {name} | {k['mean']:.4f} | {k['med']:.4f} | {k['p99']:.3f} | {k['top1']:.2f}% | {k['ppl']:.3f} |
| BF16 | 0 | 0 | 0 | 100% | 3.802 |

The mean KLD's standard error is {k['err']:.4f}. "99% KLD" is the 99th percentile. "Top-1" is the share of positions
where the pack's likeliest token is BF16's.

Unsloth's packs, measured in the same way (with their own dense weights):

| Pack | Bits for each expert weight | Mean KLD |
|---|---|---|
| This pack | {bits:.3f} | {k['mean']:.4f} |
{uns}

"Below Unsloth's packs of the same size" compares with the straight line between Unsloth's two packs nearest in bits:
{curve(bits):.4f} at {bits:.3f} bits. Read [the measurements]({DOCS}2026-11-05-release-as-shipped.md) and
[the KL of the PLE tables]({DOCS}2026-11-06-ple-table-kld.md).

## Speed

{speed(name)}
## How to run, by hand

### NanoSeedLM engines

Download the engine ([releases]({GITHUB}/releases), or [build it]({GITHUB}/blob/main/docs/guides/build.md)) and this
folder. Then chat, or serve an OpenAI-compatible API. The engines read the pack's own 8-bit PLE table. On macOS, use
`out/bin` in place of `build/bin`.

```bash
hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-{name} --local-dir Qwen3.8-Flash-Next-NanoSeedLM-{name}
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-{name} "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-{name} --ctx 32768 --port 8080
```

Read [Run and serve]({GITHUB}/blob/main/docs/guides/run-and-serve.md).

{mlx(name, omlx_note)}
## Limits

- This is a research preview. Its output is not identical to the output of the original model.
- mlx-vlm and oMLX need Trust Remote Code. The NanoSeedLM engines do not.
{tested}- We did not test image or video input.

## Files

- `model-*.safetensors`: the dense weights (8-bit MLX), the vision tower and the MTP head. `model-ple8-*.safetensors`:
  the PLE table (8-bit MLX, group 32, from Qwen's BF16).
- `model-tcq-*.safetensors`: the trellis-coded routed experts of layers 0 to 47 and of the MTP head, one shard
  a layer. Each projection has the entries `NAME.tcq_words`, `NAME.tcq_scales`, `NAME.tcq_params` and
  `NAME.tcq_err`. Read [the format]({GITHUB}/blob/main/docs/reference/model-folder.md).
- `nanoseedlm_qwen4.py`, `nanoseedlm_seeds.py`, `nanoseedlm_trellis.py`: the MLX model files. `nanoseedlm_qwen4.py`
  puts the trellis experts in place of the routed experts.
- `config.json`: the configuration of the 8-bit MLX model, with `model_file`. The tokenizer and the chat template.
- `LICENSE`: the Qwen Community License 1.0.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
Dense weights: [mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp)
and [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
PLE table: Qwen's BF16, in 8 bits by `tools/hf_ple8.py`. Activation statistics: Unsloth's imatrix.
Trellis code: QTIP ([arXiv:2406.11235](https://arxiv.org/abs/2406.11235)). Code: [NanoSeedLM]({GITHUB}).
"""


def seeds(name):
    code, bits, egb, _, dl, mem, shipped, kq8, _ = PACKS[name]
    p = 4 if name == "P4" else 3
    off = 100 * (kq8 / curve(bits) - 1)
    rel = f"{abs(off):.0f}% {'above' if off > 0 else 'below'}"
    shipped_row = kl(shipped) if shipped else "Not measured yet"
    shipped_note = "" if shipped else " In P3, the 4-bit dense weights add 0.12 to the KL."
    lead = {"P4": f"As shipped, its KL against BF16 is {kl(shipped)}, and its 4-bit dense weights cause most of it. Its "
                  f"experts are on Unsloth's quality curve: {kq8:.4f} at 4.5 bits, {rel} it.",
            "P3": f"As shipped, its KL against BF16 is {kl(shipped)}, and its 4-bit dense weights cause most of it."}[name]
    run = ""
    omlx_note = OMLX_NOTE.get(name, OMLX_SAME)
    return f"""{yaml(name)}
# Qwen3.8-Flash-Next-NanoSeedLM-{name}

**Qwen3.8-Flash-Next with SeedLM seed experts, for oMLX on a 128 GB Mac.** Its 120.8 B routed expert weights are seeds of
a linear feedback shift register: {egb} GB, at {bits:.1f} bits each. It runs in mlx-vlm and oMLX as well as in the
NanoSeedLM engines. {lead} For the lowest KL, use [TCQ4]({HF}TCQ4).

> **Research preview.** It runs in the [NanoSeedLM]({GITHUB}) engines, in mlx-vlm with Trust Remote Code, and in oMLX
> with the fix in [jundot/omlx#5112]({OMLX_PR}).

{family(name)}
## How it works

Each block of 8 expert weights is a 16-bit LFSR seed, {p} int4 coefficients and a 4-bit exponent code
([SeedLM](https://arxiv.org/abs/2410.10714)). The GPU runs the register inside the matrix multiply and rebuilds the
weights there, so the pack stores no expert weight. The seed search tried all 65,535 seeds for each block. It weighted
the error of each weight by the mean square of its input activation, from Unsloth's imatrix. The PLE table is 8-bit,
from Qwen's BF16 table. All other weights come unchanged from
[mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp) (4-bit
affine, group size 32), with its MTP head.

| Part | {name} | mlx-community 4-bit |
|---|---|---|
| Routed experts | Seeds P{p}, {bits:.1f} bits | 4-bit affine, 5.0 bits |
| The PLE table | 8-bit affine, from BF16 | 4-bit affine |
| All other weights | The same tensors | 4-bit affine, 8-bit routers |
| Size | {dl} GB | 113.3 GB |

The routed experts are 120.8 B weights (48 layers of 512 experts). The other weights are attention, DeltaNet, the shared
experts and the routers. They also include the embedding, the LM head, the vision tower and the MTP head. The 4-bit
affine weights use groups of 32 and the routers groups of 64.

## Quality

| {name} | Mean KLD against BF16 |
|---|---|
| The experts, with 8-bit dense weights | {kq8:.4f} |
| As shipped, with 4-bit dense weights | {shipped_row} |

The 8-bit dense weights of the first row are Unsloth's Q8_0, in llama.cpp.{shipped_note}

With the same dense weights, Unsloth's packs give {curve(bits):.4f} at {bits:.1f} bits: the straight line between their two
packs nearest in bits. This pack's experts are {rel} that line. Read [KL against BF16]({DOCS}2026-11-04-qwen38-kld-vs-bf16.md)
and [packs as shipped]({DOCS}2026-11-05-dense-as-shipped.md).

## Speed

{speed(name)}
## How to run

### NanoSeedLM engines

Download the engine ([releases]({GITHUB}/releases), or [build it]({GITHUB}/blob/main/docs/guides/build.md)) and this
folder. {run}Then chat, or serve an OpenAI-compatible API. On macOS, use `out/bin` in place of `build/bin`.

```bash
hf download txgsync/Qwen3.8-Flash-Next-NanoSeedLM-{name} --local-dir Qwen3.8-Flash-Next-NanoSeedLM-{name}
build/bin/nslm-chat --model Qwen3.8-Flash-Next-NanoSeedLM-{name} "Why do tide pools matter?"
build/bin/nslm-serve --model Qwen3.8-Flash-Next-NanoSeedLM-{name} --ctx 32768 --port 8080
```

The engines memory-map the pack's 8-bit PLE table, so it stays on the disk. `Q4E_DRAFTS=2` turns on the MTP drafts. Read
[Run and serve]({GITHUB}/blob/main/docs/guides/run-and-serve.md).

{mlx(name, omlx_note)}
## Limits

- This is a research preview. Its output is not identical to the output of the original model.
- mlx-vlm and oMLX need Trust Remote Code. The NanoSeedLM engines do not.
- On Apple Silicon, seed experts decode slower than affine experts of the same size.
- We did not test image or video input.

## Files

- `model-*.safetensors`: the weights. `NAME.seeds`, `NAME.nibbles` and `NAME.exp_bias` hold the routed experts.
  `model-ple8-*.safetensors`: the PLE table (8-bit MLX, group 32, from Qwen's BF16).
- `nanoseedlm_qwen4.py`, `nanoseedlm_seeds.py`: the MLX model files.
- `config.json`: the configuration of the 4-bit MLX model, with `model_file`.
- `LICENSE`: the Qwen Community License 1.0.

## Credits

Base model: Qwen, [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen Community License 1.0).
MLX conversion: [mlx-community/Qwen3.8-Flash-Next-4bit-mtp](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit-mtp).
PLE table: Qwen's BF16, in 8 bits by `tools/hf_ple8.py`.
Activation statistics: [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) imatrix. Method: SeedLM
([arXiv:2410.10714](https://arxiv.org/abs/2410.10714)). Code: [NanoSeedLM]({GITHUB}).
"""


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    kdir, out = sys.argv[1], sys.argv[2]
    ks = {n: kld(os.path.join(kdir, f"_m_rel_{n}.log")) for n in ORDER if n.startswith("TCQ")}
    for n, k in ks.items():   # the as-shipped KL of the family table
        PACKS[n] = PACKS[n][:6] + (k["mean"],) + PACKS[n][7:]
    os.makedirs(out, exist_ok=True)
    for n in ORDER:
        open(os.path.join(out, f"{n}.md"), "w").write(tcq(n, ks[n]) if n in ks else seeds(n))
        print(f"{n} -> {out}/{n}.md")


if __name__ == "__main__":
    main()
