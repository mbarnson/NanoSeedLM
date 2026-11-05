# Expert use of Qwen3.8-Flash-Next by text domain (2026-11-04)

**Result.** In each text, the most used third of each layer's experts serves 80% to 93% of the router's choices.
That set changes with the domain. Code shares little with the other texts: its hot third serves 20% to 29% of their
choices, and a random third serves 33%. Chat, wiki and the generated essay share more: 52% to 63%. So a fixed
allocation by use frequency helps only text of its own kind.

**Status.** Current. Four texts of 1,062 to 21,974 tokens, one P3 pack.

## Setup

- Machine: HF Jobs `rtx-pro-6000`, jobs `6aebf9780e0338d1bb805763` (chat, code) and `6aebfba2c78863f9c633207b`
  (wiki). The tide-pools trace is from [the cache policies](2026-11-04-qwen38-cache-policies.md).
- Pack: [Qwen3.8-Flash-Next-NanoSeedLM-P3](https://huggingface.co/txgsync/Qwen3.8-Flash-Next-NanoSeedLM-P3) at
  `5753a67`, through `nslm-qwen4-gen` with `Q4E_ROUTES`. The routing of a P3 pack can differ from BF16.
- Texts (the first 60,000 characters each):
  - Chat: UltraChat test conversations in the chat template, 11,894 tokens.
  - Code: the C sources of `engine/` and `nslm/`, 21,974 tokens.
  - Wiki: wikitext-103 train, 13,848 tokens.
  - Tide pools: a 1,024-token essay that the model wrote, with its prompt (1,062 tokens).
- The chat, code and wiki traces are prompt passes. The tide-pools trace is a decode.

```bash
hf jobs run --flavor rtx-pro-6000 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-95ab3eb -e OUT=/b/traces \
    nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-95ab3eb/tools/hf_qwen4_routes.sh
hf jobs run --flavor rtx-pro-6000 -v hf://buckets/txgsync/nslm-qwen38:/b -e SRC=/b/src-518147a -e OUT=/b/traces \
    -e TEXTS=wiki nvidia/cuda:12.8.1-devel-ubuntu24.04 bash /b/src-518147a/tools/hf_qwen4_routes.sh
```

## Results

The share of each text's router choices that the hot third of another text serves (170 of 512 experts in each
layer, all 48 layers):

| Hot set from | chat | code | wiki | tide pools |
|---|---|---|---|---|
| chat | 79.9% | 20.0% | 55.1% | 62.6% |
| code | 25.8% | 86.0% | 29.1% | 26.1% |
| wiki | 55.2% | 24.3% | 84.0% | 52.4% |
| tide pools | 57.5% | 23.6% | 53.7% | 92.7% |

A random third serves 33%. A text uses all but 2.9% to 5.5% of the experts, except the short essay (35% unused).
The four texts together leave 0.5% of the experts unused.

## Notes

- The diagonal is measured on the trace that chose the set, so it is an upper limit.
- A pack for all uses cannot rely on a static hot set. The expert cache already exploits the locality inside one
  text ([cache policies](2026-11-04-qwen38-cache-policies.md)).
- The RAM-size plan values an expert by its use frequency (step T4). These traces reduce that value for packs that
  serve more than one domain. The KLD of frequency maps measures the effect directly (step T5).
