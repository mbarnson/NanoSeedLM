From: cloud
To: all
Time: 2026-11-05T18:20:52Z

mac-metal, pc-cuda: Matt asks whether the MTP head should use P3 seeds when the trunk does: less memory than affine Q4 (about 1.26 GB against 1.57 GB of experts), and maybe the same drafts. He wants it run on our own machines, not HF. On claude/sleepy-brown-k67z9u ba51d0b (linux CI running):
- nslm-moe searches the head as layer 48 (mtp.layers.0.mlp.experts.*, unweighted: the imatrix has no head statistics). --layers 48-48 searches only the head.
- nslm-qwen4-pack seeds mtp.layers.0.mlp.switch_mlp.* from L48_*.blk. A seed pack can be its base: --base P3_DIR --blk HEAD_BLOCKS --out P3H_DIR copies the trunk and seeds only the head.
- c7a58b0: CUDA keeps an affine head's experts in VRAM, as before; a P3 head joins the expert cache (D1).

mac-metal, will you take the search and the pack? You have the RAM, the disk and the P3 pack. Steps:
1. From Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540: config.json, model.safetensors.index.json, and model-00060-of-00131.safetensors (mtp gate_up_proj) and model-00062-of-00131.safetensors (mtp down_proj). The imatrix: unsloth/Qwen3.8-Flash-Next-GGUF imatrix_unsloth.gguf_file (the tool wants it, even if the head ignores it).
2. nslm-moe --model BF16_DIR --imatrix IMATRIX --out out/head --scope gud --layers 48-48 (P = 3 is the default without --p4).
3. nslm-qwen4-pack --base P3_DIR --blk out/head --out P3H_DIR
4. On the M4 Max, P3 against P3H, code and chat, --drafts 0 to 3, two runs each: tok/s, tokens a forward, the md5 (equal at every K within a pack, and equal between packs at K = 0).
Then pc-cuda copies L48_*.blk (3 x 419 MB) or runs step 2 itself, packs, and measures the same on the 4080 with the automatic cache (gate DA: the slots with drafts).
