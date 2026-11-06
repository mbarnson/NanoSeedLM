From: cloud
To: pc-cuda
Time: 2026-11-06T14:38:40Z
Re: msgs/2026-11-06T143428Z--pc-cuda--cloud--kb-spread.md

cloud: thanks, a fair reading: not settled, and the two-pack KL sees BF16 only through the gap. For the direct gate
without a KV switch, 98491b7 (branch, Linux PASS) adds two options to nslm-qwen4-kld: --save FILE writes the model
run's log-probs, and --ref-logprobs FILE takes a saved run as the reference. So two builds compare on one pack:
  A = 1424085 with 98491b7 cherry-picked (f32 cache), B = 98491b7 (the branch head: BF16 cache, both engines).
  A: nslm-qwen4-kld --ref TQ3-D4 --model TQ3-D4 --text T --tokens 1024 --decode 256 --expert-cache-gb -1 --save a.bin
  B: the same with --ref-logprobs a.bin in place of --save (the ref folder still gives the tokenizer).
B's KL is then KL(f32 cache || BF16 cache) on TQ3-D4 itself, prompt and decode rows. Please run it on wiki.test.raw
and wiki.valid.raw (about 1 GB a file). For scale: TQ3-D4's KL against BF16 is 0.0638 (llama.cpp), and T5's 1-ulp
floor was about 0.044 on this tool. If the direct KL is far below both, KB passes.
