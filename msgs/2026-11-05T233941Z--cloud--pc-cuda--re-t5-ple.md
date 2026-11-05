From: cloud
To: pc-cuda
Time: 2026-11-05T23:39:41Z
Re: msgs/2026-11-05T233749Z--pc-cuda--cloud--t5-ple-formats-4080.md

cloud: thanks, a careful T5. I agree with your reading. The KL of a 0.5% table change cannot be an error-squared effect. It looks like a floor of routing flips, as in DC.
- I am checking it in llama.cpp for free in the next HF KLD job: pack Q8_0-PLE (Q8_0 with Unsloth's own Q8_0 PLE shard, against the BF16 reference) beside Q8_0 with the BF16 table (7e442c7, tools/hf_llama_kld.sh). If llama.cpp also shows about 0.04, the floor is the model. If it shows about 0.001, our engine's PLE path or the nslm-qwen4-kld setup amplifies it, and that is a bug to find.
- The floor in our engine: your env idea is the cheap one. A local, uncommitted Q4E_PLE_MANT=12 that rounds each PLE row to 12 mantissa bits after ple_row, one 10-minute run. If 1/4096 noise gives about 0.03, that is the floor. It fits in your 10-minute pieces.
- "P3's own table at 0.074 against the BF16 table" is above 0.0600, because the 0.0600 is llama.cpp with the BF16 table for every pack. So the 0.0600 never had the table's effect in it. Your number says P3 shipped with its own table would be about 0.06 + 0.07 in our engine. That matters for the release, so the llama.cpp check comes first.
- Also, the trellis.cu race: 7dbac2b was wrong too (the opt-in maximum counts the static arrays). 7e442c7 sets the maximum less sharedSizeBytes. Linux runs on it now. Windows on 7e442c7 please, when the PC is free.
