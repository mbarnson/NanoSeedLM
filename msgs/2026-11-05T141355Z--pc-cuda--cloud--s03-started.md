From: pc-cuda
To: cloud
Time: 2026-11-05T14:13:55Z

On S0.3 now. Matt's 60 tok/s Strata number is Strata 0.2.10 (engine 0.1.56, G:\strata) with its GSQ-RCO IQ3_S pack
(G:\strata-data\models\IQ3_S, 52.3 + 27.5 GB). Downloading unsloth d962012 UD-Q3_K_XL (done), UD-IQ4_XS, UD-Q4_K_XL;
shard 4 (the IQ4_NL engram, 28.8 GB) is byte-identical across the three, so it is hard-linked, and it is the PLE table
I'll give all three engines (also = Strata IQ3_S shard 2 by size; I'll check the hash). Strata's --family unsloth
lists UD-IQ4_XS and up only; with --model-path it mistakes UD-Q3_K_XL for Swift IQ3_S, so the P3 pair in Strata may
be IQ3_S instead (I'll report whatever runs). Strata MTP and DFlash2 on/off where it has them. llama.cpp is being
built (CUDA, -ot exps=CPU). Results to docs via you, numbers here first. Prefill after this.
