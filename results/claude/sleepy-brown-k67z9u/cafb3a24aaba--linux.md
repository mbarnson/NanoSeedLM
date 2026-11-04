Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: cafb3a24aaba37e103c38050c212ecba87a198b2
Platform: linux
Agent: cloud
Time: 2026-11-04T23:49:01Z

HF Jobs, RTX PRO 6000, CUDA 12.8, Ubuntu 24.04 (bin/hf-linux-ci.sh), job 6aebc4060e0338d1bb8048d3: ctest 31/31 passed
(skipped: test_affine, test_mova_cfg, test_tokenizer), 3 build warnings. test_qwen4_engine includes kc_bucket on 512
and 128 experts and the GEMM row of 48 through the grouped expert GEMMs.
