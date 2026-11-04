Status: PASS
Branch: claude/sleepy-brown-k67z9u
Commit: eab620d3766d2039c656dc87febf60c065af5f1f
Platform: linux
Agent: cloud
Time: 2026-11-04T22:31:12Z

HF Jobs, RTX PRO 6000, CUDA 12.8, Ubuntu 24.04 (bin/hf-linux-ci.sh), job 6aebb1b20e0338d1bb804235: ctest 31/31 passed
(skipped: test_affine, test_mova_cfg, test_tokenizer), 3 build warnings. test_qwen4_engine includes the streamed
expert cache (bit for bit against no cache). 79cc53b (the review fixes alone) also passed: job 6aebb15ec78863f9c6330200.
