Status: PASS
Branch: mac/qwen4-mtp
Commit: 79c4b78fb572216c3838eec7545a00074ed700f7
Platform: linux
Agent: mac-metal
Time: 2026-11-05T16:18:14Z

HF job txgsync/6aecabd1c78863f9c6337024 (bin/hf-linux-ci.sh, rtx-pro-6000): 100% tests passed, 0 tests failed out of 39. Skipped as usual: test_affine, test_mova_cfg, test_tokenizer. The branch changes only Metal files, the Makefile and tests/test_qwen4_metal.c, and deletes engine/qwen4_nomtp.c. Nothing in the CUDA build reads them.
