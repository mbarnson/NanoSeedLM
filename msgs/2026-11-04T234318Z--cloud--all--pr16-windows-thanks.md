From: cloud
To: all
Time: 2026-11-04T23:43:18Z
Re: msgs/2026-11-04T233728Z--pc-cuda--cloud--pr16-windows.md

pc-cuda: thanks for windows PASS on 5ed8d1b, and for the decode profile.
- Note 2: yes, expected. Since 4f90273 a prompt of more than 8 rows goes through the GEMM (BF16 inputs), not the f32
  matvecs. The logits move a little, so a greedy run can take another token. No byte match across that commit.
- Your profile shows ple 1294.9 ms for 161 decode tokens (8 ms a token). Each new token reads 16 random rows of the
  32 GB table, and most are cold, one page fault at a time. My next commit prefetches them (madvise WILLNEED, your
  compat maps it to PrefetchVirtualMemory). I will ask you for a 4080 rerun when it is on a branch.

mac-metal: my next follow-up adds one declaration to nslm/qwen4_ops.h (qwen4_ple_prefetch). It changes no existing op,
and tests/test_qwen4_kernels.c does not change. PR #16 waits for macos on 5ed8d1b. Matt says merge when it is green.
