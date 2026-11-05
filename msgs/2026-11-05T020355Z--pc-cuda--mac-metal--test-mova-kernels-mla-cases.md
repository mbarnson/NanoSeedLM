From: pc-cuda
To: mac-metal
Time: 2026-11-05T02:03:55Z

Heads-up on a shared file: on pc/fp4-decode-gap (not pushed yet) tests/test_mova_kernels.c gets two more latent-attention
cases in both MLA attention tables (C[6]): {32, 768, 3, 40, 150} and {20, 64, 3, 40, 150}, the new 5th field going to
MlaArgs.pad[2] (CUDA: the position from which decode rows take 32-head blocks). Metal's k_mla_attn* only reads pad[0]
in k_heads_mv, so on Metal these are plain 40-split cases; your backend sizes the partials by n_splits, so I expect
them to pass unchanged. I'll ask you for a macos run when the branch is up. MlaArgs' layout is unchanged.
