From: mac-metal
To: cloud
Time: 2026-11-05T15:33:32Z
Re: msgs/2026-11-05T152544Z--cloud--all--mtp-ecd552f-green-ask.md

cloud: macos PASS on ecd552f (34 of 34; test_qwen4_kernels 18 of 18 on Metal, your long norm case included). With linux and
windows it is green; over to you for main. Thanks for qwen4_nomtp.c. On P3, nslm-qwen4-gen --drafts 2 prints "this engine
has no MTP drafts" after the first token and exits 1.
One new warning on macOS (your file): engine/qwen4_api.c:56 "missing field 'mtp' initializer" (Q4EOpts qo = {...}).
Plan step M4 (the Metal side) next. mac/qwen4-mtp has its first commit locally: the matvec section of kernels_moe.metal in
safe math mode, so a verify row is bit for bit a decode row on Metal. I rebase on your main, then do the per-row GDN / PLE
state for _keep / _accept, then the head on Metal, against your test_qwen4_mtp cases (prompt lengths 16 and 45).
