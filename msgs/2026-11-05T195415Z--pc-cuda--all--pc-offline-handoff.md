From: pc-cuda
To: all
Time: 2026-11-05T19:54:15Z

pc-cuda: Matt needs the PC for other work for a few hours (from ~19:55Z). I stopped everything on it: no engine, no queue, no watcher.
I run nothing on the PC until Matt says so. Below: what is left of my queue and where it can go meanwhile.

Correction to my 19:53Z gate DA note: the plan's DA is ">= 3,100 slots with 1 to 3 drafts". P3H keeps 3101 / 3071 / 3008 at K = 1 / 2 / 3,
so DA passes at K = 1 only (each extra draft keeps one more state copy). Same text everywhere. HD1 (+3-5% with drafts) holds on code
(+7.7 to +9.2%) and partly on chat (+0.4 to +4.0%). Drafts still lose to K = 0 on the 4080 even with P3H.

What waits for the PC (Windows only, nothing else can run it):
- windows on mac/qwen4-decode b910034 and on cloud c18e7ac (cl and clang-cl). Main's fast-forwards wait for these unless Matt says
  otherwise. b910034 touches only Metal files and tests/kernel_backend_metal.m, which the Windows build does not compile.

What can move to HF Linux (cloud, HF Jobs; the P3 pack is in txgsync/nslm-qwen38 pack-p3):
- Gate DC's KL half on c18e7ac: nslm-qwen4-kld --ref P3 --model P3 --text wiki.test.raw --tokens 1024 --decode 256 --expert-cache-gb -1,
  --cpu-tail 0 and 7, Clang build, pinned (needs >= ~70 GB RAM: rtx-pro-6000 has 256). The KL does not depend on the machine.
  Reference from my 4080 run (pre-fix 9557466): 0.03508 at 0, 0.03969 at 7.
- Gate DC's speed half, as a relative A/B only: l4x4 (Ada like the 4080, use one L4) with NSLM_VRAM_RESERVE_MB set so 24 GB acts as
  16 GB, --cpu-tail 0 7 0 7 plus one Q4E_PROFILE=1 run for the CPU tail line. The plan says HF speed does not predict the PC (different
  CPU, DRAM and PCIe), so the PC number still has to come later; the HF run tells whether the overlap / 16-row items help at all.
- P3H on HF, if wanted: the head search takes ~5 min on a PRO 6000 (272 s on the 4080) from Qwen/Qwen3.8-Flash-Next@de4b8e4d shards
  00060 and 00062 plus the unsloth imatrix, so no upload from the Mac or the PC is needed. The packed shards are byte-identical either way.

What can move to the Mac: nothing on my list is Metal-side. mac-metal's own decode work is unaffected.

When I am back: windows on whatever is newest on both branches, then gate DC on the PC, then DC's profile line. Raw data on the PC:
G:\devel\qwen4-runs (p3h*.log, dc*.log, d1*.log, strata-probe\).
