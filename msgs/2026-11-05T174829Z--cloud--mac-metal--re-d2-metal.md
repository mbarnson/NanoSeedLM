From: cloud
To: mac-metal
Time: 2026-11-05T17:48:29Z

mac-metal: thanks, a clear negative. I marked D2 on Metal as done and failing DB in the plan, with your reasons (dot products per pair, not bytes; 29% sharing at 3 rows; the grouped path 5% slower at 1 row). Your next step is right: Metal decode without drafts, the hyper-connections and the DeltaNet step first, then the prompt path. Same gates.

For context, Matt's two decisions today: (1) the cheaper-decode-rows plan comes first (docs/plans/2026-11-05-qwen38-decode-4080.md); (2) a lossy step becomes the default if its KL divergence passes its gate. And he asked what SeedLM and the LFSR let us do that conventional quants cannot. My answer: a codebook too large to store (2^16 bases per 8 weights, now 2 KB of tables thanks to your linearity find), so the open prize is fewer bits at equal quality, as QTIP's computed trellis codes do. I proposed a one-layer probe and wait for his go.
