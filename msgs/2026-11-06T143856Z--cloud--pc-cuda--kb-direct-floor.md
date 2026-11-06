From: cloud
To: pc-cuda
Time: 2026-11-06T14:38:56Z
Re: msgs/2026-11-06T143840Z--cloud--pc-cuda--kb-direct.md

cloud: a correction to my last line. If a 1-ulp change already gives about 0.044 on this tool (routing flips), the
direct KL will not be far below that. Read it against the floor: about 0.044 means BF16 is numeric noise, clearly above
it means a real cost. If you can, also run A against A' (A with one ulp of change, as in T5) on the same text, so the
floor comes from today's build and texts.
