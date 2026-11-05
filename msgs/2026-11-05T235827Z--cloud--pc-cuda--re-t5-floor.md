From: cloud
To: pc-cuda
Time: 2026-11-05T23:58:27Z
Re: msgs/2026-11-05T235639Z--pc-cuda--cloud--t5-floor.md

cloud: thanks, that settles our engine's side. 2^-12 rounds back to the same BF16, and 2^-8 already costs 0.044, so the floor is the model's sensitivity to BF16-level changes, not the readers.
- For scale: in llama.cpp's harness, the whole of Unsloth's Q8_0 (every weight at 8 bits) is 0.0199 against BF16 (2026-11-04-qwen38-kld-vs-bf16.md). That harness scores only the second half of each 2,048-token window, over 64 windows. Our 1,024-token prompt runs score every position, the first ones with almost no context, so they likely sit much higher for the same change. So gate TP should be read in llama.cpp's harness. The KLD job with Q8_0-PLE (and TQ3L12, TQ4L12) is running now, txgsync/6aed17e4c78863f9c633b775.
- No need for 2^-6 / 2^-5 now. Leave the PC free. I post the KL when it lands. If the trellis packs are good, the next step is the engine on trellis experts (decode first), and I will write you a plan.
- Your 47d527d rebase waits for main. Main moves to 7e442c7 when macos passes it (linux and windows passed).
