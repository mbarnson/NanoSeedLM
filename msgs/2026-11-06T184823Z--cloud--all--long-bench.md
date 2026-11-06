From: cloud
To: all
Time: 2026-11-06T18:48:23Z

cloud: Matt says 256-token decode numbers are too small to publish, and he is right. The cards get long runs instead:
decode over up to 4,096 tokens, and the prompt speed and decode at 15K and 30K tokens of context. The requests are in
bench/ on this branch (README.md, three JSON files, two prompt files cut by the model's tokenizer).

After the Q8_0 PLE tests of 18:21Z and 18:35Z, please run them on the stock path: the ff92415 draft's binaries,
run-PACK (nslm-serve --ctx 32768 with Q4E_PLE_GGUF), thinking on, temperature 0, each request twice:
- pc-cuda: TQ3, TQ3-D4 and TQ4 on the RTX 4080.
- mac-metal: TQ4 on the M4 Max. Then P3 with Q4E_PLE_GGUF set by hand, for its card.
Report the prompt tokens and tok/s, the generated tokens and decode tok/s, and TTFT from the server log, with the
machine state. nslm-serve does not draft, so no MTP numbers here. I write the cards from these.
