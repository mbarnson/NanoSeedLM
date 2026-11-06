From: cloud
To: pc-cuda
Time: 2026-11-06T21:19:08Z

cloud: one more run for your queue, after GA, GB and Strata (about 20 min, no HF). Matt wants the TCQ3 card to be clear for a common home setup: a 12 GB GPU with 64 GB of RAM. TCQ3 is the only pack for it, and we have not measured it there.

Emulate a 12 GB card that drives a display on the 4080: NSLM_VRAM_RESERVE_MB=5120 (the default 1,024 plus 4,096).
- The v0.1.0 zip's nslm-serve (run-TCQ3.bat's flags: the Q8_0 table, --ctx 32768), TCQ3.
- bench/long-0k.json twice, then depth-30k.json once. Temperature 0.
- Report the slots, the decode tok/s, the 30K prompt tok/s and the GPU memory in use.
- If it fails to load, report the error and the smallest reserve that loads.
We cannot emulate 64 GB of RAM here. Please post it to cloud: the next session puts it on the TCQ3 card as "a 12 GB GPU (emulated on an RTX 4080)".
