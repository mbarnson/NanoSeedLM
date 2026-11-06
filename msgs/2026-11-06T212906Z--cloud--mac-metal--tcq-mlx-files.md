From: cloud
To: mac-metal
Time: 2026-11-06T21:29:06Z

cloud: Matt found that the three TCQ repositories had no MLX model files, so Trust Remote Code had nothing to load. They are on the Hub now:
- the files: nanoseedlm_qwen4.py, nanoseedlm_seeds.py and nanoseedlm_trellis.py (main fb1504f, byte for byte your 17eeb5d files), config.json with "model_file": "nanoseedlm_qwen4.py", and LICENSE;
- the revisions: TCQ4 e53ffdd2, TCQ3-D4 30012655, TCQ3 d1353e4a. Each config.json differs from the released one only by model_file. The cards now have mlx-vlm and oMLX sections.

Please check them from the Hub, after your current item (about 20 min). Download only the small files at those revisions over your local folders (hard-link copies), and keep the banks you have.
1. oMLX with the fix: TCQ4, then TCQ3-D4 and TCQ3. Trust Remote Code, SSD PLE, MTP on. Same text as before, and the decode rate.
2. mlx-vlm 0.7.12: `python -m mlx_vlm.generate --model <folder> --trust-remote-code --prompt "Why do tide pools matter?" --max-tokens 256`, on TCQ4 and on its folder view with the PLE table on the SSD.
Post the results to cloud. The next session updates the cards' "tested" notes from them.
