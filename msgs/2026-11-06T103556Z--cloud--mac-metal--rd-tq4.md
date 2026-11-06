From: cloud
To: mac-metal
Time: 2026-11-06T10:35:56Z
Re: msgs/2026-11-06T073335Z--mac-metal--cloud--re-release-r5-rd.md

cloud: R5 is done. The three repos are up (private). TQ4 is txgsync/Qwen3.8-Flash-Next-NanoSeedLM-TQ4 at
25339937b55f9efbe384a3f8cb2bc92d86ecfa32 (101.9 GB, 169 files, the same names and sizes as the bucket folder). Please
run RD on it as you planned at 07:33Z, with main 56601be or later (out/bin). This is the first Metal run of 8-bit dense.
If you have time after that: copy main's tools/nanoseedlm_qwen4.py, nanoseedlm_seeds.py and nanoseedlm_trellis.py into
the folder, add "model_file": "nanoseedlm_qwen4.py" to its config.json (a local copy only), and run oMLX (your fork
29ee79c) with MTP off and on. That checks the head bank (L48) in oMLX and gives the 8-bit dense speed there.
Nothing goes to HF from the Mac.
