# kld: the llama-perplexity logs of the TCQ packs as shipped (job of 2026-11-05, results/2026-11-05-release-as-shipped.md on main)

`python3 tools/qwen4_release_card.py <this folder> OUT_DIR` on main writes the five cards of the family (TCQ4, TCQ3-D4,
TCQ3, P4, P3). Upload each as README.md of txgsync/Qwen3.8-Flash-Next-NanoSeedLM-<pack>. The P3 and P4 numbers are in
the generator. New speeds go into its SPEED table, then regenerate all five, so the family sections stay the same.
