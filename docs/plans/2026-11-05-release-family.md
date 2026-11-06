# Plan: the trellis family of Qwen3.8-Flash-Next on Hugging Face (2026-11-05)

**Goal.** Publish three trellis packs of Qwen3.8-Flash-Next as public Hugging Face repositories. One `hf download`
gives a folder that the NanoSeedLM engines run on a PC (CUDA, Linux or Windows) and on a Mac (Metal). Each card states
the pack's KL against BF16 as shipped, its speed and the machine that it is for.

**Status.** Current: the three repositories are public (2026-11-06). Their names are TCQ4, TCQ3-D4 and TCQ3
(trellis-coded quantization), not TQ: in oMLX, TQ means TurboQuant. Matt approved the release on 2026-11-05. D0 of the
dense plan changed release 1: P3's 4-bit MLX dense weights triple the KL ([packs as
shipped](../results/2026-11-05-dense-as-shipped.md)). So the packs take 8-bit dense weights (step R2b). R1 to R4 are
done, and the folders are in the bucket. Their KL as shipped is 0.0415 (TCQ4), 0.0638 (TCQ3-D4) and 0.0778 (TCQ3) ([the
release packs as shipped](../results/2026-11-05-release-as-shipped.md)). The first R5 job failed: `hf upload` of the
bucket mount gives EIO. The script now copies each folder to the local disk first (`29a76aa`). R5's upload is done (TCQ3
`5f2a8599`, TCQ3-D4 `7e5edb2b`, TCQ4 `25339937`), and Matt made them public on 2026-11-06. RD passes on the RTX 4080
([the release packs from the Hub](../results/2026-11-06-release-downloads-4080.md)). RD and RE pass on the M4 Max ([the
release packs from the Hub on the M4 Max](../results/2026-11-06-release-downloads-m4max.md)). R6 is a draft release of
the engines (v0.1.0) with the start scripts. R7 is on `main`. P3 and P4 now ship `main`'s `nanoseedlm_seeds.py`, so the
risk of a stale loader is closed for them.

## The family

Memory counts the weights that stay resident. The PLE table (32 GB) stays on the disk, and the page cache holds the
rows that a run reads.

| Repository (`txgsync/...`) | Experts | Expert GB | Memory | KL, Q8_0 dense | For |
|---|---|---|---|---|---|
| `Qwen3.8-Flash-Next-NanoSeedLM-TCQ4` | K = 4 | 62.3 | about 66 GB | 0.0398 | A 128 GB Mac. A GPU of 24 GB or more with 96 to 128 GB of RAM |
| `Qwen3.8-Flash-Next-NanoSeedLM-TCQ3-D4` | K = 3, `down_proj` at K = 4 | 52.2 | about 56 GB | 0.0619 | A 16 GB GPU with 96 to 128 GB of RAM. A 96 GB Mac |
| `Qwen3.8-Flash-Next-NanoSeedLM-TCQ3` | K = 3 | 47.2 | about 51 GB | 0.0765 | A 16 GB GPU with 64 GB of RAM. A 64 GB Mac |

- The KL column uses Unsloth's Q8_0 dense weights. Step R3 measures each pack as shipped, with its 4-bit MLX dense
  weights.
- No K = 2 pack: its KL is 0.204. A small pack waits for exclusive tiers (trellis plan step T6).
- P3 and P4 stay published as the first seed packs.

## What a repository holds

- `config.json`, the tokenizer, the chat template and Qwen's `LICENSE`.
- 8-bit MLX dense tensors from `mlx-community/Qwen3.8-Flash-Next-oQ8e-mtp`, written in groups of 64. The engines
  read 8-bit groups of 64 now. The routers, norms and other tensors, the PLE table (4-bit MLX), the vision tower and
  the MTP head come from P3, without P3's seed experts. The engines map the PLE table from the disk.
- The trellis banks `L{l}_{gate,up,down}_proj.blkt` of layers 0 to 47 and of the MTP head (layer 48). `ns_add_banks`
  adds them as the routed experts.
- `README.md`: the card. It has the sections of the P3 card: lineage, the difference, quality, platforms, how to run,
  limits, files and credits. A family table and "which pack for your machine" come first.

The MLX loader (`nanoseedlm_qwen4.py`) has no trellis kernels yet. So `config.json` names no loader, and mlx-vlm and
oMLX cannot load these packs in this release.

## Steps and gates

| Step | Work | Who | Gate |
|---|---|---|---|
| R1 | The MTP head's banks at K = 4 for `gate_proj` and `up_proj`. All head banks into the bucket | pc-cuda | The sizes of layer 47's banks |
| R2 | `nslm-qwen4-pack --tq`: P3's folder without its experts, plus a pack's banks and `LICENSE` (done, `99cf18a`) | cloud | RA |
| R2b | `nslm-qwen4-pack --dense8 DIR`: the dense tensors of an 8-bit MLX folder, its groups of 128 split into 64 | cloud | RA |
| R3 | The KL against BF16 of each pack as shipped (`NAME/mlx` in `tools/hf_llama_kld.sh`) | cloud, HF | RB |
| R4 | The three folders, built in the bucket by an HF job (`tools/hf_release_pack.sh`), and two runs of each on an RTX PRO 6000 | cloud, HF | RC |
| R5 | The upload job to three private repositories, a download and run on the PC and on the Mac, then public | pc-cuda, mac-metal | RD |
| R6 | `start-here.bat` and `start-here.sh`, as Strata's: check the PC, pick a pack, download the engine and the pack, start `nslm-serve` | cloud, pc-cuda | RE |
| R7 | Trellis kernels in the MLX loader, and a test in oMLX | mac-metal | RF |
| R8 | The macOS binaries signed with Matt's Developer ID and notarized in `release.yml` (`cce4968`). After v0.1.0: the five secrets in a GitHub environment `release` that only `main` can deploy to | cloud, mac-metal, Matt | RG |

- **RA.** The engine opens a packed folder of the tiny model, and its logits equal those of the same banks beside the
  full folder.
- **RB.** Each pack's KL as shipped is on its card. The KL of TCQ3-D4 as shipped is at most 5% above its KL with
  Q8_0 dense weights (0.0619).
- **RC.** Each folder gives the same text in two runs.
- **RD.** On the RTX 4080 and the M4 Max, `hf download` and one command give the same text as RC.
- **RE.** On a Windows PC with no NanoSeedLM, the script reaches a chat in one run.
- **RF.** mlx-vlm with `--trust-remote-code` gives the text of the Metal engine within BF16 rounding.
- **RG.** Gatekeeper accepts the `nslm-serve` of a quarantined download, and `start-here.sh` does not clear the
  quarantine. A workflow on a branch other than `main` cannot read the signing secrets.

Release 1 is R1 to R5. R6 and R7 follow as release 1.1 and release 2.

R8 after v0.1.0, in order:

1. Make the GitHub environment `release`, with a deployment policy of `main` only. Matt can be its reviewer.
2. Set the five secrets in it (`gh secret set --env release`). Export the `.p12` again, because GitHub does not give
   back a secret.
3. Add `environment: release` to the `macos` job. Then delete the five secrets of the repository.
4. Check that the App Store Connect key has only the Developer role. Pin the actions of `release.yml` to commit SHAs.
5. The Developer ID certificate expires on 2027-02-01. A new certificate needs new `MACOS_CERT_P12` and
   `MACOS_CERT_PASSWORD` values.

## Costs

- R3: one KL job of TCQ3 and TCQ4 as shipped, with the saved BF16 reference, `rtx-pro-6000x2`, about $2.50. The mix
  and P3 come from the D0 job of [the dense plan](2026-11-05-trellis-dense.md).
- R4: a CPU job to build about 275 GB of folders, and three runs on `rtx-pro-6000`, about $3 in all.
- R5: an upload job on an HF CPU (`cpu-xl`, about 3 hours), as for P3 and P4. The PC starts it with Matt's HF token
  (`--secrets HF_TOKEN`), so the data goes from the bucket to the Hub and not over the PC's link.
- Disk on the Hub: about 275 GB. The PLE table repeats in each repository.

## Risks

- A stale MLX loader can ship. The P3 and P4 repositories ship the `nanoseedlm_seeds.py` from before `cccbd52`, which
  gives wrong experts in verify windows of 4 or more tokens. Each repository's `tools/nanoseedlm_*.py` must equal
  `main`'s at the release commit.
- The card numbers of speed come from one PC and one Mac.
- The 4-bit MLX PLE table is not measured against BF16's (trellis plan step T5). The KL of R3 uses Q8_0's table.

## See also

- [Trellis-coded experts](2026-11-05-trellis-experts.md), [trellis-coded dense weights](2026-11-05-trellis-dense.md).
- [Models](../reference/models.md).
