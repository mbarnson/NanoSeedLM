# agents: how the NanoSeedLM agents coordinate

This branch carries no code. It is where the coding agents working on NanoSeedLM tell each other what they are doing,
claim work, report test results, and notice each other's pushes. Matt reads it too, but he should never have to copy
and paste between agents.

## Who is who

| Agent | Machine | Owns (edits without asking) | Tests it can run |
|---|---|---|---|
| `pc-cuda` | Windows 11, RTX 4080 16 GB, Ryzen 7 5800X3D, 64 GB | `engine/mova_cuda.c`, `engine/kernels_moe.cu`, `engine/kernels_cuda.h`, `nslm/*.cu`, `tests/kernel_backend_cuda.c`, `win/` | `windows` (MSVC + CUDA); `linux` (HF Jobs, either agent) |
| `mac-metal` | M4 Max | `engine/mova_gpu.m`, `engine/kernels_moe.metal`, `nslm/*.metal`, `nslm/*_metal.m`, `tests/kernel_backend_metal.m` | `macos` (Metal) |

**Shared files** are everything else: `engine/engine_api.h`, `engine/mova_ext.h`, `nslm/` (C), `harness/`, `tests/`
(apart from the two backends), `tools/`, `CMakeLists.txt`, `Makefile`, `README.md`. Changing one needs a claim first
(see below), because a change there can break the other platform's build or results.

## Layout

```
README.md               this protocol
claims/<agent>.md       what that agent is working on right now; only its owner edits it
msgs/<UTC>--<from>--<to>--<slug>.md   one message per file; never edited after it is pushed
results/<branch>/<sha12>--<platform>.md   one test run of one commit on one platform
bin/agents-watch.sh     notices pushes and new messages (see "Notifications")
bin/agents-post.sh      writes a message or result, commits it, pushes it
```

Every file has exactly one writer: your own claims file, and new files you create. So `git pull --rebase` on this
branch never conflicts, and a push that loses a race just rebases and pushes again (`agents-post.sh` does that).

## Setup (once per machine)

Keep this branch in its own worktree next to the code checkout, so it never disturbs your working branch:

```sh
git fetch origin agents
git worktree add ../NanoSeedLM-agents agents     # from your NanoSeedLM checkout
cd ../NanoSeedLM-agents
export NSLM_AGENT=mac-metal                      # or pc-cuda; the scripts also take --me
sh bin/agents-watch.sh --once                    # records what is on origin now
```

Then post a message to the other agent saying you are set up (see "Messages").

## The loop

1. **Before starting a task:** `sh bin/agents-watch.sh --once`. Read new messages addressed to you or to `all`,
   and the other agent's claims file.
2. **Claim it:** edit `claims/<you>.md` (task, branch, files you will touch, date) and push. If the task touches a
   shared file the other agent has claimed, message them and agree on who goes first before you edit it.
3. **Work** on a branch of your own (`pc/<topic>`, `mac/<topic>`) or on an integration branch you both agreed on.
4. **Before pushing code:** `git fetch` and rebase onto the branch you share, build, and run your platform's tests (see
   "Green"). Then push the code, post a result for the pushed commit, and post a message if the other agent needs to
   do something (rebase, re-run tests, review an interface).
5. **When you're done:** remove the claim.

Never force-push a shared branch, and never rewrite the other agent's commits. A branch that only you push
(`pc/*`, `mac/*`) may be force-pushed.

## Messages

```sh
sh bin/agents-post.sh msg --to mac-metal --slug kvq-port <<'EOF'
Ported nslm/kvq.h to CUDA on pc/kvq (abc1234). Windows tests pass; please run macos on it.
EOF
```

This writes `msgs/2026-10-08T231500Z--pc-cuda--mac-metal--kvq-port.md` with a header (from, to, time, optional
`--re <earlier message file>`). `--to all` reaches every agent, and Matt. Keep messages short and concrete: branch,
commit, what you need, what you measured. Put long material (profiles, tables) in the commit message or a file on
your branch, and link it from the message.

**Ask Matt** (`--to matt`) before pushing to `main`, before anything outward-facing (Hugging Face uploads, oMLX PRs),
and when you and the other agent disagree.

## Green: every merged branch works on macOS, Windows and Linux

A commit is **green** when `results/<branch>/<sha12>--<platform>.md` exists and says `PASS` for `macos`, `windows` and
`linux`, for that exact commit. Nothing merges to `main` unless its head is green, and Matt approves the merge.

What each platform runs, from a clean build:

| Platform | Who | Command |
|---|---|---|
| `macos` | mac-metal | `make test` (with the model folder set, so model tests do not skip) |
| `windows` | pc-cuda | `win\build.bat test` (or `cmake --build build && ctest --test-dir build`) |
| `linux` | either (HF Jobs) | `sh bin/hf-linux-ci.sh REV` (see "Platforms") |

Report it:

```sh
sh bin/agents-post.sh result --branch pc/kvq --sha abc1234 --platform windows --status PASS <<'EOF'
ctest: 18/18 passed (test_engine, test_mova_kernels on RTX 4080). Skipped: test_affine (no MLX goldens).
nslm-mova-score MLA P=4 healed, held-out KLD 0.1003 (was 0.1001 at batch 6120fa4).
EOF
```

Rules that keep both platforms working together:

- **Tests are shared, backends are not.** A new kernel or format gets a C reference in shared code (`nslm/`,
  `tests/`), and each engine is checked against that same reference through its own `tests/kernel_backend_*`. Never
  compare one engine's output against the other's golden file. Each engine must match the C reference.
- **A shared-file change isn't finished until both platforms pass.** If you change `engine_api.h`, `nslm/`,
  `harness/` or the build files, your push is not green until the other agent posts their result. Ask them for it
  in a message.
- **New engine options default to what happens today**, so the other engine keeps working before it implements them.
  An engine that does not support an option must fail loudly, not ignore it (`engine_api.h`: never ignore a format).
- **GPU tests skip on a machine without that GPU.** They never fail there, so Linux and macOS runs stay meaningful.
- **Quality numbers name their model folder** (healed MLA delta vs the naive J768 conversion, P=3/P=4, KV format), so
  numbers from the two machines are only compared when they come from the same model.

## Notifications

`bin/agents-watch.sh` polls `git ls-remote origin` (cheap: no fetch unless something changed). It reports every
branch that moved (with its new commits and authors), every new message to you or to `all`, every new result, and
every claims change. It skips pushes to `agents` that are only your own posts. It remembers what it has shown in `.watch/` in this worktree (git-ignored).

```sh
sh bin/agents-watch.sh --once            # report changes since the last run, then exit (0)
sh bin/agents-watch.sh --until-change    # poll every 60 s; exit after printing the first change
sh bin/agents-watch.sh --follow          # poll forever, printing each change
```

- **Claude Code:** run `--until-change` as a background command (`run_in_background`). You are woken when it exits,
  then read the report, act, and start it again. Or run `--follow` under the Monitor tool.
- **An agent without background commands:** run `--once` at the start of every task and before every push.
- `--interval N` changes the poll period (default 60 s). `--me NAME` overrides `NSLM_AGENT`.

## Platforms

- **Linux** runs on Hugging Face Jobs: `sh bin/hf-linux-ci.sh REV` builds the commit with CUDA 12.8 on Ubuntu 24.04 and
  runs ctest on an RTX PRO 6000. It takes about 2 minutes and costs about $0.10. Either agent can run it (Matt's `hf` login).
  Ubuntu needs `libicu-dev` for the tokenizer. WSL Ubuntu on the PC has no CUDA toolkit and is not used.
