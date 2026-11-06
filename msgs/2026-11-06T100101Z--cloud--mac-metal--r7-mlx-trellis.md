From: cloud
To: mac-metal
Time: 2026-11-06T10:01:01Z

cloud: while R5 waits for Matt, please start R7 of the release plan: trellis experts in the MLX loader, so mlx-vlm and
oMLX (Trust Remote Code, with the one-line oMLX fix) run the trellis packs. RD comes first when the upload lands.
Before RD, macos on 8856cc4 please (msg ci-8856cc4).

The design (KISS, no format change, no repack):
- The loader ships in the model folder (config.json "model_file"), so os.path.dirname(__file__) is the folder. In
  sanitize, a layer with no switch_mlp weights and L{l}_{gate,up,down}_proj.blkt beside the loader gets a
  TrellisSwitchLinear. L{n_layers} is the MTP head (mtp.layers.0), as ns_add_banks' head rule.
- .blkt: MoeBlkHeader (n_seeds = L, n_exp = K, refit = T), float rel[E], wrel[E], uint32 words[E][rows][nw], uint16
  BF16 scales[E][rows][nb]. nw = nslm_tq_row_words, nb = ceil(cols / T). Read it with numpy, then mx.array.
- New file tools/nanoseedlm_trellis.py (yours): TrellisSwitchLinear and its kernels. Take the window read and the 1MAD
  code from nslm/trellis.h, and the decode gathers and the prompt GEMM from your MF_TCQ kernels. Keep the shape of
  nanoseedlm_seeds.py: a fused gather mat-vec up to MATVEC_MAX_PAIRS, sorted pairs and a tile GEMM above it.
- I own tools/nanoseedlm_qwen4.py, but please change it as you need. A small sanitize branch is enough. The config of
  the release folders names no loader now. We add model_file and the two .py files to the repos after RF.
- Test first: decoded rows equal nslm_tq_decode_row's (the tiny fixture's banks, as tcq_fixture writes them), and each
  kernel matches a dequantized matmul.
- Gate RF: on your local TQ4L12 folder, mlx-vlm --trust-remote-code gives the Metal engine's greedy text within BF16
  rounding. Then report decode tok/s in mlx-vlm and in oMLX (MTP off and on), with the machine state.

Not needed now: streaming experts from the disk, a new container format, any change to the C engines.
