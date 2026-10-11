# Source layout

| Folder | Contents |
|---|---|
| `engine/` | The engines: Metal (`mova_gpu.m`, `kernels_moe.metal`) and CUDA (`mova_cuda.c`, `kernels_moe.cu`) |
| `nslm/` | The LFSR, the seed search (C, Metal and CUDA), the model folder reader and writer, the packer, the CPU reference forward |
| `harness/` | The command-line tools, the server, the tokenizer (ICU), the chat template, the tool-call parser, the platform layer |
| `tests/` | One test program for each part. The same tests run on each platform. GPU kernels go through `kernel_backend.h` |
| `win/` | POSIX shims for MSVC, and `build.bat` |
| `tools/` | The MLX loader, and Python tools for calibration and reference data |
| `docs/` | This documentation. Start at [index.md](../index.md) |

## Rules for the code

- The engine, the server and the packer are C99. Objective-C is only thin Metal glue. GPU code is Metal and CUDA.
- Python is only for tools in `tools/`.
- Each GPU kernel has a C reference, and a test compares the two.
