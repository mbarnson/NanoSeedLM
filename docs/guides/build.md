# Build and test

The engine builds on macOS (Metal), Windows (CUDA) and Linux (CUDA). The three builds compile the same C sources and
run the same tests.

## Requirements

**macOS:** Apple Silicon, and the Xcode command-line tools with the Metal compiler.

**Windows and Linux:**

- An NVIDIA GPU of the Ampere generation or newer (RTX 30, 40 or 50 series, sm_80 or newer), with 12 GB or more of
  VRAM.
- 48 GB or more of RAM. Long contexts need 64 GB.
- CUDA Toolkit 12.4 or newer. The RTX 50 series needs 12.8 or newer.
- CMake 3.24 or newer with Ninja (`pip install cmake ninja`).
- Windows: Visual Studio 2022 Build Tools (C++). Windows 10 1903 and later include ICU.
- Linux: GCC or Clang, and ICU (`libicu-dev`).

The build compiles for sm_80, sm_86 and sm_89, and for sm_120 with CUDA 12.8 or newer. Other GPUs (Hopper,
datacenter Blackwell) run the compute_89 PTX. The driver compiles it on first use. To build for the installed GPU
only, add `-DCMAKE_CUDA_ARCHITECTURES=native`.

**Python tools (optional, Python 3):**

- On any platform, with `tokenizers` and `numpy`:
  - `tools/tokenizer_golden.py` makes `tests/data/tokenizer_golden.json` again from the `tokenizer.json` of a model.
  - `tools/mova_export.py windows | prompts | bench` makes the inputs for `nslm-mova-score`, `nslm-mova-gen` and
    `nslm-mova-bench`.
  - `tools/template_golden.py` also needs `transformers`.
- On macOS only, with `mlx`, `mlx-lm` and `huggingface_hub`: the MLX loader, and the tools that run the model in MLX
  (`mova_capture.py`, `mova_score.py`, `mova_affine_golden.py`, `mova_export.py routes`).

## Build

macOS:

```bash
make
make test
```

Windows, from any terminal (the script finds Visual Studio):

```bat
win\build.bat
win\build.bat test
```

Linux:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build && ctest --test-dir build
```

The binaries go to `out/bin` (macOS) or `build/bin` (Windows, Linux).

## Tests

- `test_mova_kernels` checks each GPU kernel against a C reference. It uses a small backend for each GPU API
  (`tests/kernel_backend_cuda.c`, `tests/kernel_backend_metal.m`).
- `test_mova_cfg` and `test_tokenizer` read a model folder. Without one, they skip. To give a folder, use one of these:
  - `MOVA_DIR=...` with `make test`.
  - `-DNSLM_MODEL_DIR=...` with CMake.
  - `NSLM_MODEL_DIR` in the environment before the first `win\build.bat`.
- The GPU tests skip on a machine without a GPU.
- `make test` builds the MLX goldens for `test_affine`. The CMake builds have no MLX. They skip `test_affine` unless the
  goldens exist.

To test the MLX loader, use the K2-Horizon model file of oMLX (`omlx/patches/k2_horizon/k2_horizon_model.py` in
[oMLX](https://github.com/jundot/omlx)):

```bash
OMLX_K2_MODEL=/path/to/k2_horizon_model.py make test-mlx
```

## See also

- [Run and serve](run-and-serve.md)
- [NVIDIA GPUs](cuda.md)
- [Source layout](../reference/source-layout.md)
