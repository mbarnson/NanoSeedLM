# NanoSeedLM

NanoSeedLM runs [IFM/K2-Horizon-MoVA-36B-A4B](https://huggingface.co/IFM/K2-Horizon-MoVA-36B-A4B) in less memory.
It uses two methods:

- [SeedLM](https://arxiv.org/abs/2410.10714) replaces stored weights with seeds. The GPU makes the weights from the
  seeds at run time.
- [TransMLA](https://arxiv.org/abs/2502.07864) converts the model's attention to multi-head latent attention (MLA).
  The MLA model keeps less than half the KV cache of the original.

The project has three parts:

- An inference engine in C99, with Metal kernels for Apple Silicon and CUDA kernels for NVIDIA GPUs.
- An MLX loader in each [Hugging Face model repository](docs/reference/models.md), for use in
  [oMLX](https://github.com/jundot/omlx).
- The tools that make the seeds and pack the models.

## Should you use it?

For daily use, probably not. A seed model is not faster than an affine quantization (Q4, Q8) of the same model, and
its quality is not better. A seed model is often a little smaller than an affine model of similar quality. That is
the gain.

If you want the fastest or the most accurate K2-Horizon on your machine, use an affine quantization of the original
model. Use NanoSeedLM for one of these purposes:

- To get the longest context that your Mac can hold.
- To get the smallest model file at a given quality.
- To do research on seeded weights.

## Highlights

- **Seeds instead of stored weights.** The model file keeps each block of 8 weights as a 16-bit seed and a few small
  numbers. The GPU makes the weights again from the seed inside each matrix multiply.
- **A smaller KV cache.** The MLA models keep less than half the KV cache of the original model. With the 8-bit
  cache, the full 512K-token context fits on a 128 GB Mac, and much memory stays free.
- **No patches in oMLX.** Each model repository includes its own loader. In oMLX, turn on **Trust Remote Code** for
  the model. The loader supplies the seed kernels and the MLA attention.
- **The same engine on Metal and CUDA.** The two engines compile the same C sources and run the same tests. Each GPU
  kernel must match a C reference. The two engines give the same quality on the same model.

## The trade-offs

| You get | You give |
|---|---|
| A model file a little smaller than an affine model of similar quality | Slower decode, because the GPU computes the weights |
| A KV cache less than half the size (MLA models) | Some quality, because the MLA conversion moves the model away from the original |
| Long contexts on smaller machines | Slower processing of long prompts, because MLA attention does more arithmetic |

The MLA models also repeat themselves more often than the original in long generations. Set a limit on the output
length. A small number of texts cause most of the MLA quality loss, and we do not know why yet. The measurements are
in [the documentation](docs/index.md).

## Get started

### In oMLX (Mac)

1. Download a model, for example
   [K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4).
2. In oMLX, turn on **Trust Remote Code** for the model.

For long contexts, read [The MLX loader](docs/guides/mlx.md).

### With the engine (Mac)

You need Apple Silicon and the Xcode command-line tools.

```bash
make
out/bin/nslm-chat --model MODEL_DIR "Why do tide pools matter?"
out/bin/nslm-serve --model MODEL_DIR --port 8080
```

`nslm-serve` is an OpenAI-compatible server. It supports streaming, tool calls and continuous batching. Read
[Run and serve](docs/guides/run-and-serve.md).

### With the engine (Windows or Linux)

You need an NVIDIA GPU of the Ampere generation or newer, with 12 GB or more of VRAM. Read
[Build and test](docs/guides/build.md) and [NVIDIA GPUs](docs/guides/cuda.md).

## Models

| Model | Attention | Weights | Size |
|---|---|---|---|
| [K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-NSLM-p4mx-q4v) | GQA (original) | Seeds and affine | 22.8 GB |
| [K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P4) | MLA | Seeds | 22.2 GB |
| [K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P3](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-NanoSeedLM-P3) | MLA | Seeds | 20.1 GB |
| [K2-Horizon-MoVA-36B-A4B-MLA](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA) | MLA | BF16 | 75.6 GB |
| K2-Horizon-MoVA-36B-A4B-MLA-[oQ4e](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-oQ4e), [oQ6e](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-oQ6e), [oQ8e](https://huggingface.co/txgsync/K2-Horizon-MoVA-36B-A4B-MLA-oQ8e) | MLA | Affine, for comparison | 22.9 to 40.6 GB |
| [K2-Horizon-0.9B-NSLM-seeds-4bpw](https://huggingface.co/txgsync/K2-Horizon-0.9B-NSLM-seeds-4bpw) | GQA | Seeds | 0.5 GB |

[The model list](docs/reference/models.md) tells which runtimes load each model.

## How it works

**SeedLM.** A linear feedback shift register (LFSR) makes a sequence of numbers from a 16-bit seed. For each block of
weights, a search finds the seed and the coefficients that come nearest to the original weights. The model file keeps
only the seed and the coefficients. The SeedLM paper made the weights on an FPGA. NanoSeedLM makes them in GPU
kernels. Read [SeedLM](docs/concepts/seedlm.md).

**MLA.** The original model caches a key and a value for each group of attention heads and each token. The MLA model
caches one small latent vector and one position key for each token. Attention reads the keys and values through the
latent. Read [Multi-head latent attention](docs/concepts/mla.md) and [The KV cache](docs/concepts/kv-cache.md).

**Parity.** The Metal and CUDA engines share their C code, their tests and their arithmetic. Read
[Metal and CUDA parity](docs/concepts/parity.md).

## Documentation

The documentation starts at [docs/index.md](docs/index.md). The experiment record is [docs/log.md](docs/log.md), in
date order. AI agents can start at [llms.txt](llms.txt).

## Related work

- [SeedLM](https://arxiv.org/abs/2410.10714), Shafipour et al., ICLR 2025. The method.
- [TransMLA](https://arxiv.org/abs/2502.07864), Meng, Yao and Zhang, 2025. The conversion from GQA to MLA.
- [More related work](docs/reference/related-work.md).

## License

MIT. See `LICENSE`. The model weights have their own licenses.
