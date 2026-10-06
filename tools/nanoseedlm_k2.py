# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: K2-Horizon with SeedLM P=4 expert weights, decoded on the GPU.

mlx-lm loads this file through config.json "model_file" (trust_remote_code). It needs the K2-Horizon model
(mlx_lm.models.k2_horizon, provided by oMLX).

Block: 8 weights from a 16-bit LFSR seed (K=16, taps 0,1,3,12), 4 int4 coefficients and a 4-bit exponent code.
With S[c][p] = state(4c+p+1) - 32768: w_c = bf16(R32 * 2^(exp_bias+code) * sum_p S[c][p] * q_p), R32 = fl(1/32767).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort

try:
    from mlx_lm.models import k2_horizon as _k2
except ImportError as error:
    raise ImportError("NanoSeedLM K2-Horizon models need mlx_lm.models.k2_horizon (oMLX provides it)") from error

# (token, expert) pairs at or below this use the fused mat-vec; above it, the layer's experts are decoded to BF16
# for gather_mm.
MATVEC_MAX_PAIRS = 32

_HEADER = """
constant float R32 = as_type<float>(0x38000100u);
inline float block_dot(uint g, const device uint16_t* seeds, const device uint16_t* coefs, const device uint8_t* codes,
                       const device uint32_t* table, thread const float* xv, int bias, thread float* w) {
    uint s = seeds[g];
    uint q = coefs[g];
    int code = (codes[g >> 1] >> ((g & 1) * 4)) & 15;
    float sc = ldexp(R32, bias + code);
    ulong st = ulong(s) | (ulong(table[s]) << 16);
    int q0 = int(q << 28) >> 28, q1 = int(q << 24) >> 28, q2 = int(q << 20) >> 28, q3 = int(q << 16) >> 28;
    float d = 0.0f;
    for (uint c = 0; c < 8; ++c) {
        uint k = 4 * c + 1;
        int isum = (int((st >> k) & 0xFFFF) - 32768) * q0 + (int((st >> (k + 1)) & 0xFFFF) - 32768) * q1
                 + (int((st >> (k + 2)) & 0xFFFF) - 32768) * q2 + (int((st >> (k + 3)) & 0xFFFF) - 32768) * q3;
        if (w) w[c] = float(isum) * sc;
        if (xv) d += float(isum) * xv[c];
    }
    return d * sc;
}
"""

_DECODE = """
uint g = thread_position_in_grid.x;
if (g >= BLOCKS) return;
float w[8];
block_dot(g, seeds, coefs, codes, table, nullptr, exp_bias[g / (N * KB)], w);
for (uint c = 0; c < 8; ++c) out[g * 8 + c] = static_cast<bfloat16_t>(w[c]);
"""

_MATVEC = """
uint lane = thread_position_in_threadgroup.x;
uint r0 = thread_position_in_grid.y * 4;
uint m = thread_position_in_grid.z;
uint e = indices[m];
int bias = exp_bias[e];
const device T* xm = x + m * (KB * 8);
float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (uint b = lane; b < KB; b += 32) {
    float xv[8];
    for (uint c = 0; c < 8; ++c) xv[c] = float(xm[b * 8 + c]);
    for (uint r = 0; r < 4; ++r)
        if (r0 + r < N) acc[r] += block_dot((e * N + r0 + r) * KB + b, seeds, coefs, codes, table, xv, bias, nullptr);
}
for (uint r = 0; r < 4; ++r) {
    float v = simd_sum(acc[r]);
    if (lane == 0 && r0 + r < N) out[m * N + r0 + r] = static_cast<T>(v);
}
"""

_SEED_INPUTS = ["seeds", "coefs", "codes", "exp_bias", "table"]
_kernels: dict = {}
_table = None


def lfsr_table() -> mx.array:
    """The 32 stream bits after each seed: bit k = bit 15 of state k+1."""
    global _table
    if _table is None:
        s = np.arange(65536, dtype=np.uint32)
        g = np.zeros_like(s)
        for k in range(32):
            b = (s ^ (s >> 1) ^ (s >> 3) ^ (s >> 12)) & 1
            s = (s >> 1) | (b << 15)
            g |= b << k
        _table = mx.array(g)
    return _table


def _kernel(name: str, inputs: list[str], source: str):
    if name not in _kernels:
        _kernels[name] = mx.fast.metal_kernel(
            name=f"nanoseedlm_p4_{name}", input_names=inputs, output_names=["out"], source=source, header=_HEADER
        )
    return _kernels[name]


def decode(seeds, coefs, codes, exp_bias, after=None) -> mx.array:
    """All blocks to BF16, shape (experts, rows, cols). The decode waits for `after`, so a lazy graph does not
    decode many layers ahead and hold their BF16 copies at once."""
    e, n, kb = seeds.shape
    kernel = _kernel("decode", _SEED_INPUTS + ["after"], _DECODE)
    after = mx.zeros((1,)) if after is None else after.reshape(-1)[:1]
    return kernel(
        inputs=[seeds, coefs, codes, exp_bias, lfsr_table(), after],
        template=[("BLOCKS", e * n * kb), ("N", n), ("KB", kb)],
        grid=(e * n * kb, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(e, n, kb * 8)],
        output_dtypes=[mx.bfloat16],
    )[0]


def gather_matvec(x, indices, seeds, coefs, codes, exp_bias) -> mx.array:
    """out[m] = W[indices[m]] @ x[m] for x (M, cols), indices (M,)."""
    _, n, kb = seeds.shape
    m = x.shape[0]
    kernel = _kernel("matvec", ["x", "indices"] + _SEED_INPUTS, _MATVEC)
    return kernel(
        inputs=[x, indices, seeds, coefs, codes, exp_bias, lfsr_table()],
        template=[("T", x.dtype), ("N", n), ("KB", kb)],
        grid=(32, (n + 3) // 4, m),
        threadgroup=(32, 8, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


class SeedSwitchLinear(nn.Module):
    """mlx-lm's SwitchLinear with SeedLM P=4 weights."""

    def __init__(self, input_dims: int, output_dims: int, num_experts: int):
        super().__init__()
        if input_dims % 16:
            raise ValueError("SeedLM P=4 needs input_dims divisible by 16")
        self.seeds = mx.zeros((num_experts, output_dims, input_dims // 8), mx.uint16)
        self.coefs = mx.zeros((num_experts, output_dims, input_dims // 8), mx.uint16)
        self.codes = mx.zeros((num_experts, output_dims, input_dims // 16), mx.uint8)
        self.exp_bias = mx.zeros((num_experts,), mx.int32)
        self.input_dims, self.output_dims = input_dims, output_dims

    def _seeds(self):
        return self.seeds, self.coefs, self.codes, self.exp_bias

    def decoded(self, after=None) -> mx.array:
        return decode(*self._seeds(), after)

    def __call__(self, x, indices, sorted_indices=False, after=None):
        rows = x.shape[-2]
        if indices.size * rows <= MATVEC_MAX_PAIRS:
            batch = mx.broadcast_shapes(x.shape[:-2], indices.shape)
            xs = mx.broadcast_to(x, batch + x.shape[-2:]).reshape(-1, self.input_dims)
            idx = mx.broadcast_to(indices, batch).reshape(-1).astype(mx.uint32)
            if rows > 1:
                idx = mx.repeat(idx, rows)
            return gather_matvec(xs, idx, *self._seeds()).reshape(batch + (rows, self.output_dims))
        w = self.decoded(after=x if after is None else after).astype(x.dtype)
        return mx.gather_mm(x, w.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)


class SeedSwitchGLU(SwitchGLU):
    """SwitchGLU that decodes its seed projections one after another."""

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort, after=x_up)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


@dataclass
class ModelArgs(_k2.ModelArgs):
    pass


def _module(root: nn.Module, path: str) -> nn.Module:
    m = root
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


class Model(_k2.Model):
    """K2-Horizon whose expert stacks with NAME.seeds tensors become SeedSwitchLinear."""

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        seeded = sorted({k[: -len(".seeds")] for k in weights if k.endswith(".seeds")})
        for p in seeded:
            if p + ".codes" not in weights:
                raise ValueError(f"{p}: only SeedLM P=4 tensors load in MLX")
            weights[p + ".weight"] = mx.zeros((0,))   # the stack exists: K2's sanitize must not rebuild it
        weights = super().sanitize(weights)
        for p in seeded:
            weights.pop(p + ".weight")
            e, n, kb = weights[p + ".seeds"].shape
            if not hasattr(_module(self, p), "num_experts"):
                raise ValueError(f"{p}: seed weights need a SwitchLinear module")
            self.update_modules(tree_unflatten([(p, SeedSwitchLinear(kb * 8, n, e))]))
        for parent in {p.rsplit(".", 1)[0] for p in seeded}:
            m = _module(self, parent)
            if type(m) is SwitchGLU and all(isinstance(getattr(m, k), SeedSwitchLinear) for k in ("gate_proj", "up_proj", "down_proj")):
                m.__class__ = SeedSwitchGLU
        return weights
