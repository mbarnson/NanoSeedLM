# SPDX-License-Identifier: MIT
"""NanoSeedLM for MLX: Qwen3.8-Flash-Next (qwen4_exp, mlx-vlm) whose routed experts are SeedLM P = 3 or 4 weights.

mlx-vlm and oMLX (Trust Remote Code) load this file through config.json "model_file". Each layer's
NAME.seeds / .coefs (or .nibbles) / .codes / .exp_bias tensors replace its switch_mlp projection. Trellis banks
L{l}_{gate,up,down}_proj.blkt beside this file (nanoseedlm_trellis.py, plan R7) replace layer l's projection in turn, and
L{num_hidden_layers} the MTP head's when the model has one. All other tensors are stock.
"""

import glob
import os
import sys

import mlx.core as mx
from mlx.utils import tree_unflatten

_HERE = os.path.dirname(os.path.abspath(__file__))   # nanoseedlm_seeds.py ships beside this loader
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from nanoseedlm_seeds import SeedSwitchGLU, SeedSwitchLinear, _module, seed_p  # noqa: E402
from mlx_vlm.models.qwen4_exp import *  # noqa: E402,F401,F403  (ModelConfig, TextConfig, LanguageModel, ...)
from mlx_vlm.models.qwen4_exp import Model as _Qwen4Model  # noqa: E402


class Model(_Qwen4Model):
    def sanitize(self, weights):
        for k in [k for k in weights if k.endswith(".seeds")]:   # the stock sanitize must not restack these experts
            weights[k[: -len(".seeds")] + ".weight"] = mx.zeros((0,))
        weights = super().sanitize(weights)
        seeded = sorted(k[: -len(".seeds")] for k in weights if k.endswith(".seeds"))
        for p in seeded:
            weights.pop(p + ".weight", None)
            e, n, kb = weights[p + ".seeds"].shape
            self.update_modules(tree_unflatten([(p, SeedSwitchLinear(kb * 8, n, e, seed_p(weights, p)))]))
        for glu in {p.rsplit(".", 1)[0] for p in seeded}:
            _module(self, glu).__class__ = SeedSwitchGLU
        if glob.glob(os.path.join(_HERE, "L*_*_proj.blkt")):   # trellis banks: nanoseedlm_trellis.py ships with them
            from nanoseedlm_trellis import add_banks
            weights = add_banks(self, weights, _HERE, self.config.text_config.num_hidden_layers)
        return weights
