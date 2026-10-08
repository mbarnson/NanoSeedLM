"""tools/mova_common.py - shared pieces of the MoVA Python tools: load K2-Horizon-MoVA-36B-A4B (BF16) in MLX through
the MLX reference implementation (k2_horizon_model.py, registered as mlx_lm.models.k2_horizon), the tokenizer, and the
weight overrides that turn the BF16 model into a candidate (expanded NSLM experts, or MLX affine quantize-dequantize).
MLX is needed only by the model functions: the tokenizer and the file exports (mova_export.py windows / prompts /
bench) run on any platform with `tokenizers` and `numpy` (MOVA_MODEL: any folder with the model's tokenizer.json).

Environment:
  OMLX_K2_MODEL  path to k2_horizon_model.py (default: omlx/patches/k2_horizon/k2_horizon_model.py of an importable
                 omlx package)
  MOVA_MODEL     the model snapshot directory (default: the Hugging Face cache, huggingface_hub local_files_only)
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

try:
    import mlx.core as mx   # macOS: the model functions below
except ImportError:         # elsewhere: tokenizer and exports only
    mx = None

ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "IFM/K2-Horizon-MoVA-36B-A4B"


def _omlx_model():
    if os.environ.get("OMLX_K2_MODEL"):
        return Path(os.path.expanduser(os.environ["OMLX_K2_MODEL"]))
    spec = importlib.util.find_spec("omlx")
    for loc in (spec.submodule_search_locations or []) if spec else []:
        f = Path(loc) / "patches" / "k2_horizon" / "k2_horizon_model.py"
        if f.exists():
            return f
    return None


def _mova_dir():
    if os.environ.get("MOVA_MODEL"):
        return Path(os.path.expanduser(os.environ["MOVA_MODEL"]))
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(REPO_ID, local_files_only=True))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"{REPO_ID} not found in the Hugging Face cache; download it or set MOVA_MODEL") from e


OMLX_MODEL = _omlx_model()
MOVA = _mova_dir()
SPARSE = range(3, 48)          # mlp_only_layers 0-2
N_EXPERTS, N_VEXPERTS, D, F = 100, 64, 2560, 768


def register():
    if "mlx_lm.models.k2_horizon" in sys.modules:
        return sys.modules["mlx_lm.models.k2_horizon"]
    if OMLX_MODEL is None:
        raise RuntimeError("set OMLX_K2_MODEL to the MLX reference implementation's k2_horizon_model.py")
    import mlx_lm.models
    spec = importlib.util.spec_from_file_location("mlx_lm.models.k2_horizon", OMLX_MODEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mlx_lm.models.k2_horizon"] = mod
    spec.loader.exec_module(mod)
    mlx_lm.models.k2_horizon = mod
    return mod


def load(lazy=False):
    """The BF16 model and the tokenizer."""
    register()
    from mlx_lm.utils import load_model
    model, _ = load_model(MOVA, lazy=lazy)
    return model, tokenizer()


BOS, EOS, IM_END = 0, 1, 250019


def tokenizer():
    """MoVA's tokenizer (tokenizers library: special tokens in text map to their ids; nothing is added)."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(MOVA / "tokenizer.json"))


def encode(tok, text):
    return tok.encode(text, add_special_tokens=False).ids


# ---- candidates: which tensors, and how they are replaced ----------------------------------------------------------

SCOPES = {
    "gu": ("gate_proj", "up_proj"),
    "gud": ("gate_proj", "up_proj", "down_proj"),
    "d": ("down_proj",),
    "v": ("v_experts",),
    "all": ("gate_proj", "up_proj", "down_proj", "v_experts"),
}


def scoped_linear(model, layer, proj):
    """The SwitchLinear holding a scoped tensor: mlp.experts.<proj>, or self_attn.v_experts."""
    if proj == "v_experts":
        return model.layers[layer].self_attn.v_experts
    return getattr(model.layers[layer].mlp.experts, proj)


def expert_param(model, layer, proj):
    return model.layers[layer].mlp.experts[proj]


def set_expert(model, layer, proj, w):
    getattr(model.layers[layer].mlp.experts, proj).weight = w


def apply_quant(model, scope, bits, group=64):
    """Affine quantize-dequantize (MLX's mx.quantize, group 64) of the scope's tensors, in place, kept as BF16."""
    for l in SPARSE:
        for p in SCOPES[scope]:
            lin = scoped_linear(model, l, p)
            w = lin.weight
            q, s, b = mx.quantize(w, group_size=group, bits=bits)
            lin.weight = mx.dequantize(q, s, b, group_size=group, bits=bits).astype(mx.bfloat16)
            mx.eval(lin.weight)


def apply_overlay(model, overlay_dir, scope):
    """Expanded NSLM experts: overlay_dir/L{l}_{proj}.safetensors holds 'w' [100, out, in] BF16 (nslm-moe --expand)."""
    for l in SPARSE:
        for p in SCOPES[scope]:
            f = Path(overlay_dir) / f"L{l}_{p}.safetensors"
            w = mx.load(str(f))["w"]
            lin = scoped_linear(model, l, p)
            assert w.shape == lin.weight.shape and w.dtype == mx.bfloat16, (f, w.shape, lin.weight.shape)
            lin.weight = w
            mx.eval(lin.weight)


_ROUTED = None


def apply_quant_rest(model, bits, group=64):
    """Affine quantize-dequantize (kept BF16) of every weight with ndim >= 2 OUTSIDE the routed MLP experts:
    attention (q, k, o, softplus gate, value experts, value routers), MLP routers, shared experts, the dense MLPs,
    embeddings, LM head.  Norms and router biases (1-D) stay BF16.  Returns the number of tensors."""
    import re
    from mlx.utils import tree_flatten, tree_unflatten
    routed = re.compile(r"mlp\.experts\.(gate|up|down)_proj\.weight$")
    new, n = [], 0
    for name, w in tree_flatten(model.parameters()):
        if w.ndim >= 2 and name.endswith("weight") and not routed.search(name) and w.shape[-1] % group == 0:
            q, s, b = mx.quantize(w, group_size=group, bits=bits)
            w = mx.dequantize(q, s, b, group_size=group, bits=bits).astype(w.dtype)
            mx.eval(w)
            n += 1
        new.append((name, w))
    model.update(tree_unflatten(new))
    return n


def apply_candidate(model, spec):
    """spec: 'bf16' | 'q8-gu' | 'q4-gu' | 'nslm-gu:DIR' | ... ; '+' joins parts: 'nslm-gud:DIR+q8-rest'."""
    if "+" in spec:
        for part in spec.split("+"):
            apply_candidate(model, part)
        return
    if spec in ("q8-rest", "q4-rest"):
        print(f"{spec}: {apply_quant_rest(model, int(spec[1]))} tensors", flush=True)
        return
    _apply_one(model, spec)


def _apply_one(model, spec):
    if spec == "bf16":
        return
    import re
    kind, _, arg = spec.partition(":")
    name, scope = kind.split("-", 1)
    m = re.fullmatch(r"q(\d)(?:g(\d+))?", name)       # q8, q4, q3, q4g128 ...
    if m:
        apply_quant(model, scope, int(m[1]), int(m[2] or 64))
    elif name == "nslm":
        apply_overlay(model, arg, scope)
    else:
        raise ValueError(spec)


def machine_state():
    import subprocess
    st = {}
    if sys.platform != "darwin":
        if hasattr(os, "getloadavg"):
            la = os.getloadavg()
            st["load1"], st["load5"] = round(la[0], 2), round(la[1], 2)
        return st
    try:
        st["power"] = "AC" if "AC Power" in subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout else "battery"
        la = os.getloadavg()
        st["load1"], st["load5"] = round(la[0], 2), round(la[1], 2)
        th = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
        st["thermal_warning"] = "No thermal warning" not in th
    except Exception as e:  # noqa: BLE001
        st["error"] = str(e)
    return st


def dump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
