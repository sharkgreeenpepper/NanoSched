import os
from glob import glob
import json
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    if param.shape != loaded_weight.shape:
        raise ValueError(f"Weight shape mismatch: {param.shape} != {loaded_weight.shape}")
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    parameters = dict(model.named_parameters(remove_duplicate=False))
    expected = set()
    for name in parameters:
        parts = name.split(".")
        module = parts[-2] if len(parts) > 1 else ""
        sources = [source for source, (target, _) in packed_modules_mapping.items() if module == target]
        expected.update(".".join([*parts[:-2], source, parts[-1]]) for source in sources)
        if not sources:
            expected.add(name)
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise ValueError(f"No safetensors weights in {path}")
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as stream:
            weight_map = json.load(stream)["weight_map"]
        if {os.path.basename(f) for f in files} != set(weight_map.values()):
            raise ValueError("Checkpoint shards do not match the safetensors index")
    else:
        weight_map = None
    seen = set()
    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if weight_name in seen or weight_name not in expected:
                    raise ValueError(f"Duplicate or unexpected checkpoint tensor: {weight_name}")
                if weight_map is not None and weight_map.get(weight_name) != os.path.basename(file):
                    raise ValueError(f"Tensor does not match checkpoint index: {weight_name}")
                seen.add(weight_name)
                for k in packed_modules_mapping:
                    if len(weight_name.split(".")) > 1 and k == weight_name.split(".")[-2]:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    missing = expected - seen
    # A checkpoint may store a tied embedding only once. Permit precisely this
    # alias after verifying the actual storage is shared, never arbitrary keys.
    if missing == {"lm_head.weight"} and "model.embed_tokens.weight" in seen:
        if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            missing.clear()
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
    if weight_map is not None and seen != set(weight_map):
        raise ValueError("Checkpoint tensor keys do not match the index")
    return {"loaded_tensors": len(seen), "expected_tensors": len(expected), "missing": sorted(missing)}
