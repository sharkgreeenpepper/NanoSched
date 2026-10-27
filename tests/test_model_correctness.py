"""CPU regression checks. Real model comparisons live in benchmarks.correctness."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from safetensors.torch import save_file
import torch
from torch import nn
from transformers import Qwen2Config, Qwen3Config

from nanovllm.sampling_params import SamplingParams
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


@pytest.mark.parametrize("temperature", [-1, math.inf, math.nan])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        SamplingParams(temperature=temperature)


def test_greedy_and_mixed_sampling():
    logits = torch.tensor([[1., 4., 4.], [100., -100., -100.], [2., 3., 1.]])
    original = logits.clone()
    # Exercise the numerical sampler without paying a CPU Inductor compilation cost.
    sampler = Sampler()
    forward = Sampler.forward.__wrapped__
    for seed in (0, 9, 42):
        torch.manual_seed(seed)
        result = forward(sampler, logits, torch.tensor([0., 1., 0.]))
        assert result.tolist() == [1, 0, 1]
        assert torch.equal(logits, original)
    assert SamplingParams(temperature=0).temperature == 0
    with pytest.raises(ValueError):
        SamplingParams(max_tokens=0)


@pytest.mark.parametrize("rank,temperatures", [(0, [0., 0., 0.]), (0, [0., 1., 0.]), (1, None)])
def test_runner_sampling_path(rank, temperatures):
    from nanovllm.engine.model_runner import ModelRunner
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank = rank
    logits = torch.tensor([[1., 4., 4.], [100., -100., -100.], [2., 3., 1.]])
    original = logits.clone()
    calls = []
    runner.prepare_decode = lambda seqs: (None, None)
    runner.run_model = lambda *args: logits if rank == 0 else None
    def prepare(seqs):
        calls.append("prepare")
        return torch.tensor([seq.temperature for seq in seqs])
    def sample(values, scales):
        calls.append("sample")
        return Sampler.forward.__wrapped__(Sampler(), values, scales)
    runner.prepare_sample = prepare
    runner.sampler = sample
    # TP workers receive serialized Sequence state without a temperature field.
    seqs = [SimpleNamespace() for _ in range(3)] if temperatures is None else [
        SimpleNamespace(temperature=value) for value in temperatures]
    tokens = runner.run(seqs, is_prefill=False)
    assert tokens == ([1, 0, 1] if rank == 0 else None)
    assert calls == (["prepare", "sample"] if temperatures == [0., 1., 0.] else [])
    assert torch.equal(logits, original)


class PackedModel(nn.Module):
    packed_modules_mapping = {"q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), "v_proj": ("qkv_proj", "v")}

    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(2, 6, bias=False)
        def loader(param, weight, shard):
            param.data[{"q": 0, "k": 2, "v": 4}[shard]:{"q": 2, "k": 4, "v": 6}[shard]].copy_(weight)
        self.attn.qkv_proj.weight.weight_loader = loader


def test_loader_requires_every_packed_shard(tmp_path):
    weights = {f"attn.{s}_proj.weight": torch.ones(2, 2)*i for i, s in enumerate("qkv")}
    save_file(weights, tmp_path/"model.safetensors")
    model = PackedModel()
    assert load_model(model, str(tmp_path))["loaded_tensors"] == 3
    assert model.attn.qkv_proj.weight[:, 0].tolist() == [0, 0, 1, 1, 2, 2]
    del weights["attn.v_proj.weight"]
    save_file(weights, tmp_path/"model.safetensors")
    with pytest.raises(ValueError, match="Missing checkpoint tensors"):
        load_model(model, str(tmp_path))


def test_loader_rejects_unexpected_and_absent_weights(tmp_path):
    with pytest.raises(ValueError, match="No safetensors"):
        load_model(nn.Linear(2, 2), str(tmp_path))
    save_file({"wrong.weight": torch.ones(2, 2)}, tmp_path/"model.safetensors")
    with pytest.raises(ValueError, match="unexpected"):
        load_model(nn.Linear(2, 2), str(tmp_path))


@pytest.mark.parametrize("config_cls,bias,norm", [(Qwen2Config, True, False), (Qwen3Config, False, True)])
def test_qwen_architecture_distinctions(config_cls, bias, norm):
    from nanovllm.models.qwen3 import Qwen3ForCausalLM
    config = config_cls(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32)
    with patch("torch.distributed.get_world_size", return_value=1), patch("torch.distributed.get_rank", return_value=0):
        model = Qwen3ForCausalLM(config)
    attn = model.model.layers[0].self_attn
    assert (attn.qkv_proj.bias is not None) == bias
    assert hasattr(attn, "q_norm") == norm
    assert hasattr(attn, "k_norm") == norm


def test_residual_norm_rounds_bf16_sum_before_normalization():
    from nanovllm.layers.layernorm import RMSNorm
    torch.manual_seed(0)
    x = torch.randn(2, 32).bfloat16()
    residual = torch.randn_like(x)
    norm = RMSNorm(32).bfloat16()
    rounded = x+residual
    expected = (rounded.float()*torch.rsqrt(rounded.float().square().mean(-1, keepdim=True)+norm.eps)).bfloat16()*norm.weight
    actual, saved = RMSNorm.add_rms_forward.__wrapped__(norm, x, residual)
    assert torch.equal(saved, rounded)
    assert torch.equal(actual, expected)


def test_qwen2_adapter_matches_transformers_fp32(tmp_path, monkeypatch):
    """Check the complete decoder/loader independently of GPU kernel rounding.

    Only the FlashAttention kernel is replaced by PyTorch dense attention, since
    FlashAttention requires CUDA. This is explicitly a CPU architecture check.
    """
    from transformers import Qwen2ForCausalLM as Reference
    from nanovllm.models.qwen2 import Qwen2ForCausalLM
    from nanovllm.layers.attention import Attention
    from nanovllm.layers.activation import SiluAndMul
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.rotary_embedding import RotaryEmbedding
    from nanovllm.utils.context import set_context, reset_context
    import torch.nn.functional as F
    for cls, name in [(RMSNorm, "rms_forward"), (RMSNorm, "add_rms_forward"),
                      (RotaryEmbedding, "forward"), (SiluAndMul, "forward")]:
        monkeypatch.setattr(cls, name, getattr(cls, name).__wrapped__)
    def dense_attention(self, q, k, v):
        k = k.repeat_interleave(self.num_heads//self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads//self.num_kv_heads, dim=1)
        out = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                            v.transpose(0, 1)[None], is_causal=True, scale=self.scale)
        return out[0].transpose(0, 1)
    monkeypatch.setattr(Attention, "forward", dense_attention)
    torch.manual_seed(3)
    config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)
    reference = Reference(config).eval()
    reference.save_pretrained(tmp_path)
    with patch("torch.distributed.get_world_size", return_value=1), patch("torch.distributed.get_rank", return_value=0):
        model = Qwen2ForCausalLM(config).eval()
    load_model(model, str(tmp_path))
    tokens = torch.randint(0, 32, (1, 17))
    with torch.inference_mode():
        expected = reference(tokens).logits[:, -1]
        cu = torch.tensor([0, tokens.numel()], dtype=torch.int32)
        set_context(True, cu_seqlens_q=cu, cu_seqlens_k=cu)
        try:
            actual = model.compute_logits(model(tokens[0], torch.arange(tokens.numel())))
        finally:
            reset_context()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
