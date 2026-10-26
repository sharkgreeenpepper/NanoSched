"""CPU boundaries for TP startup, rank-local weights and common KV capacity."""
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine import llm_engine
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.embed_head import VocabParallelEmbedding
from nanovllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear


@pytest.fixture
def model_config(monkeypatch):
    hf = SimpleNamespace(model_type="qwen2", architectures=["Qwen2ForCausalLM"],
                         num_attention_heads=28, num_key_value_heads=4,
                         vocab_size=152064, intermediate_size=18944,
                         max_position_embeddings=32768)
    monkeypatch.setattr("nanovllm.config.AutoConfig.from_pretrained", lambda _: hf)
    return hf


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_supported_tp_dimensions(tmp_path, model_config, tp):
    assert Config(str(tmp_path), tensor_parallel_size=tp).tensor_parallel_size == tp


@pytest.mark.parametrize("dimension", ["num_attention_heads", "num_key_value_heads", "vocab_size", "intermediate_size"])
def test_invalid_tp_dimension_rejected_before_rank_start(tmp_path, model_config, dimension):
    setattr(model_config, dimension, getattr(model_config, dimension) + 1)
    with pytest.raises(ValueError, match=dimension):
        Config(str(tmp_path), tensor_parallel_size=2)


def test_instance_rendezvous_and_shared_memory_lifecycle(tmp_path, model_config, monkeypatch):
    class Process:
        def __init__(self, *, target, args):
            self.target, self.args = target, args
            self.joined = False

        def start(self):
            pass

        def join(self):
            self.joined = True

    class Runner:
        def __init__(self, config, rank, events, shared_memory_name):
            self.config, self.rank, self.shared_memory_name = config, rank, shared_memory_name
            self.exited = False

        def call(self, method):
            assert method == "exit"
            self.exited = True

    monkeypatch.setattr(Sequence, "block_size", 256)
    monkeypatch.setattr(llm_engine.mp, "get_context", lambda _: SimpleNamespace(Event=object, Process=Process))
    monkeypatch.setattr(llm_engine, "ModelRunner", Runner)
    monkeypatch.setattr(llm_engine, "Scheduler", lambda _: None)
    monkeypatch.setattr(llm_engine.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(eos_token_id=0))
    engines = []
    try:
        engines.append(llm_engine.LLMEngine(str(tmp_path), tensor_parallel_size=2))
        engines.append(llm_engine.LLMEngine(str(tmp_path), tensor_parallel_size=2))
        engines.append(llm_engine.LLMEngine(str(tmp_path), init_method="tcp://127.0.0.1:34567"))
        first, second, explicit = engines
        assert first.config.init_method != second.config.init_method
        assert first.model_runner.shared_memory_name != second.model_runner.shared_memory_name
        stores = [Path(urlparse(engine.config.init_method).path) for engine in engines[:2]]
        for engine, store in zip(engines[:2], stores):
            assert engine.config.init_method.startswith("file://")
            assert store.parent.is_dir() and not store.exists()
            child_config, rank, _, child_name = engine.ps[0].args
            assert child_config.init_method == engine.config.init_method
            assert rank == 1 and child_name == engine.model_runner.shared_memory_name
        assert explicit.config.init_method == "tcp://127.0.0.1:34567"
        assert explicit._rendezvous_dir is None
        runners = [engine.model_runner for engine in engines]
        processes = [process for engine in engines for process in engine.ps]
    finally:
        for engine in engines:
            engine.exit()
    assert all(runner.exited for runner in runners)
    assert all(process.joined for process in processes)
    assert all(not store.parent.exists() for store in stores)


def test_child_rank_sets_device_and_sequence_block_size_before_group(monkeypatch):
    class GroupReached(Exception):
        pass

    calls = []
    monkeypatch.setattr(Sequence, "block_size", 256)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.append(rank))

    def group(backend, init_method, **kwargs):
        assert calls == [1]
        assert Sequence.block_size == 512
        assert backend == "nccl" and init_method == "file:///tmp/unused-tp-test-store"
        assert kwargs == {"world_size": 2, "rank": 1}
        raise GroupReached

    monkeypatch.setattr(dist, "init_process_group", group)
    config = SimpleNamespace(hf_config=None, kvcache_block_size=512, enforce_eager=True,
                             tensor_parallel_size=2, init_method="file:///tmp/unused-tp-test-store")
    with pytest.raises(GroupReached):
        ModelRunner(config, 1, None, "unused-tp-test-shm")


@pytest.mark.parametrize("local_available", [100, 80])
@pytest.mark.parametrize("cap, minimum, expected", [(-1, 80, 80), (64, 80, 64), (81, 80, None), (-1, 0, None)])
def test_kv_capacity_uses_minimum_rank_before_applying_cap(monkeypatch, local_available, cap, minimum, expected):
    runner = ModelRunner.__new__(ModelRunner)
    runner.rank, runner.world_size, runner.block_size = 1, 2, 4
    hf = SimpleNamespace(num_key_value_heads=4, hidden_size=8, num_attention_heads=4,
                         num_hidden_layers=1, dtype=torch.float32)
    runner.config = SimpleNamespace(hf_config=hf, gpu_memory_utilization=1.0, num_kvcache_blocks=cap)
    runner.model = SimpleNamespace(modules=lambda: [])
    # 2 K/V * 1 layer * 4 tokens * 2 local heads * 2 head dim * FP32 = 128 bytes/block.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (128 * local_available, 20000))
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {"allocated_bytes.all.peak": 0, "allocated_bytes.all.current": 0})
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        assert kwargs.pop("device") == "cuda:1"
        return tensor(*args, **kwargs)

    collective_calls = []

    def minimum_capacity(value, op):
        assert value.item() == local_available and op == dist.ReduceOp.MIN
        collective_calls.append(op)
        value.fill_(minimum)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(dist, "all_reduce", minimum_capacity)
    if expected is None:
        with pytest.raises(ValueError, match="exceeds|No KV blocks"):
            runner.allocate_kv_cache()
        assert not hasattr(runner, "kv_cache")
    else:
        runner.allocate_kv_cache()
        assert runner.config.num_kvcache_blocks == expected
        assert runner.kv_cache.shape == (2, 1, expected, 4, 2, 2)
    assert len(collective_calls) == 1


@pytest.mark.parametrize("tp", [2, 4])
def test_qwen_tp_weight_shards_preserve_packed_order(monkeypatch, tp):
    monkeypatch.setattr(dist, "get_world_size", lambda: tp)
    sources = {name: torch.arange(rows * 56).reshape(rows, 56).float() + offset
               for name, rows, offset in [("q", 56, 0), ("k", 8, 10000), ("v", 8, 20000)]}
    biases = {name: torch.arange(source.shape[0]).float() + offset * 100
              for offset, (name, source) in enumerate(sources.items())}
    dense = torch.arange(56 * 56).reshape(56, 56).float()
    gate = torch.arange(32 * 56).reshape(32, 56).float()
    up = gate + 10000
    for rank in range(tp):
        monkeypatch.setattr(dist, "get_rank", lambda rank=rank: rank)
        qkv = QKVParallelLinear(56, 2, 28, 4, bias=True)
        for name in sources:
            qkv.weight_loader(qkv.weight, sources[name], name)
            qkv.weight_loader(qkv.bias, biases[name], name)
        assert torch.equal(qkv.weight, torch.cat([source.chunk(tp)[rank] for source in sources.values()]))
        assert torch.equal(qkv.bias, torch.cat([bias.chunk(tp)[rank] for bias in biases.values()]))
        mlp = MergedColumnParallelLinear(56, [32, 32])
        mlp.weight_loader(mlp.weight, gate, 0)
        mlp.weight_loader(mlp.weight, up, 1)
        assert torch.equal(mlp.weight, torch.cat([gate.chunk(tp)[rank], up.chunk(tp)[rank]]))
        row = RowParallelLinear(56, 56)
        row.weight_loader(row.weight, dense)
        assert torch.equal(row.weight, dense.chunk(tp, dim=1)[rank])
        embedding = VocabParallelEmbedding(32, 56)
        embedding.weight_loader(embedding.weight, gate)
        assert torch.equal(embedding.weight, gate.chunk(tp)[rank])
