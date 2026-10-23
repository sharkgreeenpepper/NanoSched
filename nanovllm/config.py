import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    init_method: str | None = None
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    kv_allocation: str = "full"
    kv_admission: bool = False
    scheduler_policy: str = "prefill_first"
    max_consecutive_decodes: int = 1
    decode_prefill_token_budget: int = 1024
    target_tpot_s: float = 0.1
    prefill_max_wait_s: float = 0.5
    adaptive_chunk_sizes: tuple[int, ...] = (256, 512, 1024, 2048)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        architectures = {"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}
        hf = self.hf_config
        if hf.model_type not in architectures or hf.architectures != [architectures[hf.model_type]]:
            raise ValueError(f"Unsupported architecture: {hf.architectures}")
        for name in ("num_attention_heads", "num_key_value_heads", "vocab_size", "intermediate_size"):
            if getattr(hf, name) % self.tensor_parallel_size:
                raise ValueError(f"{name} must be divisible by tensor_parallel_size={self.tensor_parallel_size}")
        if getattr(hf, "use_sliding_window", False) or any(t != "full_attention" for t in getattr(hf, "layer_types", []) or []):
            raise ValueError("Sliding/hybrid attention is not implemented")
        rope = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default" or getattr(hf, "dual_chunk_attention_config", None):
            raise ValueError("Scaled RoPE and dual chunk attention are not implemented")
        if self.max_num_batched_tokens < 1 or self.max_num_seqs < 1 or self.max_model_len < 1:
            raise ValueError("Token, sequence and context limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks < 1:
            raise ValueError("KV block cap must be positive or -1 for automatic capacity")
        if self.kv_allocation not in ("full", "incremental"):
            raise ValueError("kv_allocation must be full or incremental")
        if self.scheduler_policy not in ("prefill_first", "decode_aware", "adaptive", "adaptive_v2"):
            raise ValueError("scheduler_policy must be prefill_first, decode_aware, adaptive or adaptive_v2")
        if self.max_consecutive_decodes < 1 or self.decode_prefill_token_budget < 1:
            raise ValueError("Decode limit and token budget must be positive")
        if self.target_tpot_s <= 0 or self.prefill_max_wait_s <= 0:
            raise ValueError("TPOT target and prefill wait bound must be positive")
        if not self.adaptive_chunk_sizes or any(n <= 0 for n in self.adaptive_chunk_sizes):
            raise ValueError("Adaptive chunk sizes must be positive")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
