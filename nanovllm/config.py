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
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        architectures = {"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}
        hf = self.hf_config
        if hf.model_type not in architectures or hf.architectures != [architectures[hf.model_type]]:
            raise ValueError(f"Unsupported architecture: {hf.architectures}")
        if getattr(hf, "use_sliding_window", False) or any(t != "full_attention" for t in getattr(hf, "layer_types", []) or []):
            raise ValueError("Sliding/hybrid attention is not implemented")
        rope = getattr(hf, "rope_parameters", None) or getattr(hf, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) != "default" or getattr(hf, "dual_chunk_attention_config", None):
            raise ValueError("Scaled RoPE and dual chunk attention are not implemented")
        if self.max_num_batched_tokens < 1 or self.max_num_seqs < 1 or self.max_model_len < 1:
            raise ValueError("Token, sequence and context limits must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
