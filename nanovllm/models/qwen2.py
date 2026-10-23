"""Qwen2/Qwen2.5 dense decoder using the common Qwen decoder components.

Qwen2 requires Q/K/V bias and has no Q/K normalization. The common decoder
selects these independently by model_type; all checkpoint tensors are required.
"""
from nanovllm.models.qwen3 import Qwen3ForCausalLM


class Qwen2ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        if config.model_type != "qwen2":
            raise ValueError("Qwen2ForCausalLM requires a qwen2 configuration")
        super().__init__(config)
