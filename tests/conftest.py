from types import SimpleNamespace
import pytest
from nanovllm.engine.sequence import Sequence


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(Sequence, "block_size", 4)
    return SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=8, eos=-1,
                           kvcache_block_size=4, num_kvcache_blocks=128,
                           scheduler_policy="prefill_first", max_consecutive_decodes=2,
                           decode_prefill_token_budget=4)
