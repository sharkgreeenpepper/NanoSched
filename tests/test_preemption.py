import pytest
from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from tests.test_block_manager import assert_consistent
from tests.test_scheduler import finish_step


@pytest.mark.parametrize("policy", ["prefill_first", "decode_aware", "adaptive", "adaptive_v2"])
def test_preemption_resume_preserves_history_and_frees_all_blocks(config, policy):
    config.scheduler_policy = policy
    config.num_kvcache_blocks = 3
    scheduler = Scheduler(config)
    first = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=3))
    second = Sequence([5, 6, 7, 8], SamplingParams(max_tokens=3))
    scheduler.add(first)
    scheduler.add(second)
    finish_step(scheduler, True)
    history = second.token_ids.copy()
    first_token_at = scheduler.first_token_at.get(second.seq_id)
    finish_step(scheduler, False)
    assert scheduler.stats["preemptions"] == 1
    assert second.token_ids == history
    assert not second.block_table
    if policy == "adaptive_v2":
        assert scheduler.first_token_at[second.seq_id] == first_token_at
    assert_consistent(scheduler.block_manager, [first, second])
    for _ in range(20):
        if scheduler.is_finished():
            break
        finish_step(scheduler)
        if policy == "adaptive_v2" and second.num_completion_tokens == 2:
            assert scheduler.first_token_at[second.seq_id] == first_token_at
    assert scheduler.is_finished()
    assert first.num_completion_tokens == second.num_completion_tokens == 3
    assert second.token_ids[:len(history)] == history
    assert scheduler.stats["recomputed_prefill_tokens"] >= 0
    assert_consistent(scheduler.block_manager, [])
    assert not scheduler.first_token_at
