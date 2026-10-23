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


@pytest.mark.parametrize("allocation", ["full", "incremental"])
@pytest.mark.parametrize("admission", [False, True])
def test_partial_prefill_cannot_block_only_decoder_from_growing(config, allocation, admission):
    config.kv_allocation = allocation
    config.kv_admission = admission
    config.scheduler_policy = "decode_aware"
    config.num_kvcache_blocks = 3
    config.max_num_batched_tokens = 2
    config.decode_prefill_token_budget = 2
    config.max_consecutive_decodes = 1
    now = [0.0]
    scheduler = Scheduler(config, clock=lambda: now[0])
    first = Sequence([1, 2, 3], SamplingParams(max_tokens=7))
    second = Sequence(list(range(10, 18)), SamplingParams(max_tokens=2))
    scheduler.add(first)
    finish_step(scheduler, True)
    finish_step(scheduler, True)
    scheduler.add(second)

    # Both final histories fit alone. The decoder crosses two block boundaries
    # while the waiting request can hold a partly computed prefill allocation.
    for _ in range(100):
        if scheduler.is_finished():
            break
        seqs, prefill = scheduler.schedule()
        for seq in seqs:
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            assert end <= len(seq.block_table) * config.kvcache_block_size
        now[0] += 0.01
        scheduler.postprocess(seqs, [999] * len(seqs), prefill)
        assert_consistent(scheduler.block_manager, [first, second])

    assert scheduler.is_finished()
    assert first.token_ids == [1, 2, 3] + [999] * 7
    assert second.token_ids == list(range(10, 18)) + [999] * 2
    assert_consistent(scheduler.block_manager, [])
    if not admission:
        assert scheduler.stats["preemptions"] >= 1


@pytest.mark.parametrize("allocation", ["full", "incremental"])
def test_admission_leaves_space_for_new_decoder_first_append(config, allocation):
    config.kv_allocation = allocation
    config.kv_admission = True
    config.scheduler_policy = "decode_aware"
    config.num_kvcache_blocks = 2
    scheduler = Scheduler(config)
    first = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=2))
    second = Sequence([5, 6, 7, 8], SamplingParams(max_tokens=2))
    scheduler.add(first)
    scheduler.add(second)
    # Both prompts fit, but admitting both would leave neither decoder room
    # for its first append. Each request can finish with this pool on its own.
    assert finish_step(scheduler, True) == [first]
    assert not second.block_table
    assert len(scheduler.block_manager.free_block_ids) == 1
    finish_step(scheduler, False)
    assert first.is_finished
    finish_step(scheduler, True)
    finish_step(scheduler, False)
    assert scheduler.is_finished()
    assert scheduler.stats["preemptions"] == 0
    assert_consistent(scheduler.block_manager, [])
