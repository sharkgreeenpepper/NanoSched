from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def make_scheduler(config):
    config.scheduler_policy = "adaptive"
    now = [0.0]
    return Scheduler(config, clock=lambda: now[0]), now


def step(scheduler, now, seconds):
    seqs, prefill = scheduler.schedule()
    now[0] += seconds
    scheduler.postprocess(seqs, [123] * len(seqs), prefill)
    return prefill


def test_feedback_reduces_budget_after_long_token_gap(config):
    scheduler, now = make_scheduler(config)
    scheduler.add(Sequence([1, 2], SamplingParams(max_tokens=20)))
    assert step(scheduler, now, 0.01)
    assert not step(scheduler, now, 0.01)
    assert not step(scheduler, now, 0.01)
    assert scheduler.chunk_index == 2
    now[0] += 0.5
    assert not step(scheduler, now, 0.01)
    assert scheduler.chunk_index == 1
    assert scheduler.decode_gap_s > config.target_tpot_s


def test_active_decode_cost_and_available_budget_cap_chunk(config):
    scheduler, _ = make_scheduler(config)
    assert scheduler._adaptive_budget() == 8
    scheduler.running.append(Sequence([1]))
    scheduler.chunk_index = 2
    scheduler.prefill_token_s = 0.01
    scheduler.decode_step_s = 0.04
    assert scheduler._adaptive_budget() == 4
    scheduler.running.append(Sequence([2]))
    assert scheduler._adaptive_budget() == 2


def test_aged_prefill_gets_service_but_cannot_starve_decode(config):
    scheduler, now = make_scheduler(config)
    scheduler.add(Sequence([1, 2], SamplingParams(max_tokens=100)))
    step(scheduler, now, 0.01)
    waiting = Sequence(list(range(32)), SamplingParams(max_tokens=2))
    scheduler.add(waiting)
    for _ in range(5):
        now[0] += config.prefill_max_wait_s
        scheduler.decode_gap_s = 2 * config.target_tpot_s
        assert not step(scheduler, now, 0.01)
        assert step(scheduler, now, 0.02)
    assert waiting.num_cached_tokens > 0


def test_adaptive_chunks_emit_only_at_boundary_and_release_kv(config):
    scheduler, now = make_scheduler(config)
    seqs = [Sequence(list(range(i, i + 17)), SamplingParams(max_tokens=3)) for i in (10, 30)]
    for seq in seqs:
        scheduler.add(seq)
    for _ in range(100):
        if scheduler.is_finished():
            break
        step(scheduler, now, 0.02)
        # Finished requests have already released their cached-token counter.
        assert all(s.is_finished or s.num_completion_tokens == 0 or s.num_cached_tokens >= s.num_prompt_tokens for s in seqs)
    assert scheduler.is_finished()
    assert all(s.num_completion_tokens == 3 for s in seqs)
    assert not scheduler.block_manager.used_block_ids
    assert all(b.ref_count == 0 for b in scheduler.block_manager.blocks)
