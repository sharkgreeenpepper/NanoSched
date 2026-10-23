import pytest

from nanovllm import SamplingParams
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence


def make_scheduler(config, policy="adaptive"):
    config.scheduler_policy = policy
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


@pytest.mark.parametrize("policy", ["adaptive", "adaptive_v2"])
def test_aged_prefill_gets_service_but_cannot_starve_decode(config, policy):
    scheduler, now = make_scheduler(config, policy)
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


@pytest.mark.parametrize("policy", ["adaptive", "adaptive_v2"])
def test_adaptive_chunks_emit_only_at_boundary_and_release_kv(config, policy):
    scheduler, now = make_scheduler(config, policy)
    seqs = [Sequence(list(range(i, i + 17)), SamplingParams(max_tokens=3)) for i in (10, 30)]
    for seq in seqs:
        scheduler.add(seq)
    for _ in range(100):
        if scheduler.is_finished():
            break
        step(scheduler, now, 0.02)
        # Finished requests have already released their cached-token counter.
        assert all(s.is_finished or s.num_completion_tokens == 0 or s.num_cached_tokens >= s.num_prompt_tokens for s in seqs)
        if policy == "adaptive_v2":
            assert all(s.seq_id not in scheduler.first_token_at for s in seqs if not s.num_completion_tokens)
    assert scheduler.is_finished()
    assert all(s.num_completion_tokens == 3 for s in seqs)
    assert not scheduler.block_manager.used_block_ids
    assert all(b.ref_count == 0 for b in scheduler.block_manager.blocks)
    assert not scheduler.first_token_at


def test_v2_slack_accounts_for_elapsed_time_and_next_token(config):
    scheduler, now = make_scheduler(config, "adaptive_v2")
    seq = Sequence([1, 2], SamplingParams(max_tokens=20))
    scheduler.add(seq)
    step(scheduler, now, 0.02)
    step(scheduler, now, 0.01)
    step(scheduler, now, 0.01)
    scheduler.step_started_at = now[0]
    # Three output tokens allow 300 ms until the next one; 20 ms has
    # elapsed since the first token and the decode itself needs 10 ms.
    assert seq.num_completion_tokens == 3
    assert scheduler._tpot_slack() == pytest.approx(0.27)
    now[0] += 0.05
    scheduler.step_started_at = now[0]
    assert scheduler._tpot_slack() == pytest.approx(0.22)


def test_v2_new_decoder_does_not_inherit_old_slack(config):
    scheduler, now = make_scheduler(config, "adaptive_v2")
    old = Sequence([1, 2], SamplingParams(max_tokens=20))
    scheduler.add(old)
    step(scheduler, now, 0.01)
    for _ in range(5):
        step(scheduler, now, 0.01)
    scheduler.step_started_at = now[0]
    assert scheduler._tpot_slack() == pytest.approx(0.54)
    new = Sequence([10, 11], SamplingParams(max_tokens=20))
    scheduler.add(new)
    assert step(scheduler, now, 0.01)
    scheduler.step_started_at = now[0]
    assert scheduler.first_token_at[old.seq_id] == pytest.approx(0.01)
    assert scheduler.first_token_at[new.seq_id] == pytest.approx(0.07)
    # Reserve a two-request decode at the last measured one-request cost.
    assert scheduler._tpot_slack() == pytest.approx(0.08)


def test_v2_fast_decode_earns_a_larger_prefill_chunk(config):
    scheduler, now = make_scheduler(config, "adaptive_v2")
    scheduler.add(Sequence([1, 2, 3, 4], SamplingParams(max_tokens=20)))
    step(scheduler, now, 0.08)
    scheduler.step_started_at = now[0]
    assert scheduler._slack_budget() == 4
    waiting = Sequence(list(range(20, 40)), SamplingParams(max_tokens=2))
    scheduler.add(waiting)
    assert not step(scheduler, now, 0.01)
    assert step(scheduler, now, 0.16)
    assert scheduler.selected_budget == 8
    assert waiting.num_cached_tokens == 8
    assert waiting.num_completion_tokens == 0


def test_v2_slack_reserves_all_capped_decode_batches(config):
    scheduler, now = make_scheduler(config, "adaptive_v2")
    for i in (1, 5, 9):
        scheduler.add(Sequence([i, i + 1], SamplingParams(max_tokens=20)))
    step(scheduler, now, 0.02)
    scheduler.max_num_seqs = 1
    assert not step(scheduler, now, 0.01)
    scheduler.step_started_at = now[0]
    # The two requests not served yet have 90 ms remaining. A full
    # three-batch decode sweep consumes 30 ms of that allowance.
    assert scheduler._tpot_slack() == pytest.approx(0.06)
