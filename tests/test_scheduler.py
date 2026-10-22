from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm import SamplingParams


def finish_step(scheduler, expected_prefill=None):
    seqs, prefill = scheduler.schedule()
    if expected_prefill is not None:
        assert prefill == expected_prefill
    scheduler.postprocess(seqs, [999]*len(seqs), prefill)
    return seqs


def test_original_policy_reproduces_decode_starvation(config):
    scheduler = Scheduler(config)
    incumbent = Sequence([1, 2], SamplingParams(max_tokens=100))
    scheduler.add(incumbent)
    finish_step(scheduler, True)
    for i in range(20):
        scheduler.add(Sequence([10+i]*8))
        finish_step(scheduler, True)
        assert incumbent.num_completion_tokens == 1


def test_decode_bound_and_prefill_starvation_guard(config):
    config.scheduler_policy = "decode_aware"
    now = [0.]
    scheduler = Scheduler(config, clock=lambda: now[0])
    incumbent = Sequence([1, 2], SamplingParams(max_tokens=100))
    scheduler.add(incumbent)
    finish_step(scheduler, True)
    long = Sequence(list(range(20)))
    scheduler.add(long)
    for i in range(3):
        finish_step(scheduler, False)
        finish_step(scheduler, False)
        seqs = finish_step(scheduler, True)
        assert long in seqs
        assert long.num_cached_tokens == 4*(i+1)
    assert incumbent.num_completion_tokens == 7
    assert scheduler.snapshot()["prefill_streak_guards"] == 3


def test_wait_age_overrides_decode_priority(config):
    config.scheduler_policy = "decode_aware"
    now = [0.]
    scheduler = Scheduler(config, clock=lambda: now[0])
    scheduler.add(Sequence([1, 2]))
    finish_step(scheduler, True)
    scheduler.add(Sequence([3]*12))
    now[0] = 0.101
    finish_step(scheduler, True)
    assert scheduler.snapshot()["prefill_age_guards"] == 1


def test_budget_changes_with_decode_backlog(config):
    config.scheduler_policy = "decode_aware"
    scheduler = Scheduler(config)
    first = Sequence(list(range(12)))
    scheduler.add(first)
    finish_step(scheduler, True)
    assert first.num_cached_tokens == 8  # no active decode: full budget
    finish_step(scheduler, True)
    second = Sequence(list(range(20, 32)))
    scheduler.add(second)
    scheduler.decode_streak = config.max_consecutive_decodes
    finish_step(scheduler, True)
    assert second.num_cached_tokens == 4  # active decode: protected budget


def test_chunk_does_not_emit_early_tokens(config):
    scheduler = Scheduler(config)
    seq = Sequence(list(range(12)), SamplingParams(max_tokens=2))
    scheduler.add(seq)
    finish_step(scheduler, True)
    assert seq.num_completion_tokens == 0
    finish_step(scheduler, True)
    assert seq.num_completion_tokens == 1
    finish_step(scheduler, False)
    assert seq.status == SequenceStatus.FINISHED
    assert not scheduler.block_manager.used_block_ids


def test_decode_round_robin_when_batch_is_capped(config):
    config.scheduler_policy = "decode_aware"
    scheduler = Scheduler(config)
    seqs = [Sequence([i, i+1]) for i in (1, 5, 9)]
    for seq in seqs:
        scheduler.add(seq)
    finish_step(scheduler, True)
    scheduler.max_num_seqs = 1
    served = [finish_step(scheduler, False)[0] for _ in range(3)]
    assert served == seqs
