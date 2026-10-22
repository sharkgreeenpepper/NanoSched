from collections import deque
from collections import Counter
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, clock=perf_counter):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.policy = config.scheduler_policy
        self.max_consecutive_decodes = config.max_consecutive_decodes
        self.decode_prefill_token_budget = config.decode_prefill_token_budget
        self.clock = clock
        self.reset_stats()

    def reset_stats(self):
        self.decode_streak = 0
        self.stats = Counter()
        self.phase = "idle"

    def snapshot(self):
        bm = self.block_manager
        return {**self.stats, "policy": self.policy, "phase": self.phase,
                "decode_streak": self.decode_streak, "waiting": len(self.waiting), "running": len(self.running),
                "oldest_prefill_wait_s": max((self.clock()-s.prefill_queued_at for s in self.waiting), default=0),
                "kv_used_blocks": len(bm.used_block_ids), "kv_total_blocks": len(bm.blocks)}

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        if seq.num_blocks > len(self.block_manager.blocks):
            raise ValueError("Prompt cannot fit in the KV pool")
        seq.prefill_queued_at = self.clock()
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        if self.is_finished():
            raise RuntimeError("Cannot schedule an empty engine")
        prefer_decode = False
        if self.policy == "decode_aware" and self.running and self.waiting:
            # Bound decode service by iterations. A waiting prefill gets a turn
            # after at most N decode batches, independent of wall-clock timing.
            prefer_decode = self.decode_streak < self.max_consecutive_decodes
            if not prefer_decode:
                self.stats["prefill_streak_guards"] += 1
        if prefer_decode:
            seqs = self._schedule_decode()
            if seqs:
                return self._record_schedule(seqs, False)
        budget = self.max_num_batched_tokens
        if self.policy == "decode_aware" and self.running:
            budget = min(budget, self.decode_prefill_token_budget)
        seqs = self._schedule_prefill(budget)
        if seqs:
            return self._record_schedule(seqs, True)
        seqs = self._schedule_decode()
        if seqs:
            return self._record_schedule(seqs, False)
        # Decode may have evicted its last sequence. Re-enter prefill instead of
        # asserting on an empty batch as upstream does under tight KV capacity.
        seqs = self._schedule_prefill(self.max_num_batched_tokens)
        if seqs:
            return self._record_schedule(seqs, True)
        raise RuntimeError("No request can make progress with the current KV capacity")

    def _record_schedule(self, seqs, is_prefill):
        phase = "prefill" if is_prefill else "decode"
        self.stats["phase_switches"] += self.phase not in (phase, "idle")
        self.phase = phase
        self.stats[f"{phase}_steps"] += 1
        self.stats[f"{phase}_tokens"] += sum(s.num_scheduled_tokens for s in seqs)
        self.decode_streak = 0 if is_prefill else self.decode_streak + 1
        return seqs, is_prefill

    def _schedule_prefill(self, budget):
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = budget - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                self.stats["prefix_cached_tokens"] += seq.num_cached_tokens
                self.stats["prefix_lookup_tokens"] += max(0, seq.num_blocks-1) * self.block_size
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _schedule_decode(self):
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < min(self.max_num_seqs, self.max_num_batched_tokens):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if self.policy == "prefill_first":
            self.running.extendleft(reversed(scheduled_seqs))
        else:
            self.running.extend(scheduled_seqs)
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        self.stats["preemptions"] += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        seq.prefill_queued_at = self.clock()
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        if len(seqs) != len(token_ids):
            raise ValueError("Model returned a different number of samples than scheduled requests")
        for seq, token_id in zip(seqs, token_ids):
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            if is_prefill:
                self.stats["recomputed_prefill_tokens"] += max(0, min(end, seq.computed_high_watermark)-seq.num_cached_tokens)
            seq.computed_high_watermark = max(seq.computed_high_watermark, end)
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                seq.prefill_queued_at = self.clock()
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
