from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from tests.test_block_manager import assert_consistent


def fill(bm, seq):
    bm.allocate(seq, bm.can_allocate(seq))
    seq.num_scheduled_tokens = len(seq)-seq.num_cached_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens


def test_prefix_refcount_and_reclaim(config):
    bm = BlockManager(5, 4)
    first, second = Sequence(list(range(8))), Sequence(list(range(8)))
    fill(bm, first)
    assert bm.can_allocate(second) == 1  # Last block is always recomputed for logits.
    fill(bm, second)
    assert first.block_table[0] == second.block_table[0]
    assert_consistent(bm, [first, second])
    bm.deallocate(first)
    assert_consistent(bm, [second])
    bm.deallocate(second)
    assert_consistent(bm, [])


def test_partial_chunk_never_publishes_unwritten_block(config):
    bm = BlockManager(6, 4)
    first = Sequence(list(range(8)))
    bm.allocate(first, 0)
    first.num_scheduled_tokens = 3
    bm.hash_blocks(first)
    assert not bm.hash_to_block_id
    first.num_cached_tokens = 3
    first.num_scheduled_tokens = 1
    bm.hash_blocks(first)
    assert bm.can_allocate(Sequence(list(range(8)))) == 1
    assert bm.can_allocate(Sequence([99]+list(range(1, 8)))) == 0


def test_eviction_removes_stale_hash(config):
    bm = BlockManager(2, 4)
    first = Sequence(list(range(8)))
    fill(bm, first)
    bm.deallocate(first)
    replacement = Sequence(list(range(20, 28)))
    fill(bm, replacement)
    bm.deallocate(replacement)
    assert bm.can_allocate(Sequence(list(range(8)))) == 0
    assert_consistent(bm, [])
