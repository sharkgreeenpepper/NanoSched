import pytest
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


def assert_consistent(bm, sequences):
    refs = {i: sum(s.block_table.count(i) for s in sequences) for i in range(len(bm.blocks))}
    assert len(set(bm.free_block_ids)) == len(bm.free_block_ids)
    assert set(bm.free_block_ids).isdisjoint(bm.used_block_ids)
    assert set(bm.free_block_ids) | bm.used_block_ids == set(range(len(bm.blocks)))
    for block in bm.blocks:
        assert block.ref_count == refs[block.block_id]
        assert (block.block_id in bm.used_block_ids) == (block.ref_count > 0)


def test_allocation_release_and_block_boundary(config):
    bm = BlockManager(3, 4)
    seq = Sequence([1, 2, 3, 4])
    bm.allocate(seq, bm.can_allocate(seq))
    assert_consistent(bm, [seq])
    seq.append_token(9)
    assert bm.can_append(seq)
    bm.may_append(seq)
    assert len(seq.block_table) == 2
    assert_consistent(bm, [seq])
    bm.deallocate(seq)
    assert_consistent(bm, [])
    assert len(bm.free_block_ids) == 3


def test_full_prompt_reservation_is_still_eager(config):
    bm = BlockManager(2, 4)
    seq = Sequence(list(range(12)))
    assert bm.can_allocate(seq) == -1  # Incremental reservation is deliberately Phase 3.
