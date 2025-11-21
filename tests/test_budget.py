import pytest

from rag_cost_governor.budget import Budget


def test_reserve_then_settle_tracks_actual():
    b = Budget(ceiling_usd=0.10)
    assert b.try_reserve(0.03)
    assert b.remaining() == pytest.approx(0.07)
    b.settle(0.03, 0.025)
    assert b.remaining() == pytest.approx(0.075)
    assert not b.exhausted


def test_reserve_refused_when_cap_would_blow():
    b = Budget(ceiling_usd=0.05)
    assert b.try_reserve(0.04)
    assert not b.try_reserve(0.02)
    assert b.exhausted


def test_outstanding_reserves_block_further_reserves():
    b = Budget(ceiling_usd=0.10)
    assert b.try_reserve(0.08)
    # 0.02 free, 0.08 outstanding; a second 0.03 must be refused.
    assert not b.try_reserve(0.03)


def test_settle_cannot_push_outstanding_negative():
    b = Budget(ceiling_usd=0.10)
    b.try_reserve(0.02)
    b.settle(0.05, 0.01)  # over-settles the reserve; outstanding floored at 0
    assert b.outstanding_usd == 0.0


def test_negative_reserve_rejected():
    b = Budget(ceiling_usd=0.10)
    with pytest.raises(ValueError):
        b.try_reserve(-0.01)
