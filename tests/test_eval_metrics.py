import math

from evals.metrics import (
    dcg_at_k,
    hit_at_k,
    ndcg_at_k,
    rank_of,
    reciprocal_rank,
)

RANKED = ["C1", "C4", "C2", "C5", "C3"]  # relevant candidate is C4 (rank 2)


def test_hit_at_k():
    relevant = {"C4"}
    assert hit_at_k(RANKED, relevant, 1) == 0.0
    assert hit_at_k(RANKED, relevant, 2) == 1.0
    assert hit_at_k(RANKED, relevant, 5) == 1.0
    assert hit_at_k(RANKED, {"Zzz"}, 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, {"C4"}) == 0.5      # rank 2
    assert reciprocal_rank(RANKED, {"C1"}) == 1.0      # rank 1
    assert reciprocal_rank(RANKED, {"C3"}) == 1 / 5    # rank 5
    assert reciprocal_rank(RANKED, {"absent"}) == 0.0


def test_rank_of():
    assert rank_of(RANKED, "C1") == 1
    assert rank_of(RANKED, "C4") == 2
    assert rank_of(RANKED, "absent") is None


def test_ndcg_perfect_and_worked_example():
    relevance = {"A": 3, "B": 1, "C": 0}
    # perfect ordering -> 1.0
    assert ndcg_at_k(["A", "B", "C"], relevance, 3) == 1.0
    # worked example B, A, C -> ~0.80
    ndcg = ndcg_at_k(["B", "A", "C"], relevance, 3)
    assert abs(ndcg - 0.7967) < 0.01


def test_ndcg_empty_relevance_is_zero():
    assert ndcg_at_k(["A", "B"], {"A": 0, "B": 0}, 2) == 0.0
    assert ndcg_at_k(["A", "B"], {}, 2) == 0.0


def test_dcg_discount():
    # single grade-1 item at rank 2 -> 1/log2(3)
    assert abs(dcg_at_k(["x", "hit"], {"hit": 1.0}, 2) - (1.0 / math.log2(3))) < 1e-9
