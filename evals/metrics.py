"""Ranking-quality metrics for the eval harness.

All functions take a `ranked_ids` list (best-first) and ground truth, and return
a score. See the README/design notes for intuition:
  - hit@k / MRR use BINARY relevance (a set of relevant ids).
  - NDCG@k uses GRADED relevance (id -> grade).
"""

import math


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant id is in the top k, else 0.0."""
    return 1.0 if any(cid in relevant_ids for cid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / rank of the first relevant id (0.0 if none present)."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Discounted cumulative gain over the top k (grade / log2(rank+1))."""
    return sum(
        relevance.get(cid, 0.0) / math.log2(rank + 1)
        for rank, cid in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Normalized DCG@k in [0, 1] (1.0 = ideal ordering). 0.0 if no positive grades."""
    ideal_grades = sorted(relevance.values(), reverse=True)
    idcg = sum(g / math.log2(rank + 1) for rank, g in enumerate(ideal_grades[:k], start=1) if g > 0)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / idcg


def rank_of(ranked_ids: list[str], target_id: str) -> int | None:
    """1-based rank of target_id in the list, or None if absent."""
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid == target_id:
            return rank
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
