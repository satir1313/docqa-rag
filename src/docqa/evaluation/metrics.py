"""Pure retrieval metrics (binary relevance). Kept free of project types so they are trivially
unit-testable against hand-computed values."""

from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank(flags: Sequence[bool], k: int) -> float:
    """1/rank of the first relevant item within the top k, else 0."""
    for rank, relevant in enumerate(flags[:k], start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(flags: Sequence[bool], total_relevant: int, k: int) -> float:
    """Binary-relevance nDCG@k. `total_relevant` is how many relevant items exist overall."""
    if total_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(flags[:k], start=1) if rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))
    return dcg / ideal


def ref_recall_at_k(best_ranks: Sequence[int | None], k: int) -> float:
    """Fraction of required references found in the top k.

    `best_ranks[i]` is the 1-based rank at which reference i was first matched (None = never).
    """
    if not best_ranks:
        return 0.0
    return sum(1 for r in best_ranks if r is not None and r <= k) / len(best_ranks)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lower, upper = math.floor(pos), math.ceil(pos)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
