"""Ranking metrics shared by retrieval and reranking experiments."""

from __future__ import annotations

from collections.abc import Iterable
from math import log2

import numpy as np


def recall_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Return the fraction of queries whose first relevant item is in top-k."""
    values = list(ranks)
    if not values:
        return 0.0
    return float(np.mean([rank is not None and rank <= k for rank in values]))


def mrr_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Return mean reciprocal rank, with ranks beyond k counted as zero."""
    values = list(ranks)
    if not values:
        return 0.0
    return float(
        np.mean([1.0 / rank if rank is not None and rank <= k else 0.0 for rank in values])
    )


def ndcg_at_k(ranks: Iterable[int | None], k: int) -> float:
    """Return nDCG@k for one known relevant context per query.

    With exactly one relevant context, the ideal DCG is 1.0 and the DCG for a
    relevant context at rank r is 1 / log2(r + 1).
    """
    values = list(ranks)
    if not values:
        return 0.0
    return float(
        np.mean(
            [
                1.0 / log2(rank + 1)
                if rank is not None and rank <= k
                else 0.0
                for rank in values
            ]
        )
    )


def ranking_summary(ranks: Iterable[int | None], cutoffs: Iterable[int] = (1, 5, 10)) -> dict[str, float]:
    """Build a consistently named Recall/MRR/nDCG metric dictionary."""
    rank_list = list(ranks)
    cutoffs = tuple(cutoffs)
    if not cutoffs:
        return {}
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in cutoffs):
        raise ValueError("cutoffs must contain positive integers.")
    summary: dict[str, float] = {}
    for k in cutoffs:
        summary[f"recall_at_{k}"] = recall_at_k(rank_list, k)
    max_cutoff = max(cutoffs)
    summary[f"mrr_at_{max_cutoff}"] = mrr_at_k(rank_list, max_cutoff)
    summary[f"ndcg_at_{max_cutoff}"] = ndcg_at_k(rank_list, max_cutoff)
    return summary
