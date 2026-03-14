from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .retrieval import RetrievalResult


@dataclass
class SelectedContext:
    selected: list[RetrievalResult]
    total_words: int



def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)



def mmr_select(
    candidates: Iterable[RetrievalResult],
    query_embedding: np.ndarray,
    final_top_k: int,
    token_budget_words: int,
    lambda_weight: float = 0.75,
) -> SelectedContext:
    remaining = list(candidates)
    selected: list[RetrievalResult] = []
    total_words = 0

    while remaining and len(selected) < final_top_k:
        best_idx = None
        best_score = float("-inf")

        for idx, candidate in enumerate(remaining):
            if total_words + candidate.chunk.word_count > token_budget_words:
                continue

            relevance = _cosine(candidate.embedding, query_embedding)
            redundancy = 0.0
            if selected:
                redundancy = max(_cosine(candidate.embedding, prev.embedding) for prev in selected)
            score = lambda_weight * relevance - (1.0 - lambda_weight) * redundancy
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        total_words += chosen.chunk.word_count

    return SelectedContext(selected=selected, total_words=total_words)
