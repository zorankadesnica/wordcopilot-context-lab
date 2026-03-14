from __future__ import annotations

import numpy as np

from context_lab.chunking import Chunk
from context_lab.retrieval import RetrievalResult
from context_lab.selection import mmr_select


def test_mmr_select_respects_budget_and_k() -> None:
    candidates = [
        RetrievalResult(chunk=Chunk(0, "a", 100), score=0.9, embedding=np.array([1.0, 0.0])),
        RetrievalResult(chunk=Chunk(1, "b", 100), score=0.8, embedding=np.array([0.9, 0.1])),
        RetrievalResult(chunk=Chunk(2, "c", 100), score=0.7, embedding=np.array([0.0, 1.0])),
    ]
    selected = mmr_select(
        candidates=candidates,
        query_embedding=np.array([1.0, 0.0]),
        final_top_k=2,
        token_budget_words=200,
        lambda_weight=0.8,
    )
    assert len(selected.selected) == 2
    assert selected.total_words <= 200
