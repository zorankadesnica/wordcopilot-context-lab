from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    embedding: np.ndarray


class EmbeddingRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))

    def retrieve(self, query: str, chunks: Sequence[Chunk], top_k: int = 6) -> list[RetrievalResult]:
        if not chunks:
            return []
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_emb = self.encode_texts(chunk_texts)
        query_emb = self.encode_texts([query])[0]
        scores = chunk_emb @ query_emb
        order = np.argsort(-scores)[:top_k]
        return [
            RetrievalResult(
                chunk=chunks[int(idx)],
                score=float(scores[int(idx)]),
                embedding=chunk_emb[int(idx)],
            )
            for idx in order
        ]
