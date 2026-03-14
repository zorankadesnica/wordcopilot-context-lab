from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re

from .utils import safe_sentence_split


@dataclass
class Chunk:
    chunk_id: int
    text: str
    word_count: int


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "been", "being", "this", "that", "it", "as", "at", "by", "from",
    "we", "you", "they", "he", "she", "i", "me", "my", "our", "your", "their", "but", "if",
    "so", "do", "does", "did", "not", "can", "could", "would", "should", "about", "what",
}


def chunk_document(text: str, chunk_size_words: int = 140, stride_words: int = 40) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(1, chunk_size_words - stride_words)
    chunk_id = 0
    for start in range(0, len(words), step):
        end = min(len(words), start + chunk_size_words)
        chunk_words = words[start:end]
        if not chunk_words:
            continue
        chunks.append(Chunk(chunk_id=chunk_id, text=" ".join(chunk_words), word_count=len(chunk_words)))
        chunk_id += 1
        if end >= len(words):
            break
    return chunks


def top_document_terms(text: str, top_k: int = 25) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    counts = Counter(token for token in tokens if token not in _STOPWORDS)
    return [term for term, _ in counts.most_common(top_k)]


def build_global_summary(text: str, max_sentences: int = 8) -> str:
    sentences = safe_sentence_split(text)
    if not sentences:
        return ""

    terms = set(top_document_terms(text, top_k=40))
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        words = set(re.findall(r"[A-Za-z][A-Za-z\-]{2,}", sentence.lower()))
        overlap = len(words & terms)
        length_penalty = math.log(max(2, len(sentence.split())))
        score = overlap / length_penalty
        scored.append((score, sentence))

    top = [sent for _, sent in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]]
    ordered = [sent for sent in sentences if sent in set(top)]
    return " ".join(ordered[:max_sentences]).strip()
