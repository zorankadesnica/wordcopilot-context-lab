from __future__ import annotations

import re
from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from .chunking import top_document_terms
from .utils import safe_sentence_split


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


class SupportScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def score(self, prediction: str, context: str) -> float:
        pred_sentences = [s for s in safe_sentence_split(prediction) if s.strip()]
        ctx_sentences = [s for s in safe_sentence_split(context) if s.strip()]
        if not pred_sentences or not ctx_sentences:
            return 0.0

        pred_emb = np.asarray(self.model.encode(pred_sentences, normalize_embeddings=True, show_progress_bar=False))
        ctx_emb = np.asarray(self.model.encode(ctx_sentences, normalize_embeddings=True, show_progress_bar=False))
        sims = pred_emb @ ctx_emb.T
        return float(np.mean(np.max(sims, axis=1)))



def term_recall(prediction: str, document: str, top_k: int = 25) -> float:
    top_terms = set(top_document_terms(document, top_k=top_k))
    if not top_terms:
        return 0.0
    pred_terms = set(re.findall(r"[A-Za-z][A-Za-z\-]{2,}", prediction.lower()))
    return float(len(top_terms & pred_terms) / len(top_terms))



def evaluate_row(row: dict[str, Any], support_scorer: SupportScorer, top_terms_k: int = 25) -> dict[str, float]:
    reference = row["reference"]
    prediction = row["prediction"]
    context = row.get("packed_context", "")
    document = row["document"]

    rouge_l = _ROUGE.score(reference, prediction)["rougeL"].fmeasure
    support = support_scorer.score(prediction, context)
    terms = term_recall(prediction, document, top_k=top_terms_k)
    latency = float(row.get("latency_sec", 0.0))

    return {
        "rougeL": rouge_l,
        "support": support,
        "term_recall": terms,
        "latency_sec": latency,
    }



def summarize_metrics(rows: list[dict[str, Any]], support_scorer: SupportScorer, top_terms_k: int = 25) -> dict[str, float]:
    metric_rows = [evaluate_row(row, support_scorer=support_scorer, top_terms_k=top_terms_k) for row in rows]
    if not metric_rows:
        return {"rougeL": 0.0, "support": 0.0, "term_recall": 0.0, "latency_sec": 0.0}
    keys = metric_rows[0].keys()
    return {key: float(np.mean([item[key] for item in metric_rows])) for key in keys}
