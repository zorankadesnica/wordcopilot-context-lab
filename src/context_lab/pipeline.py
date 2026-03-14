from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .chunking import build_global_summary, chunk_document
from .config import Config
from .data import Example
from .generation import HFGenerator, build_baseline_prompt, build_context_aware_prompt
from .retrieval import EmbeddingRetriever
from .selection import mmr_select


@dataclass
class PredictionRow:
    example_id: str
    document_id: str
    system: str
    query: str
    reference: str
    prediction: str
    document: str
    packed_context: str
    global_summary: str
    selected_chunk_ids: list[int]
    latency_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextLabPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.retriever = EmbeddingRetriever(config.get("models", "retriever_name"))
        self.generator = HFGenerator(config.get("models", "generator_name"))

    def run_example(self, example: Example, system: str) -> PredictionRow:
        chunk_size = int(self.config.get("chunking", "chunk_size_words", default=140))
        stride = int(self.config.get("chunking", "stride_words", default=40))
        initial_top_k = int(self.config.get("retrieval", "initial_top_k", default=6))
        final_top_k = int(self.config.get("retrieval", "final_top_k", default=3))
        token_budget_words = int(self.config.get("retrieval", "token_budget_words", default=500))
        lambda_weight = float(self.config.get("retrieval", "mmr_lambda", default=0.75))
        max_new_tokens = int(self.config.get("prompting", "max_new_tokens", default=128))
        temperature = float(self.config.get("prompting", "temperature", default=0.0))
        max_global_sentences = int(self.config.get("chunking", "max_global_sentences", default=8))

        chunks = chunk_document(example.document, chunk_size_words=chunk_size, stride_words=stride)
        retrieved = self.retriever.retrieve(example.query, chunks, top_k=initial_top_k)
        if not retrieved:
            packed_context = ""
            global_summary = ""
            selected_chunk_ids: list[int] = []
        elif system == "baseline":
            packed_context = retrieved[0].chunk.text
            global_summary = ""
            selected_chunk_ids = [retrieved[0].chunk.chunk_id]
        else:
            global_summary = build_global_summary(example.document, max_sentences=max_global_sentences)
            query_embedding = self.retriever.encode_texts([example.query])[0]
            selected = mmr_select(
                candidates=retrieved,
                query_embedding=query_embedding,
                final_top_k=final_top_k,
                token_budget_words=token_budget_words,
                lambda_weight=lambda_weight,
            )
            packed_context = "\n\n".join(item.chunk.text for item in selected.selected)
            selected_chunk_ids = [item.chunk.chunk_id for item in selected.selected]

        if system == "baseline":
            prompt = build_baseline_prompt(query=example.query, context=packed_context)
        else:
            prompt = build_context_aware_prompt(
                query=example.query,
                global_summary=global_summary,
                local_context=packed_context,
            )

        output = self.generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        return PredictionRow(
            example_id=example.example_id,
            document_id=example.document_id,
            system=system,
            query=example.query,
            reference=example.reference,
            prediction=output.text,
            document=example.document,
            packed_context=packed_context,
            global_summary=global_summary,
            selected_chunk_ids=selected_chunk_ids,
            latency_sec=output.latency_sec,
        )
