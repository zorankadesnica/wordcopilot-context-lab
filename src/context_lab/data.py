from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


@dataclass
class Example:
    example_id: str
    document_id: str
    query: str
    reference: str
    document: str
    source: str = "qmsum"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _join_transcript(record: dict[str, Any]) -> str:
    transcript = record.get("meeting_transcripts") or record.get("meeting_transcript") or []
    if isinstance(transcript, str):
        return transcript

    parts: list[str] = []
    for item in transcript:
        if isinstance(item, dict):
            speaker = item.get("speaker", "")
            content = item.get("content", "")
            if speaker and content:
                parts.append(f"{speaker}: {content}")
            elif content:
                parts.append(content)
        elif isinstance(item, str):
            parts.append(item)

    return "\n".join(parts).strip()


def _extract_queries(record: dict[str, Any]) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []

    for key, prefix in (("general_query_list", "general"), ("specific_query_list", "specific")):
        items = record.get(key, [])
        if not isinstance(items, list):
            continue

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            query = " ".join(str(item.get("query", "")).split())
            answer = " ".join(str(item.get("answer", "")).split())

            if query and answer:
                result.append((f"{prefix}_{idx}", query, answer))

    return result


def _resolve_qmsum_file(split: str, data_dir: str | Path | None = None) -> Path:
    """
    Expected structure:
      external/QMSum/data/ALL/
        train.jsonl
        val.jsonl
        test.jsonl
    """
    if data_dir is None:
        data_dir = Path("external") / "QMSum" / "data" / "ALL"
    else:
        data_dir = Path(data_dir)

    split_map = {
        "train": "train.jsonl",
        "validation": "val.jsonl",
        "val": "val.jsonl",
        "test": "test.jsonl",
    }

    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Use train / validation / test.")

    path = data_dir / split_map[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find QMSum file at {path}. "
            f"Make sure train.jsonl / val.jsonl / test.jsonl are in external/QMSum/data/ALL/"
        )
    return path


def load_qmsum_examples(
    split: str = "validation",
    max_samples: int | None = None,
    data_dir: str | Path | None = None,
) -> list[Example]:
    path = _resolve_qmsum_file(split, data_dir=data_dir)
    examples: list[Example] = []

    with path.open("r", encoding="utf-8") as f:
        for rec_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            document = _join_transcript(record)
            if not document:
                continue

            document_id = str(record.get("id") or record.get("meeting_id") or f"doc_{rec_idx}")

            for local_id, query, reference in _extract_queries(record):
                examples.append(
                    Example(
                        example_id=f"{document_id}::{local_id}",
                        document_id=document_id,
                        query=query,
                        reference=reference,
                        document=document,
                    )
                )

                if max_samples is not None and len(examples) >= max_samples:
                    return examples

    return examples