from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_sentence_split(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    sentences: list[str] = []
    current = []
    for token in text.split():
        current.append(token)
        if token.endswith((".", "?", "!")):
            sentences.append(" ".join(current).strip())
            current = []
    if current:
        sentences.append(" ".join(current).strip())
    return sentences
