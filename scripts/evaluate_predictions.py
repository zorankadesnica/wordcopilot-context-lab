from __future__ import annotations

import argparse
import json
from pathlib import Path

from context_lab.config import Config
from context_lab.eval import SupportScorer, summarize_metrics
from context_lab.utils import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    rows = read_jsonl(args.predictions)
    support_scorer = SupportScorer(config.get("models", "retriever_name"))
    metrics = summarize_metrics(
        rows,
        support_scorer=support_scorer,
        top_terms_k=int(config.get("metrics", "top_terms_k", default=25)),
    )
    print(json.dumps(metrics, indent=2))

    out_path = Path(args.predictions).with_suffix(".metrics.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
