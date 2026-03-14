from __future__ import annotations

import argparse
from pathlib import Path

from context_lab.config import Config
from context_lab.data import load_qmsum_examples
from context_lab.pipeline import ContextLabPipeline
from context_lab.utils import ensure_dir, set_seed, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--system", type=str, choices=["baseline", "context_aware"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    set_seed(int(config.get("runtime", "seed", default=42)))

    examples = load_qmsum_examples(split=args.split, max_samples=args.max_samples)
    pipeline = ContextLabPipeline(config)

    rows = []
    for example in examples:
        rows.append(pipeline.run_example(example, system=args.system).to_dict())

    outputs_dir = ensure_dir("outputs")
    path = Path(outputs_dir) / f"{args.system}_{args.split}_predictions.jsonl"
    write_jsonl(path, rows)
    print(f"Saved {len(rows)} predictions to {path}")


if __name__ == "__main__":
    main()
