from __future__ import annotations

import argparse

from context_lab.config import Config
from context_lab.data import load_qmsum_examples
from context_lab.pipeline import ContextLabPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--split", type=str, default="validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_qmsum_examples(split=args.split, max_samples=args.example_index + 1)
    example = examples[args.example_index]
    config = Config.from_yaml(args.config)
    pipeline = ContextLabPipeline(config)

    baseline = pipeline.run_example(example, system="baseline")
    context_aware = pipeline.run_example(example, system="context_aware")

    print("=" * 80)
    print("QUERY")
    print(example.query)
    print("\nREFERENCE")
    print(example.reference)
    print("\nBASELINE")
    print(baseline.prediction)
    print("\nCONTEXT AWARE")
    print(context_aware.prediction)
    print("\nSELECTED CHUNKS")
    print(context_aware.selected_chunk_ids)
    print("=" * 80)


if __name__ == "__main__":
    main()
