import json

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

baseline = load_jsonl("outputs/baseline_train_predictions.jsonl")
context_aware = load_jsonl("outputs/context_aware_train_predictions.jsonl")

for b, c in zip(baseline, context_aware):
    print("=" * 100)
    print("EXAMPLE ID:", b["example_id"])
    print("QUERY:")
    print(b["query"])
    print()
    print("REFERENCE:")
    print(b["reference"])
    print()
    print("BASELINE PREDICTION:")
    print(b["prediction"])
    print()
    print("CONTEXT-AWARE PREDICTION:")
    print(c["prediction"])
    print()
    print("BASELINE CHUNKS:", b["selected_chunk_ids"])
    print("CONTEXT-AWARE CHUNKS:", c["selected_chunk_ids"])
    print("BASELINE LATENCY:", b["latency_sec"])
    print("CONTEXT-AWARE LATENCY:", c["latency_sec"])
    print()