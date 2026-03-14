#  Context Lab

A compact research repository for long-document, document-grounded question answering and summarization.

## Overview

This repository studies the following problem:

> Given a long document $D$ and a query $q$, construct a context $C(D,q)$ such that the generated answer is relevant, grounded, and efficient to produce.

Two systems are implemented:

1. **Baseline**: retrieve the top chunk and answer from it.
2. **Context-aware**: retrieve top candidate chunks, add a compact global document summary, and select context under a budget using an MMR-style rule.

The repository is accompanied by a technical report containing the formal problem formulation, mathematical description of the pipeline, metric definitions, and numerical results.

## Dataset

The project uses **QMSum**, a query-based meeting summarization dataset with:

- long multi-speaker transcripts,
- user-style information-seeking queries,
- grounded reference answers.

## Method

Let $D$ be a document, $q$ a query, and $y^\star$ a reference answer. The system constructs a packed context $C(D,q)$ and generates

$$
y = G(q, C(D,q)).
$$

### Baseline

The baseline uses a single retrieved chunk:

$$
C_{\mathrm{base}}(D,q)=d_{j^\star}, \qquad
j^\star=\arg\max_j s(q,d_j),
$$

where $s(q,d_j)$ is a dense retrieval score between the query and chunk $d_j$.

### Context-aware system

The context-aware system retrieves candidate chunks and solves a budgeted selection problem:

$$ S^\star = \arg\max_S \Big[ \sum_{d_i \in S} r_i - \lambda \sum_{d_i,d_j \in S} \mathrm{sim}(d_i,d_j) \Big] \quad \text{subject to} \quad \sum_{d_i \in S} \ell(d_i) \le B. $$

Here:

- $S$ is a selected subset of retrieved candidate chunks,
- $S^\star$ is the final selected subset,
- $d_i$ is the $i$-th chunk,
- $r_i$ is the relevance score of chunk $d_i$ with respect to query $q$,
- $\mathrm{sim}(d_i,d_j)$ is a redundancy score between chunks $d_i$ and $d_j$,
- $\lambda > 0$ controls the trade-off between relevance and redundancy,
- $\ell(d_i)$ is the length of chunk $d_i$,
- $B$ is the total context budget.

The first term rewards relevance, the second penalizes redundant context, and the constraint enforces a fixed budget.

The final context combines:

- selected local chunks,
- a lightweight global document summary.

## Evaluation

The repository reports:

- **ROUGE-L** against the reference answer,
- **support score** for grounding in retrieved context,
- **term recall** for salient reference terminology,
- **latency**.

This yields a small multi-objective evaluation setting balancing quality, grounding, and efficiency.

## Quick start

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# or
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
````

### Run experiments

```bash
python scripts/run_experiment.py --config configs/default.yaml --split train --max-samples 20 --system baseline
python scripts/run_experiment.py --config configs/default.yaml --split train --max-samples 20 --system context_aware
```

### Evaluate

```bash
python scripts/evaluate_predictions.py --predictions outputs/context_aware_train_predictions.jsonl --config configs/default.yaml
```

### Demo

```bash
python scripts/demo_example.py --config configs/default.yaml --example-index 0
```

## Purpose

The main purpose of this repository is to study whether adding compressed global document context improves long-document answering beyond a purely local retrieval baseline, and how this interacts with grounding and latency.

## Technical report

The accompanying technical report **Numerical_Optimization.pdf** provides:

* formal problem definition,
* mathematical derivation of the baseline and context-aware systems,
* metric definitions,
* dataset description,
* numerical results and interpretation.

## Repository structure

```text
wordcopilot-context-lab/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_experiment.py
│   ├── evaluate_predictions.py
│   └── demo_example.py
├── src/
│   └── context_lab/
│       ├── chunking.py
│       ├── config.py
│       ├── data.py
│       ├── eval.py
│       ├── generation.py
│       ├── pipeline.py
│       ├── retrieval.py
│       ├── selection.py
│       └── utils.py
└── tests/
    └── test_selection.py
```
