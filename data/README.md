
# Data

This repository uses a real public dataset at runtime instead of checking large files into Git.

Default dataset: **QMSum**.

## Setup

Create an `external/` directory in the project root and clone the official QMSum repository there:

```bash
mkdir -p external
git clone https://github.com/Yale-LILY/QMSum.git external/QMSum
````

On Windows PowerShell:

```powershell
mkdir external
git clone https://github.com/Yale-LILY/QMSum.git external/QMSum
```

After cloning, copy the JSONL split files from:

```text
external/QMSum/data/ALL/jsonl/
```

to:

```text
external/QMSum/data/ALL/
```

so that the final structure is:

```text
external/
└── QMSum/
    └── data/
        └── ALL/
            ├── train.jsonl
            ├── val.jsonl
            └── test.jsonl
```

## Notes

* The dataset is **not** committed to this repository.
* Add `external/` to `.gitignore`.
* The loader reads the local QMSum files and flattens each meeting into `(document, query, reference answer)` examples.

```
```
