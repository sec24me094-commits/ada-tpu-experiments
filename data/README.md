# Dataset Pipeline

Status: **not started**. This directory documents the intended
tokenize -> filter -> deduplicate -> pack pipeline (see `preprocess/`); no
dataset has been processed through it yet. `[FILL]` markers throughout the
source documents (dataset name, tokenizer, size) need real values before
Phase 1 training can run against real data — until then, `scripts/train.py
--synthetic` is the only working data path.

## Intended pipeline order

```bash
python data/preprocess/filter.py --input <raw> --output <filtered>
python data/preprocess/deduplicate.py --input <filtered> --output <deduped>
python data/preprocess/tokenize.py --input <deduped> --output <tokenized> --tokenizer <name_or_path>
python data/preprocess/pack.py --input <tokenized> --output <packed> --seq_len 2048
```

See `DATASET_CARD.md` for what needs documenting once a real dataset is
chosen, and `stats/dataset_statistics.md` for the (currently empty) token
count / source-breakdown report.
