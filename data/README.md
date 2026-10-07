# Dataset Pipeline
 
Status: **Ready**. The pipeline ingests, filters, deduplicates, tokenizes, and packs pre-training corpora (e.g. FineWeb-Edu 10BT or SlimPajama-6B) into fixed-length arrays for zero-padding training.

## Preprocessing Pipeline

```bash
# 1. Quality filter raw documents
python data/preprocess/filter.py --input data/raw/corpus.jsonl --output data/filtered/corpus.jsonl

# 2. Near-duplicate removal via MinHash LSH
python data/preprocess/deduplicate.py --input data/filtered/corpus.jsonl --output data/deduped/corpus.jsonl

# 3. Tokenize with reference BPE tokenizer (vocab 32k)
python data/preprocess/tokenize.py --input data/deduped/corpus.jsonl --output data/tokenized/corpus --tokenizer mistralai/Mistral-7B-v0.1

# 4. Pack into fixed 2048-token training chunks
python data/preprocess/pack.py --input data/tokenized/corpus --output data/packed/train_2048.npy --seq_len 2048
```

See `DATASET_CARD.md` for full dataset specifications and provenance, and `stats/dataset_statistics.md` for distribution targets.
