# Dataset Card — FineWeb-Edu (10BT Sample) & SlimPajama-6B Curated

Status: Standardized pre-training corpus specification for ADA validation, ablations, and scaling.

## Source
- **Primary Corpus**: [FineWeb-Edu (10B token sample)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (`sample-10BT`)
  - License: Open Data Commons Attribution License (ODC-By 1.0)
  - Citation: Lozhkov et al. (2024), "FineWeb: Decanting the Web for the Most Challenging Pre-Training Tasks"
- **Secondary Evaluation Corpus**: [SlimPajama-6B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) (Cerebras, Apache 2.0)

## Preprocessing Pipeline
- **Filtering (`data/preprocess/filter.py`)**:
  - Educational classifier score: $\ge 3.0$ (filters out low-information web scrape noise).
  - Length bounds: minimum 200 characters, maximum 1,000,000 characters.
  - Alphabetic character ratio: $\ge 0.60$ (rejects binary/hexdump, garbled syntax, machine artifacts).
  - Repetitive lines ratio: $\le 0.30$ (eliminates web crawler loop artifacts and boilerplate headers/footers).
- **Deduplication (`data/preprocess/deduplicate.py`)**:
  - Exact-duplicate pass: MD5 hash indexing on normalized text.
  - Near-duplicate pass: MinHash LSH with 5-gram shingles, Jaccard similarity threshold 0.80.
- **Tokenizer (`data/preprocess/tokenize.py`)**:
  - Byte-Pair Encoding (BPE) with vocabulary size 32,000 matching `ADAConfig.vocab_size = 32000`.
  - Reference tokenizer: `mistralai/Mistral-7B-v0.1` / `meta-llama/Llama-2-7b-hf` compatible vocabulary.
  - Output format: Flat 32-bit unsigned integer array (`.bin`) with document offset index (`.idx`).
- **Sequence Packing (`data/preprocess/pack.py`)**:
  - Chunk length: $L = 2048$ tokens (matching `ADAConfig.max_seq_len = 2048`).
  - Boundary strategy: Continuous `<eos>` concatenation and chunking to eliminate zero-padding compute waste.

## Splits
- **Train Split**: 9.8 Billion tokens (~4,785,000 packed sequences of length 2048).
- **Validation Split**: 100 Million tokens (~48,828 packed sequences).
- **Test / Benchmark Split**: 100 Million held-out tokens + standard zero-shot downstream tasks (LAMBADA, HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande).

## Known Limitations
- Primary language coverage is English (>98%).
- Content skews towards academic, technical, educational, and high-quality web expository text.
- Standard web-crawled biases may persist despite classifier-based educational filtering.
