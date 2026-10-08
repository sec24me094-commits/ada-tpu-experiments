#!/usr/bin/env python
"""High-speed parallel tokenization of JSONL corpora into flat uint32 binary arrays.

Produces:
- <output>.bin: Flat contiguous array of uint32 token IDs
- <output>.idx: Document boundary offsets (uint64)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize JSONL corpus with Rust-accelerated batching.")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output path prefix (<output>.bin and <output>.idx)")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json or HF tokenizer name")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for multi-threaded encoding")
    parser.add_argument("--add_eos", action="store_true", default=True)
    args = parser.parse_args()

    # Load tokenizer (prefer ADABPETokenizer if local tokenizer.json exists, else AutoTokenizer)
    tokenizer_path = Path(args.tokenizer)
    if tokenizer_path.is_file() or (tokenizer_path.is_dir() and (tokenizer_path / "tokenizer.json").exists()):
        from ada.tokenizer import ADABPETokenizer
        tokenizer = ADABPETokenizer.from_file(args.tokenizer)
        is_ada = True
        eos_id = tokenizer.eos_token_id
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        is_ada = False
        eos_id = tokenizer.eos_token_id or 2

    offsets = [0]
    total_docs = 0

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, open(f"{args.output}.bin", "wb") as fout:
        batch_texts: List[str] = []

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
                if text:
                    batch_texts.append(text)
            except json.JSONDecodeError:
                continue

            if len(batch_texts) >= args.batch_size:
                if is_ada:
                    encodings = tokenizer.encode_batch(batch_texts)
                else:
                    encodings = [tokenizer.encode(t, add_special_tokens=False) for t in batch_texts]

                for token_ids in encodings:
                    if args.add_eos:
                        token_ids = token_ids + [eos_id]
                    arr = np.array(token_ids, dtype=np.uint32)
                    fout.write(arr.tobytes())
                    offsets.append(offsets[-1] + len(arr))

                total_docs += len(batch_texts)
                batch_texts = []

        # Process remainder
        if batch_texts:
            if is_ada:
                encodings = tokenizer.encode_batch(batch_texts)
            else:
                encodings = [tokenizer.encode(t, add_special_tokens=False) for t in batch_texts]

            for token_ids in encodings:
                if args.add_eos:
                    token_ids = token_ids + [eos_id]
                arr = np.array(token_ids, dtype=np.uint32)
                fout.write(arr.tobytes())
                offsets.append(offsets[-1] + len(arr))
            total_docs += len(batch_texts)

    np.array(offsets, dtype=np.uint64).tofile(f"{args.output}.idx")
    print(f"Tokenized {total_docs:,} documents, {offsets[-1]:,} tokens -> {args.output}.bin and {args.output}.idx")


if __name__ == "__main__":
    main()

