#!/usr/bin/env python
"""Tokenize a JSONL corpus into a flat token-id array per document, saved as
a single .bin (uint32) file + a .idx file of document boundary offsets.

Status: reference implementation using a HuggingFace tokenizer. The actual
tokenizer choice is a `[FILL]` in the dataset card — pass it via
--tokenizer (any name/path `AutoTokenizer.from_pretrained` accepts).
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL, one {'text': ...} per line")
    parser.add_argument("--output", required=True, help="Output path prefix (writes <output>.bin and <output>.idx)")
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer name or local path")
    parser.add_argument("--add_eos", action="store_true", default=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tokenizer.eos_token_id

    offsets = [0]
    with open(args.input, "r") as fin, open(f"{args.output}.bin", "wb") as fout:
        for line in fin:
            doc = json.loads(line)
            ids = tokenizer.encode(doc.get("text", ""))
            if args.add_eos and eos_id is not None:
                ids = ids + [eos_id]
            arr = np.array(ids, dtype=np.uint32)
            fout.write(arr.tobytes())
            offsets.append(offsets[-1] + len(arr))

    np.array(offsets, dtype=np.uint64).tofile(f"{args.output}.idx")
    print(f"Tokenized {len(offsets) - 1} documents, {offsets[-1]:,} tokens -> {args.output}.bin")


if __name__ == "__main__":
    main()
