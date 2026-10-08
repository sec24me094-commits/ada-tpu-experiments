#!/usr/bin/env python
"""Train an efficient Byte-Level BPE Tokenizer for ADA on text or JSONL files.

Usage:
    # Train from raw text files:
    python scripts/train_tokenizer.py --input corpus.txt --output data/tokenizer --vocab_size 32000

    # Train from JSONL:
    python scripts/train_tokenizer.py --input corpus.jsonl --output data/tokenizer --vocab_size 32000 --jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from ada.tokenizer import ADABPETokenizer


def text_stream(file_path: Path, is_jsonl: bool) -> Iterator[str]:
    """Streams text lines/documents without loading entire file into memory."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if is_jsonl:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue
            else:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE tokenizer for ADA.")
    parser.add_argument("--input", required=True, help="Input text or JSONL file path")
    parser.add_argument("--output", required=True, help="Output directory to save tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size (default: 32000)")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--jsonl", action="store_true", help="Input is JSONL with {'text': ...} objects")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Training ADA Byte-Level BPE on {input_path} (vocab_size={args.vocab_size})...")
    tokenizer = ADABPETokenizer()
    stream = text_stream(input_path, is_jsonl=args.jsonl)
    tokenizer.train(stream, vocab_size=args.vocab_size, min_frequency=args.min_frequency)

    tokenizer.save(output_path)
    print(f"Tokenizer saved successfully to {output_path} (final vocab: {tokenizer.vocab_size:,})")


if __name__ == "__main__":
    main()
