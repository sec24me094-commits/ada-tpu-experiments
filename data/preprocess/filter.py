#!/usr/bin/env python
"""Quality filtering for raw text documents (JSONL, one {"text": ...} per line).

Status: reference heuristics only — replace thresholds once a real corpus is
chosen (see ../DATASET_CARD.md's "[FILL: quality filter rules]").

Filters applied (each independently toggleable):
  --min_chars / --max_chars   length bounds
  --min_alpha_ratio           fraction of characters that must be alphabetic
                               (rejects code/markup-heavy or garbled text)
  --max_repeated_line_ratio   rejects documents dominated by repeated lines
                               (common boilerplate/scrape artifact)
"""

from __future__ import annotations

import argparse
import json


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


def repeated_line_ratio(text: str) -> float:
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return 1 - (len(set(lines)) / len(lines))


def passes_filters(text: str, args) -> bool:
    if not (args.min_chars <= len(text) <= args.max_chars):
        return False
    if alpha_ratio(text) < args.min_alpha_ratio:
        return False
    if repeated_line_ratio(text) > args.max_repeated_line_ratio:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL, one {'text': ...} per line")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_chars", type=int, default=200)
    parser.add_argument("--max_chars", type=int, default=1_000_000)
    parser.add_argument("--min_alpha_ratio", type=float, default=0.6)
    parser.add_argument("--max_repeated_line_ratio", type=float, default=0.3)
    args = parser.parse_args()

    kept, total = 0, 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            total += 1
            doc = json.loads(line)
            if passes_filters(doc.get("text", ""), args):
                fout.write(line)
                kept += 1

    print(f"Kept {kept}/{total} documents ({kept / max(1, total):.1%})")


if __name__ == "__main__":
    main()
