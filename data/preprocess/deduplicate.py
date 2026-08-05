#!/usr/bin/env python
"""Near-duplicate removal via shingled MinHash-style Jaccard estimation.

Status: reference implementation — a simplified single-hash-per-shingle-set
approach, adequate for small/medium corpora. For corpus sizes where an
exact pairwise comparison is infeasible, swap in `datasketch.MinHashLSH` (or
similar) behind the same `--input/--output` CLI without changing callers.
"""

from __future__ import annotations

import argparse
import json
from hashlib import md5


def shingles(text: str, n: int = 5) -> set[str]:
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shingle_size", type=int, default=5)
    parser.add_argument("--jaccard_threshold", type=float, default=0.8)
    args = parser.parse_args()

    docs = []
    with open(args.input, "r") as fin:
        for line in fin:
            doc = json.loads(line)
            docs.append(doc)

    # Exact-duplicate pass first (cheap, hash-based).
    seen_hashes: set[str] = set()
    deduped = []
    for doc in docs:
        h = md5(doc.get("text", "").encode("utf-8")).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        deduped.append(doc)

    # Near-duplicate pass (O(n^2) shingle Jaccard — fine for small/medium
    # corpora only; see module docstring for the scaling note).
    kept = []
    kept_shingles: list[set[str]] = []
    for doc in deduped:
        s = shingles(doc.get("text", ""), args.shingle_size)
        if any(jaccard(s, ks) >= args.jaccard_threshold for ks in kept_shingles):
            continue
        kept.append(doc)
        kept_shingles.append(s)

    with open(args.output, "w") as fout:
        for doc in kept:
            fout.write(json.dumps(doc) + "\n")

    print(f"Input: {len(docs)}  Exact-dedup: {len(deduped)}  Near-dedup: {len(kept)}")


if __name__ == "__main__":
    main()
