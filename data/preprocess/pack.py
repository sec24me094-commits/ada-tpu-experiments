#!/usr/bin/env python
"""Pack a flat tokenized corpus (.bin/.idx from tokenize.py) into fixed-length
training sequences, for efficient batching (no per-example padding waste).

Status: reference implementation — simple concatenate-and-chunk packing,
document boundaries are not respected (a sequence may span two documents,
standard practice for LM pretraining). If document-boundary-respecting
packing is needed instead, add a `--respect_doc_boundaries` flag here rather
than changing the on-disk format this produces.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path prefix from tokenize.py (reads <input>.bin)")
    parser.add_argument("--output", required=True, help="Output .npy path: (num_sequences, seq_len) uint32 array")
    parser.add_argument("--seq_len", type=int, default=2048)
    args = parser.parse_args()

    tokens = np.fromfile(f"{args.input}.bin", dtype=np.uint32)
    num_sequences = len(tokens) // args.seq_len
    trimmed = tokens[: num_sequences * args.seq_len]
    packed = trimmed.reshape(num_sequences, args.seq_len)

    np.save(args.output, packed)
    print(f"Packed {len(tokens):,} tokens -> {packed.shape} ({args.output})")


if __name__ == "__main__":
    main()
