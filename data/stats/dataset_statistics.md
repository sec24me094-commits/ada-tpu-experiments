# Dataset Statistics

Status: Target distributions for FineWeb-Edu 10BT Sample and SlimPajama-6B validation split.

## Token Counts & Splits
| Split | Total Tokens | Sequences (seq_len=2048) | Storage (uint32 .bin) |
|---|---|---|---|
| **Train (Phase 1 ADA-Nano)** | 5,000,000,000 (5.0B) | 2,441,406 | ~20 GB |
| **Train (Phase 2 Ablations)** | 5,000,000,000 (5.0B / run) | 2,441,406 | ~20 GB |
| **Train (Phase 3 Full Corpus)** | 10,000,000,000 (10.0B) | 4,882,812 | ~40 GB |
| **Validation Split** | 100,000,000 (100M) | 48,828 | ~400 MB |
| **Test Split** | 100,000,000 (100M) | 48,828 | ~400 MB |

## Preprocessing Pipeline Yield
- **Input Corpus**: ~12.5 Billion raw web tokens from FineWeb-Edu.
- **Quality Filtering Yield**: ~87.4% accepted (12.6% discarded due to low educational score < 3.0 or character ratio < 0.60).
- **Deduplication Rate**: ~6.8% duplicate/near-duplicate documents removed via MinHash LSH.
- **Final Packed Yield**: 10.0 Billion tokens across 4.88M packed sequences with 0% padding waste.

## Domain Distribution
- STEM & Scientific Literature: 34%
- Humanities, History & Social Sciences: 26%
- Technical Documentation & Verified Web: 22%
- General High-Quality Expository Web: 18%
