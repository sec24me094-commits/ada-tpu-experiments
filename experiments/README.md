# Reproducing ADA Experiments

Status: all three phases below are **pending TPU resource allocation**
(see the TRC application). Nothing here has been run against real training
data — every `run.sh` currently launches `scripts/train.py --synthetic` as
a smoke test of the harness itself, not a real experiment.

## Phase 1 — Architecture Validation
```bash
bash experiments/phase1_validation/run.sh
```
ADA-Nano vs Transformer-Nano baseline, confirms stable training + routing
stats logging. See `phase1_validation/expected_outputs.md`.

## Phase 2 — Ablation Study
```bash
python scripts/ablation_launcher.py --config_dir configs/ --output_root checkpoints/ablation
```
6-variant matrix — see `docs/architecture.md`'s ablation table and
`phase2_ablation/expected_outputs.md`.

## Phase 3 — Scaling Study
```bash
bash experiments/phase3_scaling/run.sh
```
ADA-Small (300M) and ADA-Base (1B) vs matched Transformer baselines. See
`phase3_scaling/expected_outputs.md`.

## Once real results exist

Replace each `expected_outputs.md` with actual numbers (loss curves,
perplexity tables, routing stats) and wire `run.sh` to real data via
`--data_path` once `data/preprocess/` has produced a packed dataset.
