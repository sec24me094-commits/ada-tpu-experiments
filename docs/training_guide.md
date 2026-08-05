# Training Guide

## Quick start (smoke test, no real data required)

```bash
pip install -e ".[dev]"
python scripts/train.py --config configs/ada_nano.yaml \
    --output_dir /tmp/ada-nano-smoke --synthetic --max_steps 20
```

This trains on random tokens — useful for confirming the loop runs
end-to-end (forward, loss, backward, checkpoint), not for producing a real
model.

## Training on real data

Not wired up yet. `scripts/train.py --data_path ...` currently raises
`NotImplementedError` — see that script's module docstring for the expected
pipeline (`data/preprocess/tokenize.py` -> `pack.py` output format).

## Key hyperparameters

See `docs/hyperparameter_guide.md` for what each config field means.
`configs/ada_nano.yaml` / `ada_small.yaml` / `ada_base.yaml` hold the three
reference sizes from the roadmap; sizes are first estimates — run
`python scripts/profile_flops.py --config <path>` to check actual parameter
count and adjust `hidden_dim`/`num_layers` if needed.

## Distributed / multi-GPU

```bash
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/ada_nano.yaml \
    --data_path /path/to/tokenized/dataset \
    --output_dir checkpoints/ada-nano
```

Note: `ada/training/trainer.py`'s `Trainer` is currently single-device —
DDP wrapping (`torch.nn.parallel.DistributedDataParallel`) is not yet added
to `Trainer.__init__`. This is a straightforward follow-up
(wrap `self.model` and gate `dist.init_process_group` on `torchrun`'s env
vars) but isn't done in this scaffold.

## TPU

See `tpu/README.md` — status: not started, no TPU access yet.

## Resuming from a checkpoint

```bash
python scripts/train.py --config configs/ada_nano.yaml \
    --output_dir checkpoints/ada-nano --resume --synthetic
```

Loads `checkpoints/ada-nano/latest.pt` if present (see
`ada/utils/checkpointing.py`).
