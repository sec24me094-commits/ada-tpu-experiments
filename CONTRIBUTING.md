# Contributing to ADA

Thanks for your interest in the Adaptive Depth Architecture (ADA) project.
This is an independent research project; contributions, issues, and
questions are welcome.

## Getting started

```bash
git clone https://github.com/sec24me094-commits/ada-tpu-experiments
cd ada-tpu-experiments
pip install -e ".[dev]"
pytest tests/ -v
```

## Development workflow

1. Open an issue first for anything beyond a trivial fix (typo, docs) so we
   can agree on the approach before you spend time on it.
2. Fork the repo and create a feature branch off `main`.
3. Keep changes scoped — one logical change per PR.
4. Add or update unit tests for any change to `ada/`.
5. Run `pytest tests/ -v`, `ruff check ada/`, and `mypy ada/ --ignore-missing-imports`
   before opening a PR — the CI workflow runs the same checks.
6. Fill in a description of *why*, not just *what*, in your PR.

## Code style

- Python 3.11+, type hints on public functions.
- `ruff` for linting, `mypy` for type checking (see `.github/workflows/ci.yml`).
- Keep tensors documented with shape comments, e.g. `# (batch, seq_len, hidden)`.

## Research questions vs. bugs

If you have a question about the architecture, training results, or how to
reproduce an experiment, please use the "Research Question" issue template
rather than the bug template — it helps keep the tracker organized.

## Reporting issues

Please include:
- Python / PyTorch / (if relevant) PyTorch-XLA versions
- The config file used (`configs/*.yaml`)
- A minimal repro if possible
