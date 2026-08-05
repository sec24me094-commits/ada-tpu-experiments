#!/usr/bin/env bash
# TPU VM environment setup. Run on a fresh TPU VM (TRC-allocated) via `gcloud
# compute tpus tpu-vm ssh`. UNTESTED — fill in the PyTorch/XLA version that
# matches the TPU runtime version TRC provisions before relying on this.
set -euo pipefail

echo "== ADA TPU VM setup =="

# Match this to the TPU VM's runtime image (check with: gcloud compute tpus
# tpu-vm describe <name> --zone=<zone> --format='value(acceleratorType,runtimeVersion)')
PYTORCH_XLA_VERSION="${PYTORCH_XLA_VERSION:-2.3.0}"

pip install --upgrade pip
pip install "torch~=2.3.0" "torch_xla[tpu]~=${PYTORCH_XLA_VERSION}" \
  -f https://storage.googleapis.com/libtpu-releases/index.html

cd "$(dirname "$0")/.."
pip install -e ".[tpu]"

python -c "import torch_xla.core.xla_model as xm; print('XLA device:', xm.xla_device())"

echo "== Setup complete =="
