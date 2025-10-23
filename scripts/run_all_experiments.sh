#!/usr/bin/env bash
set -euo pipefail

# Master script to run both dependencies for GD and SGD.

SCRIPT_DIR="/workspace/scripts"

# Ensure dependencies are installed (optional quick check)
if ! python -c 'import torch, numpy, matplotlib, tqdm' >/dev/null 2>&1; then
  echo "[INFO] Installing Python dependencies from requirements.txt" >&2
  pip install -r /workspace/requirements.txt
fi

# m-dependency: med vs n (k fixed)
"${SCRIPT_DIR}/run_m_dependency.sh" gd
"${SCRIPT_DIR}/run_m_dependency.sh" sgd

# k-dependency: med vs k (n fixed)
"${SCRIPT_DIR}/run_k_dependency.sh" gd
"${SCRIPT_DIR}/run_k_dependency.sh" sgd

echo "All experiments completed."