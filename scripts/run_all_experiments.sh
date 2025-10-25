#!/usr/bin/env bash
set -euo pipefail

# Master script to run joint grid (k x n) for GD and SGD,
# plus separate projections if desired.

SCRIPT_DIR="/workspace/scripts"

# Ensure dependencies are installed (optional quick check)
if ! python -c 'import torch, numpy, matplotlib, tqdm' >/dev/null 2>&1; then
  echo "[INFO] Installing Python dependencies from requirements.txt" >&2
  pip install -r /workspace/requirements.txt
fi

# Joint grid runs
"${SCRIPT_DIR}/run_joint_dependency.sh" gd
"${SCRIPT_DIR}/run_joint_dependency.sh" sgd

# Optional: individual dependency runs (uncomment if needed)
# "${SCRIPT_DIR}/run_m_dependency.sh" gd
# "${SCRIPT_DIR}/run_m_dependency.sh" sgd
# "${SCRIPT_DIR}/run_k_dependency.sh" gd
# "${SCRIPT_DIR}/run_k_dependency.sh" sgd

echo "All experiments completed."