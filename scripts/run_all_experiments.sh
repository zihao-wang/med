#!/usr/bin/env bash
set -euo pipefail

# Master script to run all experiment sets.
# Allows overriding environment variables for specific scripts.

SCRIPT_DIR="/workspace/scripts"

# Ensure dependencies are installed (optional quick check)
if ! python -c 'import torch, numpy, matplotlib, tqdm' >/dev/null 2>&1; then
  echo "[INFO] Installing Python dependencies from requirements.txt" >&2
  pip install -r /workspace/requirements.txt
fi

# Run Goal 1 (k=2 vs n) for GD and SGD
"${SCRIPT_DIR}/run_goal1_gd.sh"
"${SCRIPT_DIR}/run_goal1_sgd.sh"

# Run Goal 2 (k sweep) for GD and SGD
"${SCRIPT_DIR}/run_goal2_gd.sh"
"${SCRIPT_DIR}/run_goal2_sgd.sh"

echo "All experiments completed."