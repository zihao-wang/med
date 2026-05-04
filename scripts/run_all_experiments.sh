#!/usr/bin/env bash
set -euo pipefail

# Master script to run joint grid (k x n) for GD and SGD.
# Execute from the repository root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "[INFO] Project root: ${PROJECT_ROOT}"

# Joint grid runs
"${SCRIPT_DIR}/run_joint_dependency.sh" gd
"${SCRIPT_DIR}/run_joint_dependency.sh" sgd

echo "All experiments completed."
