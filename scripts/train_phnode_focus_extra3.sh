#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE="oc"
SEEDS="45 46 47"
MODELS="phnode_full ablate_no_mass_prior ablate_diag_damping ablate_no_lift ablate_bu_only"
SUITE_NAME="sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47"
DATASET=""
DEVICE=""
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_phnode_focus_extra3.sh [options]

This script trains only the 5 PHNODE-focused models for the additional
seeds 45/46/47. It does not run rollout evaluation.

Options:
  --suite-name NAME         Output suite directory name under checkpoints/
  --seeds "45 46 47"        Space-separated training seeds
  --dataset PATH            Explicit dataset .pkl path
  --device DEVICE           Forwarded to train_auv_hamnode.py
  --extra-train-arg ARG     Extra arg forwarded to training; repeatable
  --help                    Show this message

Notes:
  1. Run this on the remote training machine.
  2. After training finishes, copy the entire checkpoints/<suite-name> directory
     back to this repo on the local machine.
  3. Then run bash scripts/eval_phnode_focus_extra3.sh locally.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-name)
      SUITE_NAME="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --extra-train-arg)
      EXTRA_TRAIN_ARGS+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cmd=(
  bash "${ROOT_DIR}/scripts/batch_train_models.sh"
  --profile "${PROFILE}"
  --group all
  --models "${MODELS}"
  --seeds "${SEEDS}"
  --suite-name "${SUITE_NAME}"
)
if [[ -n "${DATASET}" ]]; then
  cmd+=(--dataset "${DATASET}")
fi
if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
    cmd+=(--extra-train-arg "${arg}")
  done
fi

echo "Training suite: ${SUITE_NAME}"
echo "Models: ${MODELS}"
echo "Seeds: ${SEEDS}"
"${cmd[@]}"

echo
echo "Remote training complete."
echo "Copy this directory back to the local machine before evaluation:"
echo "  ${ROOT_DIR}/checkpoints/${SUITE_NAME}"
