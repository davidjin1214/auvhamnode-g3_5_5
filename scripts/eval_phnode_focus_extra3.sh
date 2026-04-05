#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUITE_NAME="sweep_oc_phnode_focus_extra3_auv_oc_traj1000_blk150_s23_d0be9434_s45-46-47"
SUITE_DIR=""
MODE="resampled"
NUM_TRAJ_PER_SCENARIO=30
TIMES="10 30 60"
SCENARIOS="PRBS CHIRP OU"
EVAL_SEED=42
DEVICE=""
PROGRESS_EVERY=5
NUM_DIAGNOSTIC_PLOTS=6
SUMMARY_HORIZON=60
EXTRA_EVAL_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_phnode_focus_extra3.sh [options]

This script runs rollout evaluation and sweep-level summaries for the
PHNODE-focused supplementary seed sweep after the trained checkpoints have
been copied back from the remote machine.

Options:
  --suite-dir PATH             Explicit local sweep directory under checkpoints/
  --suite-name NAME            Sweep directory name under checkpoints/
  --mode {heldout|resampled}   Rollout benchmark mode. Default: resampled
  --num-traj-per-scenario N    Default: 30
  --times "10 30 60"           Evaluation horizons
  --scenarios "PRBS CHIRP OU"  Evaluation scenarios
  --eval-seed N                Base seed for resampled evaluation. Default: 42
  --device DEVICE              Forwarded to evaluate_rollout_benchmark.py
  --progress-every N           Default: 5
  --num-diagnostic-plots N     Default: 6
  --summary-horizon N          Default: 60
  --extra-eval-arg ARG         Extra arg forwarded to evaluation; repeatable
  --help                       Show this message

Notes:
  1. Run this only after the remote training suite has been copied into
     checkpoints/<suite-name> on the local machine.
  2. This script does not train anything.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir)
      SUITE_DIR="$2"
      shift 2
      ;;
    --suite-name)
      SUITE_NAME="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --num-traj-per-scenario)
      NUM_TRAJ_PER_SCENARIO="$2"
      shift 2
      ;;
    --times)
      TIMES="$2"
      shift 2
      ;;
    --scenarios)
      SCENARIOS="$2"
      shift 2
      ;;
    --eval-seed)
      EVAL_SEED="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --num-diagnostic-plots)
      NUM_DIAGNOSTIC_PLOTS="$2"
      shift 2
      ;;
    --summary-horizon)
      SUMMARY_HORIZON="$2"
      shift 2
      ;;
    --extra-eval-arg)
      EXTRA_EVAL_ARGS+=("$2")
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

if [[ -z "${SUITE_DIR}" ]]; then
  SUITE_DIR="${ROOT_DIR}/checkpoints/${SUITE_NAME}"
fi

if [[ ! -d "${SUITE_DIR}" ]]; then
  echo "Suite directory not found: ${SUITE_DIR}" >&2
  echo "Copy the remote training outputs into checkpoints/ before running evaluation." >&2
  exit 1
fi

eval_cmd=(
  bash "${ROOT_DIR}/scripts/batch_eval_models.sh"
  --suite-dir "${SUITE_DIR}"
  --mode "${MODE}"
  --num-traj-per-scenario "${NUM_TRAJ_PER_SCENARIO}"
  --times "${TIMES}"
  --scenarios "${SCENARIOS}"
  --seed "${EVAL_SEED}"
  --progress-every "${PROGRESS_EVERY}"
  --num-diagnostic-plots "${NUM_DIAGNOSTIC_PLOTS}"
)
if [[ -n "${DEVICE}" ]]; then
  eval_cmd+=(--device "${DEVICE}")
fi
if [[ ${#EXTRA_EVAL_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_EVAL_ARGS[@]}"; do
    eval_cmd+=(--extra-eval-arg "${arg}")
  done
fi

echo "[stage=eval]"
"${eval_cmd[@]}"

echo "[stage=summarize]"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/summarize_sweep.py" \
  --suite-dir "${SUITE_DIR}" \
  --horizon "${SUMMARY_HORIZON}"

echo "[stage=report]"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_experiment_report.py" \
  --suite-dir "${SUITE_DIR}" \
  --horizon "${SUMMARY_HORIZON}"

echo
echo "Local evaluation complete."
echo "Summary CSV: ${SUITE_DIR}/sweep_model_metrics.csv"
echo "Report: ${SUITE_DIR}/experiment_report.md"
