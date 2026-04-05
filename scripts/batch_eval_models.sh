#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUITE_DIR=""
MODE="heldout"
NUM_TRAJ_PER_SCENARIO=30
TIMES=(10 30 60)
SCENARIOS=(PRBS CHIRP OU)
BASE_SEED=42
DEVICE=""
PROGRESS_EVERY=5
NUM_DIAGNOSTIC_PLOTS=6
EXTRA_EVAL_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/batch_eval_models.sh --suite-dir CHECKPOINT_SUITE [options]

Options:
  --suite-dir PATH             Sweep directory produced by batch_train_models.sh
  --mode {heldout|resampled}   Rollout benchmark mode. Default: heldout
  --num-traj-per-scenario N    Default: 30
  --times "10 30 60"           Space-separated horizons in seconds
  --scenarios "PRBS CHIRP OU"  Space-separated scenario names
  --seed N                     Base random seed for resampled evaluation. Default: 42
  --device DEVICE              Forwarded to evaluate_rollout_benchmark.py
  --progress-every N           Default: 5
  --num-diagnostic-plots N     Default: 6
  --extra-eval-arg ARG         Extra arg forwarded to evaluation; repeatable
  --help                       Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir)
      SUITE_DIR="$2"
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
      read -r -a TIMES <<< "$2"
      shift 2
      ;;
    --scenarios)
      read -r -a SCENARIOS <<< "$2"
      shift 2
      ;;
    --seed)
      BASE_SEED="$2"
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
  echo "--suite-dir is required." >&2
  usage >&2
  exit 1
fi

SUITE_DIR="$(cd "${SUITE_DIR}" && pwd)"
MANIFEST_PATH="${SUITE_DIR}/runs.tsv"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

echo "Suite directory: ${SUITE_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Mode: ${MODE}"

tail -n +2 "${MANIFEST_PATH}" | while IFS=$'\t' read -r model_group model_type seed run_name run_dir checkpoint_path; do
  if [[ -z "${checkpoint_path}" ]]; then
    continue
  fi

  local_run_dir="${run_dir}"
  if [[ ! -d "${local_run_dir}" ]]; then
    local_run_dir="${SUITE_DIR}/$(basename "${run_dir}")"
  fi

  local_checkpoint_path="${checkpoint_path}"
  if [[ ! -f "${local_checkpoint_path}" ]]; then
    local_checkpoint_path="${local_run_dir}/$(basename "${checkpoint_path}")"
  fi

  if [[ ! -f "${local_checkpoint_path}" ]]; then
    echo "[skip] Missing checkpoint: ${checkpoint_path}" >&2
    continue
  fi

  eval_root="${local_run_dir}/rollout_benchmark"
  mkdir -p "${eval_root}"
  eval_name="${MODE}_traj${NUM_TRAJ_PER_SCENARIO}_seed${BASE_SEED}"
  summary_pattern="${eval_root}/${eval_name}_*/summary.txt"
  if compgen -G "${summary_pattern}" > /dev/null; then
    echo "[skip] ${run_name} already evaluated under ${eval_root}"
    continue
  fi

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/evaluate_rollout_benchmark.py"
    --checkpoint "${local_checkpoint_path}"
    --mode "${MODE}"
    --output_dir "${eval_root}"
    --run_name "${eval_name}"
    --num_traj_per_scenario "${NUM_TRAJ_PER_SCENARIO}"
    --seed "${BASE_SEED}"
    --progress_every "${PROGRESS_EVERY}"
    --num_diagnostic_plots "${NUM_DIAGNOSTIC_PLOTS}"
    --times "${TIMES[@]}"
    --scenarios "${SCENARIOS[@]}"
  )
  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ ${#EXTRA_EVAL_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_EVAL_ARGS[@]}")
  fi

  echo "[eval] ${run_name}"
  "${cmd[@]}"
done

echo "Batch evaluation complete."
