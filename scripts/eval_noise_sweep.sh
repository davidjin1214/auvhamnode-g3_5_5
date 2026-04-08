#!/usr/bin/env bash
# Batch rollout evaluation with explicit noise profiles.
# Mirrors batch_eval_models.sh but makes --noise-profiles a first-class option
# so clean vs. noisy IC conditions are easy to compare in one sweep.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUITE_DIR=""
NOISE_PROFILES=("clean" "nominal_eval" "degraded_eval")
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
  scripts/eval_noise_sweep.sh --suite-dir CHECKPOINT_SUITE [options]

Options:
  --suite-dir PATH               Sweep directory produced by train_noise_sweep.sh
                                 or batch_train_models.sh
  --noise-profiles "P1 P2 ..."   Space-separated noise profiles for evaluation.
                                 Choices: clean nominal_eval degraded_eval all
                                 Default: "clean nominal_eval degraded_eval"
  --mode {heldout|resampled}     Rollout benchmark mode. Default: heldout
  --num-traj-per-scenario N      Default: 30
  --times "10 30 60"             Space-separated horizons in seconds
  --scenarios "PRBS CHIRP OU"    Space-separated scenario names
  --seed N                       Base random seed. Default: 42
  --device DEVICE                Forwarded to evaluate_rollout_benchmark.py
  --progress-every N             Default: 5
  --num-diagnostic-plots N       Default: 6
  --extra-eval-arg ARG           Extra arg forwarded to evaluation; repeatable
  --help                         Show this message

Examples:
  scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/my_sweep
  scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/my_sweep --noise-profiles "clean nominal_eval"
  scripts/eval_noise_sweep.sh --suite-dir ./checkpoints/my_sweep --noise-profiles all --mode resampled
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-dir)
      SUITE_DIR="$2"
      shift 2
      ;;
    --noise-profiles)
      read -r -a NOISE_PROFILES <<< "$2"
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

# Expand "all" shorthand.
EXPANDED_PROFILES=()
for p in "${NOISE_PROFILES[@]}"; do
  case "${p}" in
    all)
      EXPANDED_PROFILES+=("clean" "nominal_eval" "degraded_eval")
      ;;
    clean|nominal_eval|degraded_eval)
      EXPANDED_PROFILES+=("${p}")
      ;;
    *)
      echo "Unsupported noise profile: ${p}." >&2
      echo "Choices: clean nominal_eval degraded_eval all" >&2
      exit 1
      ;;
  esac
done
NOISE_PROFILES=("${EXPANDED_PROFILES[@]}")

SUITE_DIR="$(cd "${SUITE_DIR}" && pwd)"
MANIFEST_PATH="${SUITE_DIR}/runs.tsv"
if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

echo "Suite directory:  ${SUITE_DIR}"
echo "Noise profiles:   ${NOISE_PROFILES[*]}"
echo "Mode:             ${MODE}"

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
  profiles_tag="$(printf "%s-" "${NOISE_PROFILES[@]}")"
  profiles_tag="${profiles_tag%-}"
  summary_pattern="${eval_root}/${eval_name}_noise${profiles_tag}_*/summary.txt"
  if compgen -G "${summary_pattern}" > /dev/null; then
    echo "[skip] ${run_name} already evaluated under ${eval_root}"
    continue
  fi

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/evaluate_rollout_benchmark.py"
    --checkpoint "${local_checkpoint_path}"
    --mode "${MODE}"
    --output_dir "${eval_root}"
    --run_name "${eval_name}_noise${profiles_tag}"
    --num_traj_per_scenario "${NUM_TRAJ_PER_SCENARIO}"
    --seed "${BASE_SEED}"
    --progress_every "${PROGRESS_EVERY}"
    --num_diagnostic_plots "${NUM_DIAGNOSTIC_PLOTS}"
    --times "${TIMES[@]}"
    --scenarios "${SCENARIOS[@]}"
    --noise_profiles "${NOISE_PROFILES[@]}"
  )
  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ ${#EXTRA_EVAL_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_EVAL_ARGS[@]}")
  fi

  echo "[eval] ${run_name} | noise: ${NOISE_PROFILES[*]}"
  "${cmd[@]}"
done

echo "Noise evaluation sweep complete."
