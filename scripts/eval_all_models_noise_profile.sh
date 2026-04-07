#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUITE_NAME=""
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
NOISE_PROFILES="clean nominal_eval degraded_eval"
NOISE_SEED=2024
EXTRA_EVAL_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval_all_models_noise_profile.sh [options]

Run rollout evaluation, sweep summary, and experiment report for a
profile-based noisy training sweep. If no suite is provided, the script
picks the latest noise-tagged sweep under checkpoints/.

Options:
  --suite-dir PATH             Explicit sweep directory under checkpoints/
  --suite-name NAME            Sweep directory name under checkpoints/
  --mode {heldout|resampled}   Rollout benchmark mode. Default: resampled
  --num-traj-per-scenario N    Default: 30
  --times "10 30 60"           Evaluation horizons
  --scenarios "PRBS CHIRP OU"  Evaluation scenarios
  --eval-seed N                Base seed for evaluation. Default: 42
  --device DEVICE              Forwarded to evaluate_rollout_benchmark.py
  --progress-every N           Default: 5
  --num-diagnostic-plots N     Default: 6
  --summary-horizon N          Default: 60
  --noise-profiles "A B"       Rollout init-noise profiles forwarded to benchmark.
                               Default: "clean nominal_eval degraded_eval"
  --noise-seed N               Base seed for noisy initialization. Default: 2024
  --extra-eval-arg ARG         Extra arg forwarded to evaluation; repeatable
  --help                       Show this message

Examples:
  bash scripts/eval_all_models_noise_profile.sh
  bash scripts/eval_all_models_noise_profile.sh --suite-name sweep_oc_all_noise_nominal_train_...
  bash scripts/eval_all_models_noise_profile.sh --suite-dir ./checkpoints/my_suite --device cuda:0
EOF
}

latest_profile_suite() {
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "sweep_*_noise_*_*" | sort | tail -n 1
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
    --noise-profiles)
      NOISE_PROFILES="$2"
      shift 2
      ;;
    --noise-seed)
      NOISE_SEED="$2"
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

if [[ -n "${SUITE_DIR}" && -n "${SUITE_NAME}" ]]; then
  echo "Use either --suite-dir or --suite-name, not both." >&2
  exit 1
fi

if [[ -z "${SUITE_DIR}" ]]; then
  if [[ -n "${SUITE_NAME}" ]]; then
    SUITE_DIR="${CHECKPOINT_ROOT}/${SUITE_NAME}"
  else
    SUITE_DIR="$(latest_profile_suite)"
  fi
fi

if [[ -z "${SUITE_DIR}" || ! -d "${SUITE_DIR}" ]]; then
  echo "Noise-profile sweep directory not found." >&2
  echo "Use --suite-dir or --suite-name to point to an existing training suite." >&2
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
  --extra-eval-arg "--noise_profiles"
  --extra-eval-arg "${NOISE_PROFILES}"
  --extra-eval-arg "--noise_seed"
  --extra-eval-arg "${NOISE_SEED}"
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
echo "Noise-profile sweep evaluation complete."
echo "Suite directory: ${SUITE_DIR}"
echo "Summary CSV: ${SUITE_DIR}/sweep_model_metrics.csv"
echo "Report: ${SUITE_DIR}/experiment_report.md"
