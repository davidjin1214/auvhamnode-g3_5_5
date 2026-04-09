#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n mytorch1 python)
else
  PYTHON_CMD=(python)
fi

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
NOISE_PROFILES="auto"
NOISE_REFERENCE="auto"
NOISE_SEED=2024
EXTRA_EVAL_ARGS=()
NOISE_PROFILES_ARR=()

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
                               Default: auto
                               auto -> noc: "clean nominal_eval degraded_eval heading_biased_eval"
                                       oc/dr:  "clean nominal_eval degraded_eval heading_biased_eval"
                                       oc/ins: "clean nominal_eval degraded_eval heading_biased_eval current_bias_eval"
  --noise-reference REF        remus100_dr | remus100_ins | auto
                               Default: auto
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

resolve_suite_profile() {
  local suite_dir="$1"
  local config_file="${suite_dir}/suite_config.txt"
  local profile=""

  if [[ -f "${config_file}" ]]; then
    profile="$(awk -F= '$1=="profile"{print $2}' "${config_file}" | tail -n 1)"
  fi
  if [[ -z "${profile}" ]]; then
    case "$(basename "${suite_dir}")" in
      sweep_oc_*) profile="oc" ;;
      sweep_noc_*) profile="noc" ;;
    esac
  fi
  printf "%s" "${profile}"
}

resolve_suite_noise_reference() {
  local suite_dir="$1"
  local config_file=""

  config_file="$(find "${suite_dir}" -mindepth 2 -maxdepth 2 -type f -name "config.json" | sort | head -n 1)"
  if [[ -z "${config_file}" || ! -f "${config_file}" ]]; then
    printf ""
    return
  fi

  "${PYTHON_CMD[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("noise_reference", ""))' "${config_file}"
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
    --noise-reference)
      NOISE_REFERENCE="$2"
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

if [[ "${NOISE_REFERENCE}" == "auto" ]]; then
  NOISE_REFERENCE="$(resolve_suite_noise_reference "${SUITE_DIR}")"
  if [[ -z "${NOISE_REFERENCE}" ]]; then
    NOISE_REFERENCE="remus100_dr"
  fi
fi

case "${NOISE_REFERENCE}" in
  remus100_dr|remus100_ins) ;;
  *)
    echo "Unsupported --noise-reference: ${NOISE_REFERENCE}." >&2
    echo "Expected remus100_dr, remus100_ins, or auto." >&2
    exit 1
    ;;
esac

if [[ "${NOISE_PROFILES}" == "auto" ]]; then
  SUITE_PROFILE="$(resolve_suite_profile "${SUITE_DIR}")"
  if [[ "${SUITE_PROFILE}" == "oc" && "${NOISE_REFERENCE}" == "remus100_ins" ]]; then
    NOISE_PROFILES="clean nominal_eval degraded_eval heading_biased_eval current_bias_eval"
  else
    NOISE_PROFILES="clean nominal_eval degraded_eval heading_biased_eval"
  fi
fi

read -r -a NOISE_PROFILES_ARR <<< "${NOISE_PROFILES}"

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
)
for profile_name in "${NOISE_PROFILES_ARR[@]}"; do
  eval_cmd+=(--extra-eval-arg "${profile_name}")
done
eval_cmd+=(--extra-eval-arg "--noise_reference")
eval_cmd+=(--extra-eval-arg "${NOISE_REFERENCE}")
eval_cmd+=(--extra-eval-arg "--noise_seed")
eval_cmd+=(--extra-eval-arg "${NOISE_SEED}")
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
"${PYTHON_CMD[@]}" "${ROOT_DIR}/scripts/summarize_sweep.py" \
  --suite-dir "${SUITE_DIR}" \
  --horizon "${SUMMARY_HORIZON}"

echo "[stage=report]"
"${PYTHON_CMD[@]}" "${ROOT_DIR}/scripts/build_experiment_report.py" \
  --suite-dir "${SUITE_DIR}" \
  --horizon "${SUMMARY_HORIZON}"

echo
echo "Noise-profile sweep evaluation complete."
echo "Suite directory: ${SUITE_DIR}"
echo "Summary CSV: ${SUITE_DIR}/sweep_model_metrics.csv"
echo "Report: ${SUITE_DIR}/experiment_report.md"
