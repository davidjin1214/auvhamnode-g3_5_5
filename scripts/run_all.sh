#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PROFILE="oc"
GROUP="all"
MODELS=""
SEEDS="42 43 44"
PREFIX="default"
SUITE_NAME=""
DATASET=""
DEVICE=""
EVAL_MODE="heldout"
NUM_TRAJ_PER_SCENARIO=30
TIMES="10 30 60"
SCENARIOS="PRBS CHIRP OU"
EVAL_SEED=42
PROGRESS_EVERY=5
NUM_DIAGNOSTIC_PLOTS=6
SUMMARY_HORIZON=60
TRAIN_EXTRA_ARGS=()
EVAL_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/run_all.sh [options]

Train a model sweep first, then run batch rollout evaluation on the resulting checkpoints.

Options:
  --profile {oc|noc}              Dataset/training preset. Default: oc
  --group {main|baseline|ablation|core|all}
                                  Model subset to run. Default: all
  --models "A B C"                Explicit model list. Overrides --group in training
  --dataset PATH                  Explicit dataset .pkl path
  --seeds "42 43 44"              Training seeds. Default: "42 43 44"
  --prefix NAME                   Prefix tag embedded in suite folder name. Default: default
  --suite-name NAME               Explicit suite folder name under checkpoints/
  --device DEVICE                 Shared device for train/eval
  --eval-mode {heldout|resampled} Default: heldout
  --num-traj-per-scenario N       Default: 30
  --times "10 30 60"              Evaluation horizons
  --scenarios "PRBS CHIRP OU"     Evaluation scenarios
  --eval-seed N                   Base seed for resampled evaluation. Default: 42
  --progress-every N              Evaluation progress cadence. Default: 5
  --num-diagnostic-plots N        Default: 6
  --summary-horizon N             Horizon used by summarize_sweep.py. Default: 60
  --train-extra-arg ARG           Extra arg forwarded to training; repeatable
  --eval-extra-arg ARG            Extra arg forwarded to evaluation; repeatable
  --help                          Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --suite-name)
      SUITE_NAME="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --eval-mode)
      EVAL_MODE="$2"
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
    --train-extra-arg)
      TRAIN_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --eval-extra-arg)
      EVAL_EXTRA_ARGS+=("$2")
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

train_cmd=(
  "${ROOT_DIR}/scripts/batch_train_models.sh"
  --profile "${PROFILE}"
  --group "${GROUP}"
  --seeds "${SEEDS}"
  --prefix "${PREFIX}"
)
if [[ -n "${MODELS}" ]]; then
  train_cmd+=(--models "${MODELS}")
fi
if [[ -n "${DATASET}" ]]; then
  train_cmd+=(--dataset "${DATASET}")
fi
if [[ -n "${SUITE_NAME}" ]]; then
  train_cmd+=(--suite-name "${SUITE_NAME}")
fi
if [[ -n "${DEVICE}" ]]; then
  train_cmd+=(--device "${DEVICE}")
fi
if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  for arg in "${TRAIN_EXTRA_ARGS[@]}"; do
    train_cmd+=(--extra-train-arg "${arg}")
  done
fi

echo "[stage=train]"
"${train_cmd[@]}"

if [[ -z "${SUITE_NAME}" ]]; then
  latest_suite="$(find "${ROOT_DIR}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name "sweep_${PROFILE}_${GROUP}_${PREFIX}_*" | sort | tail -n 1)"
  if [[ -z "${latest_suite}" ]]; then
    echo "Unable to locate the suite directory after training." >&2
    exit 1
  fi
  SUITE_NAME="$(basename "${latest_suite}")"
fi
SUITE_DIR="${ROOT_DIR}/checkpoints/${SUITE_NAME}"

eval_cmd=(
  "${ROOT_DIR}/scripts/batch_eval_models.sh"
  --suite-dir "${SUITE_DIR}"
  --mode "${EVAL_MODE}"
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
if [[ ${#EVAL_EXTRA_ARGS[@]} -gt 0 ]]; then
  for arg in "${EVAL_EXTRA_ARGS[@]}"; do
    eval_cmd+=(--extra-eval-arg "${arg}")
  done
fi

echo "[stage=eval]"
"${eval_cmd[@]}"

summary_cmd=(
  "${ROOT_DIR}/scripts/summarize_sweep.py"
  --suite-dir "${SUITE_DIR}"
  --horizon "${SUMMARY_HORIZON}"
)

echo "[stage=summarize]"
"${summary_cmd[@]}"

echo "Sweep complete: ${SUITE_DIR}"
