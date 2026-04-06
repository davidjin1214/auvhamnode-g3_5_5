#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints"

PROFILE="oc"
GROUP="all"
SEEDS="43 44 45"
DEVICE=""
DATASET=""
SUITE_NAME=""
NOISE_LEVEL=2
NOISE_SCALE="1.0"
NOISE_RAMP=100
PREFIX=""
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_all_models_noise.sh [options]

Train all models with training-time observation noise enabled. By default,
the script uses the traj1000 dataset under data/ and forwards the noise
configuration to scripts/batch_train_models.sh.

Options:
  --profile {oc|noc}        Dataset preset used for auto-discovery. Default: oc
  --group {main|baseline|ablation|core|all}
                            Model subset to train. Default: all
  --dataset PATH            Explicit dataset .pkl path. Overrides auto-discovery
  --seeds "43 44 45"        Space-separated training seeds. Default: "43 44 45"
  --device DEVICE           Forwarded to train_auv_hamnode.py
  --noise-level {1|2|3}     Training noise level. Default: 2
  --noise-scale FLOAT       Global noise scale. Default: 1.0
  --noise-ramp N            Noise curriculum ramp epochs. Default: 100
  --prefix NAME             Prefix embedded in the suite folder name
  --suite-name NAME         Explicit suite folder name under checkpoints/
  --extra-train-arg ARG     Extra arg forwarded to training; repeatable
  --help                    Show this message

Examples:
  bash scripts/train_all_models_noise.sh
  bash scripts/train_all_models_noise.sh --group main
  bash scripts/train_all_models_noise.sh --group core --noise-level 3
  bash scripts/train_all_models_noise.sh --noise-level 3 --device cuda:0
  bash scripts/train_all_models_noise.sh --dataset ./data/my.pkl --seeds "43 44"
EOF
}

resolve_default_dataset() {
  local profile="$1"
  local pattern=""
  case "${profile}" in
    oc)
      pattern="auv_oc_traj1000*.pkl"
      ;;
    noc)
      pattern="auv_noc_traj1000*.pkl"
      ;;
    *)
      echo "Unsupported profile: ${profile}. Expected oc or noc." >&2
      exit 1
      ;;
  esac

  find "${ROOT_DIR}/data" -maxdepth 1 -type f -name "${pattern}" | sort | tail -n 1
}

latest_noise_suite() {
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "sweep_*_noise_l*_*" | sort | tail -n 1
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
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --noise-level)
      NOISE_LEVEL="$2"
      shift 2
      ;;
    --noise-scale)
      NOISE_SCALE="$2"
      shift 2
      ;;
    --noise-ramp)
      NOISE_RAMP="$2"
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

case "${GROUP}" in
  main|baseline|ablation|core|all) ;;
  *)
    echo "Unsupported --group: ${GROUP}. Expected main, baseline, ablation, core, or all." >&2
    exit 1
    ;;
esac

case "${NOISE_LEVEL}" in
  1|2|3) ;;
  *)
    echo "Unsupported --noise-level: ${NOISE_LEVEL}. Expected 1, 2, or 3." >&2
    exit 1
    ;;
esac

if [[ -z "${DATASET}" ]]; then
  DATASET="$(resolve_default_dataset "${PROFILE}")"
fi

if [[ -z "${DATASET}" || ! -f "${DATASET}" ]]; then
  echo "Unable to resolve a traj1000 dataset under ${ROOT_DIR}/data for profile=${PROFILE}." >&2
  echo "Use --dataset to provide an explicit .pkl path." >&2
  exit 1
fi

if [[ -z "${PREFIX}" ]]; then
  PREFIX="noise_l${NOISE_LEVEL}"
fi

cmd=(
  bash "${ROOT_DIR}/scripts/batch_train_models.sh"
  --profile "${PROFILE}"
  --group "${GROUP}"
  --dataset "${DATASET}"
  --seeds "${SEEDS}"
  --prefix "${PREFIX}"
  --extra-train-arg "--noise_level"
  --extra-train-arg "${NOISE_LEVEL}"
  --extra-train-arg "--noise_scale"
  --extra-train-arg "${NOISE_SCALE}"
  --extra-train-arg "--noise_ramp"
  --extra-train-arg "${NOISE_RAMP}"
)
if [[ -n "${SUITE_NAME}" ]]; then
  cmd+=(--suite-name "${SUITE_NAME}")
fi
if [[ -n "${DEVICE}" ]]; then
  cmd+=(--device "${DEVICE}")
fi
if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
  for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
    cmd+=(--extra-train-arg "${arg}")
  done
fi

echo "Training all models with noise."
echo "Dataset: ${DATASET}"
echo "Group: ${GROUP}"
echo "Seeds: ${SEEDS}"
echo "Noise config: level=${NOISE_LEVEL}, scale=${NOISE_SCALE}, ramp=${NOISE_RAMP}"
"${cmd[@]}"

if [[ -n "${SUITE_NAME}" ]]; then
  resolved_suite="${CHECKPOINT_ROOT}/${SUITE_NAME}"
else
  resolved_suite="$(latest_noise_suite)"
fi

if [[ -n "${resolved_suite:-}" && -d "${resolved_suite}" ]]; then
  echo
  echo "Training complete."
  echo "Suite directory: ${resolved_suite}"
  echo "Next step:"
  echo "  bash scripts/eval_all_models_noise.sh --suite-dir \"${resolved_suite}\""
else
  echo
  echo "Training completed, but the suite directory could not be resolved automatically." >&2
fi
