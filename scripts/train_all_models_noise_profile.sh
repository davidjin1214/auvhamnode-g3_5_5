#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints"

PROFILE="oc"
GROUP="all"
MODELS=""
SEEDS="43 44 45"
DEVICE=""
DATASET=""
SUITE_NAME=""
NOISE_PROFILE="nominal_train"
NOISE_SCALE="1.0"
NOISE_WARMUP_EPOCHS="20"
NOISE_RAMP="80"
NOISE_MIX_RATIO="0.5"
BLOCK_EVAL_NOISE_PROFILES="auto"
HELDOUT_EVAL_NOISE_PROFILES="auto"
PREFIX=""
EXTRA_TRAIN_ARGS=()
BLOCK_EVAL_NOISE_PROFILES_ARR=()
HELDOUT_EVAL_NOISE_PROFILES_ARR=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_all_models_noise_profile.sh [options]

Train a model sweep with the current profile-based noisy-IC interface.
This is the recommended wrapper for noisy training experiments.

Options:
  --profile {oc|noc}                Dataset preset used for auto-discovery. Default: oc
  --group {main|baseline|ablation|core|all}
                                    Model subset to train. Default: all
  --models "A B C"                  Explicit model list. Overrides --group
  --dataset PATH                    Explicit dataset .pkl path. Overrides auto-discovery
  --seeds "43 44 45"                Space-separated training seeds. Default: "43 44 45"
  --device DEVICE                   Forwarded to train_auv_hamnode.py
  --noise-profile PROFILE           One of clean nominal_train nominal_eval degraded_eval
                                    Default: nominal_train
  --noise-scale FLOAT               Global noise multiplier. Default: 1.0
  --noise-warmup-epochs N           Fully clean warmup epochs. Default: 20
  --noise-ramp N                    Ramp length after warmup. Default: 80
  --noise-mix-ratio FLOAT           Fraction of training samples using noisy IC. Default: 0.5
  --block-eval-noise-profiles "A B" Post-training block eval profiles.
                                    Default: auto
                                    auto -> noc: "clean nominal_eval"
                                            oc:  "clean nominal_eval heading_biased_eval current_bias_eval"
  --heldout-eval-noise-profiles "A B"
                                    Post-training heldout eval profiles.
                                    Default: auto
                                    auto -> noc: "clean nominal_eval degraded_eval heading_biased_eval"
                                            oc:  "clean nominal_eval degraded_eval heading_biased_eval current_bias_eval"
  --prefix NAME                     Prefix embedded in the suite folder name
  --suite-name NAME                 Explicit suite folder name under checkpoints/
  --extra-train-arg ARG             Extra arg forwarded to training; repeatable
  --help                            Show this message

Examples:
  bash scripts/train_all_models_noise_profile.sh
  bash scripts/train_all_models_noise_profile.sh --group main
  bash scripts/train_all_models_noise_profile.sh --noise-profile nominal_eval
  bash scripts/train_all_models_noise_profile.sh --noise-scale 0.7 --noise-mix-ratio 0.3
  bash scripts/train_all_models_noise_profile.sh --models "phnode_full se3_accel_blackbox"
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

latest_profile_suite() {
  local noise_tag="$1"
  find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "sweep_*_${noise_tag}_*" | sort | tail -n 1
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
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --noise-profile)
      NOISE_PROFILE="$2"
      shift 2
      ;;
    --noise-scale)
      NOISE_SCALE="$2"
      shift 2
      ;;
    --noise-warmup-epochs)
      NOISE_WARMUP_EPOCHS="$2"
      shift 2
      ;;
    --noise-ramp)
      NOISE_RAMP="$2"
      shift 2
      ;;
    --noise-mix-ratio)
      NOISE_MIX_RATIO="$2"
      shift 2
      ;;
    --block-eval-noise-profiles)
      BLOCK_EVAL_NOISE_PROFILES="$2"
      shift 2
      ;;
    --heldout-eval-noise-profiles)
      HELDOUT_EVAL_NOISE_PROFILES="$2"
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

case "${NOISE_PROFILE}" in
  clean|nominal_train|nominal_eval|degraded_eval) ;;
  *)
    echo "Unsupported --noise-profile: ${NOISE_PROFILE}." >&2
    echo "Expected clean, nominal_train, nominal_eval, or degraded_eval." >&2
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
  PREFIX="noise_${NOISE_PROFILE}"
fi

if [[ "${BLOCK_EVAL_NOISE_PROFILES}" == "auto" ]]; then
  if [[ "${PROFILE}" == "oc" ]]; then
    BLOCK_EVAL_NOISE_PROFILES="clean nominal_eval heading_biased_eval current_bias_eval"
  else
    BLOCK_EVAL_NOISE_PROFILES="clean nominal_eval"
  fi
fi

if [[ "${HELDOUT_EVAL_NOISE_PROFILES}" == "auto" ]]; then
  if [[ "${PROFILE}" == "oc" ]]; then
    HELDOUT_EVAL_NOISE_PROFILES="clean nominal_eval degraded_eval heading_biased_eval current_bias_eval"
  else
    HELDOUT_EVAL_NOISE_PROFILES="clean nominal_eval degraded_eval heading_biased_eval"
  fi
fi

read -r -a BLOCK_EVAL_NOISE_PROFILES_ARR <<< "${BLOCK_EVAL_NOISE_PROFILES}"
read -r -a HELDOUT_EVAL_NOISE_PROFILES_ARR <<< "${HELDOUT_EVAL_NOISE_PROFILES}"

cmd=(
  bash "${ROOT_DIR}/scripts/batch_train_models.sh"
  --profile "${PROFILE}"
  --group "${GROUP}"
  --dataset "${DATASET}"
  --seeds "${SEEDS}"
  --prefix "${PREFIX}"
  --extra-train-arg "--noise_profile"
  --extra-train-arg "${NOISE_PROFILE}"
  --extra-train-arg "--noise_scale"
  --extra-train-arg "${NOISE_SCALE}"
  --extra-train-arg "--noise_warmup_epochs"
  --extra-train-arg "${NOISE_WARMUP_EPOCHS}"
  --extra-train-arg "--noise_ramp"
  --extra-train-arg "${NOISE_RAMP}"
  --extra-train-arg "--noise_mix_ratio"
  --extra-train-arg "${NOISE_MIX_RATIO}"
  --extra-train-arg "--block_eval_noise_profiles"
)
for profile_name in "${BLOCK_EVAL_NOISE_PROFILES_ARR[@]}"; do
  cmd+=(--extra-train-arg "${profile_name}")
done
cmd+=(--extra-train-arg "--heldout_eval_noise_profiles")
for profile_name in "${HELDOUT_EVAL_NOISE_PROFILES_ARR[@]}"; do
  cmd+=(--extra-train-arg "${profile_name}")
done
if [[ -n "${MODELS}" ]]; then
  cmd+=(--models "${MODELS}")
fi
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

echo "Training all models with profile-based noisy IC."
echo "Dataset: ${DATASET}"
echo "Group: ${GROUP}"
if [[ -n "${MODELS}" ]]; then
  echo "Explicit models: ${MODELS}"
fi
echo "Seeds: ${SEEDS}"
echo "Noise config: profile=${NOISE_PROFILE}, scale=${NOISE_SCALE}, warmup=${NOISE_WARMUP_EPOCHS}, ramp=${NOISE_RAMP}, mix_ratio=${NOISE_MIX_RATIO}"
echo "Auto-eval profiles: block=[${BLOCK_EVAL_NOISE_PROFILES}] heldout=[${HELDOUT_EVAL_NOISE_PROFILES}]"
"${cmd[@]}"

if [[ -n "${SUITE_NAME}" ]]; then
  resolved_suite="${CHECKPOINT_ROOT}/${SUITE_NAME}"
else
  resolved_suite="$(latest_profile_suite "${PREFIX}")"
fi

if [[ -n "${resolved_suite:-}" && -d "${resolved_suite}" ]]; then
  echo
  echo "Training complete."
  echo "Suite directory: ${resolved_suite}"
  echo "Next step:"
  echo "  bash scripts/eval_all_models_noise_profile.sh --suite-dir \"${resolved_suite}\""
else
  echo
  echo "Training completed, but the suite directory could not be resolved automatically." >&2
fi
