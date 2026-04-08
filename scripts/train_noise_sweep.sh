#!/usr/bin/env bash
# Batch training sweep with a specific noise profile.
# Mirrors batch_train_models.sh but makes --noise-profile a first-class option
# and bakes it into the suite name so noise and clean runs stay separate.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROFILE="oc"
NOISE_PROFILE="nominal_train"
NOISE_SCALE="1.0"
NOISE_RAMP="100"
NOISE_WARMUP="20"
NOISE_MIX="0.5"
GROUP="all"
MODELS=()
SEEDS=(42 43 44)
PREFIX="default"
DEVICE=""
DATASET=""
SUITE_NAME=""
EXTRA_TRAIN_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/train_noise_sweep.sh [options]

Options:
  --noise-profile PROFILE   Noise profile for IC regularization.
                            Choices: clean nominal_train nominal_eval degraded_eval
                            Default: nominal_train
  --noise-scale FLOAT       Global noise magnitude multiplier. Default: 1.0
  --noise-ramp N            Curriculum ramp length (epochs). Default: 100
  --noise-warmup N          Clean warmup epochs before noisy IC starts. Default: 20
  --noise-mix FLOAT         Fraction of batches using noisy IC. Default: 0.5
  --profile {oc|noc}        Dataset/training preset. Default: oc
  --group {main|baseline|ablation|core|all}
                            Model subset to run. Default: all
  --models "A B C"          Explicit model list. Overrides --group
  --dataset PATH            Explicit dataset .pkl path. Overrides profile default
  --seeds "42 43 44"        Space-separated seeds. Default: "42 43 44"
  --prefix NAME             Prefix tag in the suite folder name. Default: default
  --suite-name NAME         Explicit suite folder name under checkpoints/
  --device DEVICE           Forwarded to train_auv_hamnode.py
  --extra-train-arg ARG     Extra arg forwarded to training; repeatable
  --help                    Show this message

Examples:
  scripts/train_noise_sweep.sh --noise-profile nominal_train --group core
  scripts/train_noise_sweep.sh --noise-profile degraded_eval --seeds "42 43" --device cuda:0
  scripts/train_noise_sweep.sh --noise-profile clean --group main --prefix baseline
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --noise-profile)
      NOISE_PROFILE="$2"
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
    --noise-warmup)
      NOISE_WARMUP="$2"
      shift 2
      ;;
    --noise-mix)
      NOISE_MIX="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --group)
      GROUP="$2"
      shift 2
      ;;
    --models)
      read -r -a MODELS <<< "$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --seeds)
      read -r -a SEEDS <<< "$2"
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

case "${NOISE_PROFILE}" in
  clean|nominal_train|nominal_eval|degraded_eval) ;;
  *)
    echo "Unsupported noise profile: ${NOISE_PROFILE}." >&2
    echo "Choices: clean nominal_train nominal_eval degraded_eval" >&2
    exit 1
    ;;
esac

case "${PROFILE}" in
  oc)
    DEFAULT_DATASET="${ROOT_DIR}/data/auv_oc_traj1000_blk150_s23_d0be9434.pkl"
    ;;
  noc)
    DEFAULT_DATASET="${ROOT_DIR}/data/auv_noc_traj1000_blk150_s23_d0be9434.pkl"
    ;;
  *)
    echo "Unsupported profile: ${PROFILE}. Expected oc or noc." >&2
    exit 1
    ;;
esac

if [[ -z "${DATASET}" ]]; then
  DATASET="${DEFAULT_DATASET}"
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  echo "Provide --dataset with an existing .pkl file before launching the sweep." >&2
  exit 1
fi

MAIN_MODELS=("phnode_full")
BASELINE_MODELS=(
  "phnode_merged_force"
  "phnode_qforce"
  "se3_momentum_blackbox"
  "se3_accel_blackbox"
  "blackbox_fullstate"
)
ABLATION_MODELS=(
  "ablate_no_mass_prior"
  "ablate_diag_damping"
  "ablate_no_lift"
  "ablate_bu_only"
)
CORE_MODELS=("${MAIN_MODELS[@]}" "${BASELINE_MODELS[@]}")
ALL_MODELS=("${CORE_MODELS[@]}" "${ABLATION_MODELS[@]}")

case "${GROUP}" in
  main)      MODEL_LIST=("${MAIN_MODELS[@]}") ;;
  baseline)  MODEL_LIST=("${BASELINE_MODELS[@]}") ;;
  ablation)  MODEL_LIST=("${ABLATION_MODELS[@]}") ;;
  core)      MODEL_LIST=("${CORE_MODELS[@]}") ;;
  all)       MODEL_LIST=("${ALL_MODELS[@]}") ;;
  *)
    echo "Unsupported group: ${GROUP}" >&2
    exit 1
    ;;
esac

if [[ ${#MODELS[@]} -gt 0 ]]; then
  MODEL_LIST=()
  for model_type in "${MODELS[@]}"; do
    case " ${ALL_MODELS[*]} " in
      *" ${model_type} "*) ;;
      *)
        echo "Unsupported model in --models: ${model_type}" >&2
        echo "Valid models: ${ALL_MODELS[*]}" >&2
        exit 1
        ;;
    esac
    MODEL_LIST+=("${model_type}")
  done
fi

seed_tag="$(printf "%s-" "${SEEDS[@]}")"
seed_tag="${seed_tag%-}"
dataset_stem="$(basename "${DATASET}" .pkl)"
timestamp="$(date +"%Y%m%d_%H%M%S")"
if [[ -z "${SUITE_NAME}" ]]; then
  SUITE_NAME="noise_sweep_${NOISE_PROFILE}_${PROFILE}_${GROUP}_${PREFIX}_${dataset_stem}_s${seed_tag}_${timestamp}"
fi
SUITE_DIR="${CHECKPOINT_ROOT}/${SUITE_NAME}"
mkdir -p "${SUITE_DIR}"

MANIFEST_PATH="${SUITE_DIR}/runs.tsv"
echo -e "group\tmodel_type\tseed\trun_name\trun_dir\tcheckpoint" > "${MANIFEST_PATH}"

cat > "${SUITE_DIR}/suite_config.txt" <<EOF
suite_name=${SUITE_NAME}
profile=${PROFILE}
noise_profile=${NOISE_PROFILE}
noise_scale=${NOISE_SCALE}
noise_ramp=${NOISE_RAMP}
noise_warmup=${NOISE_WARMUP}
noise_mix=${NOISE_MIX}
group=${GROUP}
models=${MODELS[*]:-}
dataset=${DATASET}
seeds=${SEEDS[*]}
python_bin=${PYTHON_BIN}
device=${DEVICE:-auto}
extra_train_args=${EXTRA_TRAIN_ARGS[*]:-}
EOF

echo "Suite directory: ${SUITE_DIR}"
echo "Dataset:         ${DATASET}"
echo "Noise profile:   ${NOISE_PROFILE} (scale=${NOISE_SCALE}, warmup=${NOISE_WARMUP}, ramp=${NOISE_RAMP}, mix=${NOISE_MIX})"
echo "Models (${#MODEL_LIST[@]}): ${MODEL_LIST[*]}"
echo "Seeds:           ${SEEDS[*]}"

for model_type in "${MODEL_LIST[@]}"; do
  model_group="baseline"
  case "${model_type}" in
    phnode_full) model_group="main" ;;
    ablate_*)    model_group="ablation" ;;
  esac

  for seed in "${SEEDS[@]}"; do
    run_name="${model_group}_${model_type}_${NOISE_PROFILE}_seed${seed}"
    run_dir="${SUITE_DIR}/${run_name}"
    checkpoint_path="${run_dir}/best_model.pt"

    if [[ -f "${checkpoint_path}" ]]; then
      echo "[skip] ${run_name} already exists at ${checkpoint_path}"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${model_group}" "${model_type}" "${seed}" "${run_name}" "${run_dir}" "${checkpoint_path}" \
        >> "${MANIFEST_PATH}"
      continue
    fi

    cmd=(
      "${PYTHON_BIN}" "${ROOT_DIR}/train_auv_hamnode.py"
      --dataset "${DATASET}"
      --model_type "${model_type}"
      --save_dir "${SUITE_DIR}"
      --run_name "${run_name}"
      --seed "${seed}"
      --noise_profile "${NOISE_PROFILE}"
      --noise_scale "${NOISE_SCALE}"
      --noise_ramp "${NOISE_RAMP}"
      --noise_warmup_epochs "${NOISE_WARMUP}"
      --noise_mix_ratio "${NOISE_MIX}"
    )
    if [[ -n "${DEVICE}" ]]; then
      cmd+=(--device "${DEVICE}")
    fi
    if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_TRAIN_ARGS[@]}")
    fi

    echo "[train] ${run_name}"
    "${cmd[@]}"

    if [[ ! -f "${checkpoint_path}" ]]; then
      echo "Expected checkpoint missing after training: ${checkpoint_path}" >&2
      exit 1
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${model_group}" "${model_type}" "${seed}" "${run_name}" "${run_dir}" "${checkpoint_path}" \
      >> "${MANIFEST_PATH}"
  done
done

echo "Noise training sweep complete."
echo "Manifest: ${MANIFEST_PATH}"
echo "Next step:"
echo "  scripts/eval_noise_sweep.sh --suite-dir \"${SUITE_DIR}\""
