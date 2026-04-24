#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_ROOT="${ROOT_DIR}/checkpoints"
LOCAL_PROXY_ROOT="${LOCAL_PROXY_ROOT:-/content/_proxy_suites}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n mytorch1 python)
else
  PYTHON_CMD=(python)
fi

MODE="${MODE:-}"
RUN_TAG="${RUN_TAG:-phase1a_oc_v4lite_cleanrun_v1}"
DATASET="${DATASET:-${ROOT_DIR}/data/auv_oc_traj1000_blk150_s23_d0be9434.pkl}"
if [[ "${DATASET}" != /* ]]; then
  DATASET="${ROOT_DIR}/${DATASET}"
fi
DEVICE="${DEVICE:-}"
NOISE_REFERENCE="${NOISE_REFERENCE:-remus100_dr}"

PHASE1A_MODELS="${PHASE1A_MODELS:-phnode_full ablate_no_mass_prior ablate_no_lift}"
SMOKE1_MODELS="${SMOKE1_MODELS:-phnode_full}"
SMOKE_SEEDS="${SMOKE_SEEDS:-42 44 46}"
DECISION_SEEDS="${DECISION_SEEDS:-42 43 44 45 46}"

SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO="${SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO:-6}"
DECISION_EVAL_NUM_TRAJ_PER_SCENARIO="${DECISION_EVAL_NUM_TRAJ_PER_SCENARIO:-30}"
EVAL_TIMES="${EVAL_TIMES:-10 30 60}"
EVAL_SCENARIOS="${EVAL_SCENARIOS:-PRBS CHIRP OU}"
EVAL_BASE_SEED="${EVAL_BASE_SEED:-42}"
EVAL_NOISE_SEED="${EVAL_NOISE_SEED:-2024}"
EVAL_PROGRESS_EVERY="${EVAL_PROGRESS_EVERY:-5}"
EVAL_NUM_DIAGNOSTIC_PLOTS="${EVAL_NUM_DIAGNOSTIC_PLOTS:-6}"
IID_EVAL_PROFILES="${IID_EVAL_PROFILES:-clean nominal_eval}"
V4_EVAL_PROFILES="${V4_EVAL_PROFILES:-nominal_eval}"
STRICT_ZERO_NOISE_AUDIT="${STRICT_ZERO_NOISE_AUDIT:-1}"
SOFT_MIN_EPOCH_SCALE="${SOFT_MIN_EPOCH_SCALE:-0.05}"

usage() {
  cat <<'EOF'
Usage:
  MODE=<mode> bash scripts/run_phase1a_oc_v4lite.sh

Modes:
  preflight
  smoke1_train
  smoke1_eval
  smoke3_train
  smoke3_eval
  decision_train
  decision_eval
  decision_summarize

Required workflow order:
  preflight -> smoke1_train -> smoke1_eval -> smoke3_train -> smoke3_eval
  -> decision_train -> decision_eval -> decision_summarize
EOF
}

if [[ -z "${MODE}" || "${MODE}" == "help" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi

phase_suite() {
  local phase="$1"
  local protocol="$2"
  printf "sweep_oc_phase1a_%s_%s_%s" "${phase}" "${protocol}" "${RUN_TAG}"
}

phase_proxy() {
  local phase="$1"
  printf "sweep_oc_phase1a_%s_proxy_%s" "${phase}" "${RUN_TAG}"
}

require_absent() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    echo "Refusing clean Phase-1A execution because target already exists: ${path}" >&2
    echo "Change RUN_TAG or remove the target directory before rerunning this mode." >&2
    exit 2
  fi
}

require_suite() {
  local suite_name="$1"
  local suite_dir="${CHECKPOINT_ROOT}/${suite_name}"
  if [[ ! -f "${suite_dir}/runs.tsv" ]]; then
    echo "Missing suite manifest: ${suite_dir}/runs.tsv" >&2
    exit 2
  fi
}

utils_cmd() {
  "${PYTHON_CMD[@]}" "${ROOT_DIR}/scripts/phase1a_oc_v4lite_utils.py" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --local-proxy-root "${LOCAL_PROXY_ROOT}" \
    "$@"
}

print_config() {
  echo "[phase1a]"
  echo "MODE=${MODE}"
  echo "RUN_TAG=${RUN_TAG}"
  echo "DATASET=${DATASET}"
  echo "DEVICE=${DEVICE:-auto}"
  echo "PHASE1A_MODELS=${PHASE1A_MODELS}"
  echo "SMOKE1_MODELS=${SMOKE1_MODELS}"
  echo "SMOKE_SEEDS=${SMOKE_SEEDS}"
  echo "DECISION_SEEDS=${DECISION_SEEDS}"
  echo "NOISE_REFERENCE=${NOISE_REFERENCE}"
}

train_suite() {
  local suite_name="$1"
  local models="$2"
  local seeds="$3"
  local noise_profile="$4"
  local noise_protocol="$5"

  require_absent "${CHECKPOINT_ROOT}/${suite_name}"

  cmd=(
    bash "${ROOT_DIR}/scripts/train_all_models_noise_profile.sh"
    --profile oc
    --models "${models}"
    --dataset "${DATASET}"
    --seeds "${seeds}"
    --suite-name "${suite_name}"
    --noise-profile "${noise_profile}"
    --noise-protocol "${noise_protocol}"
    --noise-reference "${NOISE_REFERENCE}"
  )
  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi

  echo "[train] suite=${suite_name} models=${models} seeds=${seeds} protocol=${noise_protocol}"
  "${cmd[@]}"
}

train_protocol_triple() {
  local phase="$1"
  local models="$2"
  local seeds="$3"

  train_suite "$(phase_suite "${phase}" clean)" "${models}" "${seeds}" clean auto
  train_suite "$(phase_suite "${phase}" iid)" "${models}" "${seeds}" nominal_train iid_noisy_ic
  train_suite "$(phase_suite "${phase}" v4lite)" "${models}" "${seeds}" nominal_train v4_lite
}

audit_triple() {
  local phase="$1"
  local output_path="${2:-}"
  local cmd=(
    audit
    --suite-name "$(phase_suite "${phase}" clean)"
    --suite-name "$(phase_suite "${phase}" iid)"
    --suite-name "$(phase_suite "${phase}" v4lite)"
    --soft-min-epoch-scale "${SOFT_MIN_EPOCH_SCALE}"
  )
  if [[ "${STRICT_ZERO_NOISE_AUDIT}" == "1" || "${STRICT_ZERO_NOISE_AUDIT}" == "true" ]]; then
    cmd+=(--strict-zero-noise)
  fi
  if [[ -n "${output_path}" ]]; then
    cmd+=(--output "${output_path}")
  fi
  utils_cmd "${cmd[@]}"
}

validate_v4() {
  local phase="$1"
  utils_cmd validate --suite-name "$(phase_suite "${phase}" v4lite)"
}

eval_suite() {
  local suite_name="$1"
  local eval_protocol="$2"
  local run_name="$3"
  local noise_profiles="$4"
  local num_traj="$5"
  local num_plots="$6"

  require_suite "${suite_name}"

  cmd=(
    bash "${ROOT_DIR}/scripts/batch_eval_models.sh"
    --suite-dir "${CHECKPOINT_ROOT}/${suite_name}"
    --mode resampled
    --num-traj-per-scenario "${num_traj}"
    --times "${EVAL_TIMES}"
    --scenarios "${EVAL_SCENARIOS}"
    --seed "${EVAL_BASE_SEED}"
    --progress-every "${EVAL_PROGRESS_EVERY}"
    --num-diagnostic-plots "${num_plots}"
    --extra-eval-arg "--run_name"
    --extra-eval-arg "${run_name}"
    --extra-eval-arg "--noise_protocol"
    --extra-eval-arg "${eval_protocol}"
    --extra-eval-arg "--noise_reference"
    --extra-eval-arg "${NOISE_REFERENCE}"
    --extra-eval-arg "--noise_seed"
    --extra-eval-arg "${EVAL_NOISE_SEED}"
    --extra-eval-arg "--noise_profiles"
  )
  read -r -a profile_array <<< "${noise_profiles}"
  for profile in "${profile_array[@]}"; do
    cmd+=(--extra-eval-arg "${profile}")
  done
  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi

  echo "[eval] suite=${suite_name} eval_protocol=${eval_protocol} profiles=${noise_profiles}"
  "${cmd[@]}"
}

eval_protocol_triple() {
  local phase="$1"
  local num_traj="$2"
  local num_plots="$3"
  local iid_run_name="phase1a_iideval_traj${num_traj}_seed${EVAL_BASE_SEED}"
  local v4_run_name="phase1a_v4eval_traj${num_traj}_seed${EVAL_BASE_SEED}"

  for protocol in clean iid v4lite; do
    eval_suite "$(phase_suite "${phase}" "${protocol}")" iid_noisy_ic "${iid_run_name}" "${IID_EVAL_PROFILES}" "${num_traj}" "${num_plots}"
  done
  for protocol in clean iid v4lite; do
    eval_suite "$(phase_suite "${phase}" "${protocol}")" v4_lite "${v4_run_name}" "${V4_EVAL_PROFILES}" "${num_traj}" "${num_plots}"
  done
}

register_proxy() {
  local phase="$1"
  local export_flag="$2"
  local audit_path="${3:-}"
  local proxy_name
  proxy_name="$(phase_proxy "${phase}")"

  require_absent "${LOCAL_PROXY_ROOT}/${proxy_name}"
  if [[ "${export_flag}" == "1" ]]; then
    require_absent "${CHECKPOINT_ROOT}/${proxy_name}"
  fi

  cmd=(
    register-proxy
    --proxy-suite-name "${proxy_name}"
    --suite-name "$(phase_suite "${phase}" clean)"
    --suite-name "$(phase_suite "${phase}" iid)"
    --suite-name "$(phase_suite "${phase}" v4lite)"
  )
  if [[ -n "${audit_path}" ]]; then
    cmd+=(--audit-path "${audit_path}")
  fi
  if [[ "${export_flag}" == "1" ]]; then
    cmd+=(--export)
  fi
  utils_cmd "${cmd[@]}"
}

print_config

case "${MODE}" in
  preflight)
    utils_cmd preflight --run-tag "${RUN_TAG}"
    ;;
  smoke1_train)
    train_protocol_triple smoke1 "${SMOKE1_MODELS}" "${SMOKE_SEEDS}"
    audit_triple smoke1
    validate_v4 smoke1
    ;;
  smoke1_eval)
    eval_protocol_triple smoke1 "${SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO}" 2
    register_proxy smoke1 0
    ;;
  smoke3_train)
    train_protocol_triple smoke3 "${PHASE1A_MODELS}" "${SMOKE_SEEDS}"
    audit_triple smoke3
    validate_v4 smoke3
    ;;
  smoke3_eval)
    eval_protocol_triple smoke3 "${SMOKE_EVAL_NUM_TRAJ_PER_SCENARIO}" 2
    register_proxy smoke3 0
    ;;
  decision_train)
    train_protocol_triple decision "${PHASE1A_MODELS}" "${DECISION_SEEDS}"
    audit_triple decision
    validate_v4 decision
    ;;
  decision_eval)
    eval_protocol_triple decision "${DECISION_EVAL_NUM_TRAJ_PER_SCENARIO}" "${EVAL_NUM_DIAGNOSTIC_PLOTS}"
    ;;
  decision_summarize)
    mkdir -p "${LOCAL_PROXY_ROOT}"
    audit_path="${LOCAL_PROXY_ROOT}/phase1a_decision_train_audit_${RUN_TAG}.csv"
    audit_triple decision "${audit_path}"
    register_proxy decision 1 "${audit_path}"
    echo "Decision artifacts: ${CHECKPOINT_ROOT}/$(phase_proxy decision)"
    ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    usage >&2
    exit 2
    ;;
esac
