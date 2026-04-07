from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import pickle

import numpy as np

from data_collection import (
    ScenarioType,
    create_scenario_generator,
    deserialize_config,
    serialize_config,
    replay_trajectory_setup,
    sample_initial_state as sample_initial_state_from_config,
    _generate_ocean_current,
)
from remus100_core import (
    Remus100Dynamics,
    Remus100Simulator,
    rotation_matrix_from_quaternion,
)


# ---------------------------------------------------------------------------
# Lazy-loaded runtime dependencies (PyTorch, torchdiffeq, model classes)
# ---------------------------------------------------------------------------

torch = None
odeint = None
plt = None
AUVHamNODE = None
TrainConfig = None
train_model_builder = None
canonicalize_model_type = None
compute_rotation_matrix = None
geodesic_distance_so3 = None
NoiseConfig = None
StateNormalizer = None
build_noisy_initial_condition = None


def ensure_runtime_imports():
    global torch, odeint, AUVHamNODE, TrainConfig, train_model_builder
    global canonicalize_model_type
    global compute_rotation_matrix, geodesic_distance_so3
    global NoiseConfig, StateNormalizer, build_noisy_initial_condition

    if torch is not None:
        return

    import torch as torch_mod
    from torchdiffeq import odeint as odeint_fn

    from AUVHamNODE import AUVHamNODE as AUVHamNODE_cls
    from train_auv_hamnode import (
        _build_model as train_model_builder_fn,
        canonicalize_model_type as canonicalize_model_type_fn,
    )
    from train_utils import (
        TrainConfig as TrainConfig_cls,
        NoiseConfig as NoiseConfig_cls,
        StateNormalizer as StateNormalizer_cls,
        build_noisy_initial_condition as build_noisy_initial_condition_fn,
        compute_rotation_matrix as compute_rotation_matrix_fn,
        geodesic_distance_so3 as geodesic_distance_so3_fn,
    )

    torch = torch_mod
    odeint = odeint_fn
    AUVHamNODE = AUVHamNODE_cls
    TrainConfig = TrainConfig_cls
    train_model_builder = train_model_builder_fn
    canonicalize_model_type = canonicalize_model_type_fn
    compute_rotation_matrix = compute_rotation_matrix_fn
    geodesic_distance_so3 = geodesic_distance_so3_fn
    NoiseConfig = NoiseConfig_cls
    StateNormalizer = StateNormalizer_cls
    build_noisy_initial_condition = build_noisy_initial_condition_fn


def get_torch():
    ensure_runtime_imports()
    return torch


def get_odeint():
    ensure_runtime_imports()
    return odeint


def get_model_classes():
    ensure_runtime_imports()
    return AUVHamNODE, TrainConfig


def get_rotation_helpers():
    ensure_runtime_imports()
    return compute_rotation_matrix, geodesic_distance_so3


def get_plot_module():
    global plt
    if plt is None:
        try:
            import matplotlib.pyplot as plt_mod
        except ModuleNotFoundError:  # pragma: no cover
            plt = False
        else:
            plt = plt_mod
    return None if plt is False else plt


# ---------------------------------------------------------------------------
# Array factory helpers
# ---------------------------------------------------------------------------

def _empty_vector():
    return np.empty((0,), dtype=np.float64)


def _empty_bool_vector():
    return np.empty((0,), dtype=bool)


def _empty_pos():
    return np.empty((0, 3), dtype=np.float64)


def _empty_rot():
    return np.empty((0, 3, 3), dtype=np.float64)


def _empty_nu():
    return np.empty((0, 6), dtype=np.float64)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RolloutResult:
    scenario: str
    seed: int
    eta0: np.ndarray
    nu0: np.ndarray
    time: np.ndarray = field(default_factory=_empty_vector)
    gt_pos: np.ndarray = field(default_factory=_empty_pos)
    pred_pos: np.ndarray = field(default_factory=_empty_pos)
    gt_rotation: np.ndarray = field(default_factory=_empty_rot)
    pred_rotation_raw: np.ndarray = field(default_factory=_empty_rot)
    gt_nu: np.ndarray = field(default_factory=_empty_nu)
    pred_nu: np.ndarray = field(default_factory=_empty_nu)
    gt_nu_total: np.ndarray = field(default_factory=_empty_nu)
    pred_nu_total: np.ndarray = field(default_factory=_empty_nu)
    pred_energy: np.ndarray = field(default_factory=_empty_vector)
    failure_reason: str = "completed"
    completed_time: float = 0.0

    @property
    def trajectory_id(self) -> str:
        return f"{self.scenario}_{self.seed}"


@dataclass(frozen=True)
class RolloutAnalysis:
    position_error: np.ndarray
    depth_error: np.ndarray
    rotation_geodesic: np.ndarray
    relative_linear_velocity_error: np.ndarray
    relative_angular_velocity_error: np.ndarray
    total_linear_velocity_error: np.ndarray
    total_angular_velocity_error: np.ndarray
    so3_det_error: np.ndarray
    so3_orth_error: np.ndarray
    so3_violation: np.ndarray
    depth_violation: np.ndarray
    total_velocity_violation: np.ndarray
    relative_velocity_violation: np.ndarray
    energy_delta: np.ndarray
    block_time_s: np.ndarray
    block_position_error: np.ndarray
    block_rotation_geodesic: np.ndarray
    block_relative_linear_velocity_error: np.ndarray
    block_relative_angular_velocity_error: np.ndarray
    terminal_position_error: float


@dataclass(frozen=True)
class RolloutEvaluation:
    rollout: RolloutResult
    analysis: RolloutAnalysis
    horizon_rows: list[dict]


@dataclass(frozen=True)
class TrajectorySpec:
    scenario_type: ScenarioType
    seed: int
    eta0: np.ndarray
    nu0: np.ndarray
    contract_mode: str
    gt_blocks: np.ndarray | None = None


@dataclass(frozen=True)
class GroundTruthPayload:
    scenario: str
    seed: int
    eta0: np.ndarray
    nu0: np.ndarray
    gt_pos: np.ndarray
    gt_rotation: np.ndarray
    gt_nu: np.ndarray
    gt_nu_total: np.ndarray
    time: np.ndarray
    base_state: np.ndarray | None
    pred_origin: np.ndarray | None
    block_u_cmd: np.ndarray
    block_v_c: np.ndarray | None
    completed_blocks: int
    failure_reason: str
    completed_time: float


# ---------------------------------------------------------------------------
# Model loading and simulation setup
# ---------------------------------------------------------------------------

def _instantiate_model(model_type, device, train_cfg):
    """Create an uninitialized model instance from the saved training config."""
    return train_model_builder(
        model_type,
        device,
        hidden_dim=train_cfg.hidden_dim,
        M_init=None,
        coupled_damping=getattr(train_cfg, 'coupled_damping', True),
        include_depth_in_potential=getattr(train_cfg, 'include_depth_in_potential', False),
        ocean_current=getattr(train_cfg, 'ocean_current', False),
        actuation_current_feature=getattr(train_cfg, 'actuation_current_feature', 'current_body'),
        dj_current_feature=getattr(train_cfg, 'dj_current_feature', 'none'),
        t_actuator_init=getattr(train_cfg, 't_actuator_init', None),
        u_act_scale=getattr(train_cfg, 'u_act_scale', None),
        u_dim=getattr(train_cfg, 'u_dim', 3),
        absolute_depth_context=getattr(train_cfg, 'absolute_depth_context', False),
    )


def build_model(checkpoint_path, device):
    torch = get_torch()
    _, TrainConfig = get_model_classes()

    # Training checkpoints store metadata beyond raw weights, so PyTorch 2.6+
    # must load them with weights_only=False.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_cfg = checkpoint["config"]
    cfg_dict = raw_cfg if isinstance(raw_cfg, dict) else raw_cfg.to_dict()
    train_cfg = TrainConfig.from_dict(cfg_dict)
    train_cfg.model_type = canonicalize_model_type(train_cfg.model_type)
    model_type = train_cfg.model_type
    model = _instantiate_model(model_type, device, train_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = None
    if "normalizer" in checkpoint:
        normalizer = StateNormalizer.from_dict(checkpoint["normalizer"], device=device)
    return model, train_cfg, normalizer


def load_dataset_artifact(dataset_path):
    with open(dataset_path, "rb") as handle:
        return pickle.load(handle)


def validate_benchmark_dataset_contract(dataset, train_cfg=None):
    if not isinstance(dataset, dict):
        raise ValueError("Dataset artifact must be a dict.")

    cfg = dataset.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Dataset must contain a dict config payload.")

    required = [
        "schema",
        "state_dim",
        "u_dim",
        "ocean_current",
        "velocity_convention",
        "generation_config",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(
            "Dataset config is missing required fields for rollout benchmarking: "
            + ", ".join(missing)
        )

    if str(cfg["velocity_convention"]) != "body_total":
        raise ValueError(
            "Rollout benchmark requires datasets with velocity_convention='body_total'."
        )

    expected_state_dim = 24 + (3 if bool(cfg["ocean_current"]) else 0) + (
        1 if bool(cfg.get("absolute_depth_available", False)) else 0
    )
    if int(cfg["state_dim"]) != expected_state_dim:
        raise ValueError(
            "Dataset state_dim does not match the stored ocean_current/"
            f"absolute_depth_available contract: state_dim={cfg['state_dim']}."
        )

    if train_cfg is not None:
        if bool(getattr(train_cfg, "ocean_current", False)) != bool(cfg["ocean_current"]):
            raise ValueError(
                "Checkpoint ocean_current flag disagrees with the dataset contract."
            )
        if bool(getattr(train_cfg, "absolute_depth_context", False)) != bool(
            cfg.get("absolute_depth_available", False)
        ):
            raise ValueError(
                "Checkpoint absolute_depth_context flag disagrees with the dataset contract."
            )
        if int(getattr(train_cfg, "u_dim", 3)) != int(cfg["u_dim"]):
            raise ValueError("Checkpoint u_dim disagrees with the dataset contract.")

    return cfg


def resolve_dataset_path(train_cfg, dataset_path=None):
    candidate = dataset_path or getattr(train_cfg, "dataset_path", None)
    if not candidate:
        return None
    path = Path(candidate).expanduser()
    if path.exists():
        return path.resolve()

    repo_root = Path(__file__).resolve().parent
    fallback = repo_root / "data" / path.name
    if fallback.exists():
        return fallback.resolve()

    cwd_fallback = Path.cwd() / "data" / path.name
    if cwd_fallback.exists():
        return cwd_fallback.resolve()

    return path


def resolve_generation_config_payload(train_cfg, dataset=None, dataset_path=None):
    """Resolve the full data-generation contract used for benchmark rollouts."""
    payload = getattr(train_cfg, "dataset_generation_config", None)
    if payload:
        return payload, "checkpoint"

    if dataset is not None:
        dataset_cfg = validate_benchmark_dataset_contract(dataset, train_cfg=train_cfg)
        payload = dataset_cfg.get("generation_config")
        if payload:
            return payload, "dataset"

    return None, None


def build_sim_config(rollout_time, generation_config, ocean_current=None):
    base_cfg = (
        deserialize_config(generation_config)
        if isinstance(generation_config, dict)
        else generation_config
    )
    num_blocks = int(round(rollout_time / base_cfg.dt_ctrl))

    if num_blocks <= 0:
        raise ValueError("rollout_time must be positive.")

    effective_time = num_blocks * base_cfg.dt_ctrl
    if not np.isclose(effective_time, rollout_time, atol=1e-9):
        raise ValueError(
            f"rollout_time must be a multiple of dt_ctrl={base_cfg.dt_ctrl:.2f}s; "
            f"got {rollout_time}."
        )

    sim_cfg = deserialize_config(serialize_config(base_cfg))
    sim_cfg.blocks_per_trajectory = num_blocks
    if ocean_current is not None and bool(ocean_current) != bool(sim_cfg.ocean_current):
        raise ValueError(
            "Benchmark ocean_current flag disagrees with the stored generation config."
        )
    return sim_cfg


def parse_scenarios(names):
    scenarios = []
    for name in names:
        key = name.upper()
        if key not in ScenarioType.__members__:
            raise ValueError(f"Unknown scenario: {name}")
        scenarios.append(ScenarioType[key])
    return scenarios


def sample_initial_state(rng, config):
    return sample_initial_state_from_config(rng, config)


def build_resampled_trajectory_specs(scenarios, num_traj_per_scenario, seed, sim_cfg):
    specs = []
    for scenario_idx, scenario_type in enumerate(scenarios):
        for traj_idx in range(num_traj_per_scenario):
            init_seed = seed + scenario_idx * 100000 + traj_idx
            rng = np.random.default_rng(init_seed)
            eta0, nu0 = sample_initial_state(rng, sim_cfg)
            specs.append(
                TrajectorySpec(
                    scenario_type=scenario_type,
                    seed=int(init_seed),
                    eta0=eta0,
                    nu0=nu0,
                    contract_mode="benchmark",
                )
            )
    return specs


def build_heldout_trajectory_specs(dataset, scenarios, num_traj_per_scenario):
    validate_benchmark_dataset_contract(dataset)
    meta_list = list(dataset.get("test_meta", []))
    if not meta_list:
        raise ValueError("Heldout benchmark mode requires dataset['test_meta'].")
    trajectories = dataset.get("test_trajectories")
    if trajectories is not None and len(trajectories) < len(meta_list):
        raise ValueError("Dataset test_trajectories/test_meta length mismatch.")

    requested = {scenario.name: scenario for scenario in scenarios}
    per_scenario = {name: [] for name in requested}

    for idx, meta in enumerate(meta_list):
        scenario_name = str(meta.get("scenario", "UNKNOWN"))
        if scenario_name not in requested:
            continue
        eta0 = np.asarray(meta.get("eta0"), dtype=np.float64)
        nu0 = np.asarray(meta.get("nu0"), dtype=np.float64)
        if eta0.shape != (6,) or nu0.shape != (6,):
            raise ValueError("Heldout trajectory metadata must contain eta0/nu0 with shape (6,).")
        per_scenario[scenario_name].append(
            TrajectorySpec(
                scenario_type=requested[scenario_name],
                seed=int(meta.get("seed")),
                eta0=eta0,
                nu0=nu0,
                contract_mode="dataset",
                gt_blocks=(
                    np.asarray(trajectories[idx], dtype=np.float64)
                    if trajectories is not None
                    else None
                ),
            )
        )

    specs = []
    for scenario in scenarios:
        selected = sorted(per_scenario[scenario.name], key=lambda item: item.seed)
        if len(selected) < num_traj_per_scenario:
            raise ValueError(
                f"Requested {num_traj_per_scenario} heldout trajectories for scenario "
                f"{scenario.name}, but only found {len(selected)}."
            )
        specs.extend(selected[:num_traj_per_scenario])
    return specs


# ---------------------------------------------------------------------------
# Divergence checks
# ---------------------------------------------------------------------------

def has_sim_diverged(sim, config):
    if np.any(np.isnan(sim.eta)) or np.any(np.isnan(sim.nu)):
        return True
    if abs(sim.eta[3]) > config.max_attitude or abs(sim.eta[4]) > config.max_attitude:
        return True
    if sim.eta[2] < config.depth_bounds[0] or sim.eta[2] > config.depth_bounds[1]:
        return True
    if np.any(np.abs(sim.nu) > config.velocity_max * 1.5):
        return True
    return False


def has_prediction_diverged(abs_pos, nu, config):
    if np.any(np.isnan(abs_pos)) or np.any(np.isnan(nu)):
        return True
    if np.any(np.isinf(abs_pos)) or np.any(np.isinf(nu)):
        return True
    depth = abs_pos[:, 2]
    if np.any(depth < config.depth_bounds[0]) or np.any(depth > config.depth_bounds[1]):
        return True
    if np.any(np.abs(nu) > config.velocity_max.reshape(1, -1) * 1.5):
        return True
    return False


# ---------------------------------------------------------------------------
# Rollout execution
# ---------------------------------------------------------------------------


def _exact_nu_r(sim, v_c_inertial, sim_step_idx, n_sim_steps):
    """Compute exact body-frame relative velocity nu_r = nu - R^T v_c^n.

    This matches the state convention used by AUVHamNODE and the simulator's
    current transform: total body velocity minus body-frame current.
    """
    R = rotation_matrix_from_quaternion(sim.quaternion)
    vc_idx = min(sim_step_idx, n_sim_steps - 1)
    nu = sim.nu.copy()
    nu[:3] -= R.T @ v_c_inertial[vc_idx]
    return nu


def _reconstruct_total_velocity(rotation, nu_r, v_c_inertial):
    """Recover total body velocity from relative velocity and inertial current."""
    nu_total = np.asarray(nu_r, dtype=np.float64).copy()
    nu_total[:3] += np.asarray(rotation, dtype=np.float64).T @ np.asarray(v_c_inertial, dtype=np.float64)
    return nu_total

def make_empty_rollout_result(scenario_type, seed, eta0, nu0, failure_reason):
    return RolloutResult(
        scenario=scenario_type.name,
        seed=int(seed),
        eta0=eta0,
        nu0=nu0,
        failure_reason=failure_reason,
        completed_time=0.0,
    )


def _initialize_rollout_rng(seed, sim_cfg, scenario_type, eta0, nu0, contract_mode):
    if contract_mode == "dataset":
        rng_setup, sampled_eta0, sampled_nu0, sampled_scenario = replay_trajectory_setup(seed, sim_cfg)
        if sampled_scenario != scenario_type:
            raise ValueError(
                f"Heldout scenario {scenario_type.name} does not match replayed seed contract "
                f"{sampled_scenario.name}."
            )
    elif contract_mode == "benchmark":
        rng_setup = np.random.default_rng(seed)
        sampled_eta0, sampled_nu0 = sample_initial_state(rng_setup, sim_cfg)
    else:
        raise ValueError(f"Unknown trajectory contract_mode: {contract_mode!r}.")

    if not np.allclose(sampled_eta0, eta0) or not np.allclose(sampled_nu0, nu0):
        raise ValueError(
            f"Stored initial state for seed={seed} does not match the {contract_mode} contract."
        )

    return rng_setup


def _empty_ground_truth_payload(scenario_type, seed, eta0, nu0, failure_reason):
    return GroundTruthPayload(
        scenario=scenario_type.name,
        seed=int(seed),
        eta0=np.asarray(eta0, dtype=np.float64),
        nu0=np.asarray(nu0, dtype=np.float64),
        gt_pos=_empty_pos(),
        gt_rotation=_empty_rot(),
        gt_nu=_empty_nu(),
        gt_nu_total=_empty_nu(),
        time=_empty_vector(),
        base_state=None,
        pred_origin=None,
        block_u_cmd=np.empty((0, 0), dtype=np.float64),
        block_v_c=None,
        completed_blocks=0,
        failure_reason=failure_reason,
        completed_time=0.0,
    )


def _data_state_to_model_state_array(state, use_current):
    array = np.asarray(state, dtype=np.float64)
    if not use_current:
        return array.copy()

    result = array.copy().reshape(-1, array.shape[-1])
    rotation = result[:, 3:12].reshape(-1, 3, 3)
    current = result[:, 24:27]
    current_body = np.matmul(rotation.transpose(0, 2, 1), current[..., None]).squeeze(-1)
    result[:, 12:15] -= current_body
    return result.reshape(array.shape)


def _ground_truth_cache_path(cache_dir, spec, sim_cfg, use_current):
    if cache_dir is None:
        return None

    cache_dir = Path(cache_dir)
    cache_key = {
        "version": 2,
        "scenario": spec.scenario_type.name,
        "seed": int(spec.seed),
        "eta0": np.asarray(spec.eta0, dtype=np.float64),
        "nu0": np.asarray(spec.nu0, dtype=np.float64),
        "contract_mode": spec.contract_mode,
        "ocean_current": bool(use_current),
        "sim_cfg": serialize_config(sim_cfg),
    }
    digest = hashlib.sha1(pickle.dumps(cache_key)).hexdigest()
    return cache_dir / f"{digest}.pkl"


def _load_ground_truth_payload_from_cache(cache_path):
    if cache_path is None or not cache_path.exists():
        return None
    with open(cache_path, "rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, GroundTruthPayload):
        return payload
    if isinstance(payload, dict):
        return GroundTruthPayload(**payload)
    raise ValueError(f"Unsupported ground-truth cache payload: {type(payload)!r}.")


def _save_ground_truth_payload_to_cache(cache_path, payload):
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(payload, handle)


def _simulate_warmup_start_position(
    sim_cfg,
    scenario_type,
    seed,
    eta0,
    nu0,
    contract_mode,
    use_current,
):
    rng_setup = _initialize_rollout_rng(
        seed=seed,
        sim_cfg=sim_cfg,
        scenario_type=scenario_type,
        eta0=eta0,
        nu0=nu0,
        contract_mode=contract_mode,
    )
    total_time = sim_cfg.warmup_time + sim_cfg.blocks_per_trajectory * sim_cfg.dt_ctrl + 5.0
    n_sim_steps = int(total_time / sim_cfg.dt_sim) + 100
    if use_current:
        v_c_inertial, _, _ = _generate_ocean_current(sim_cfg, rng_setup, n_sim_steps, sim_cfg.dt_sim)
    else:
        v_c_inertial = np.zeros((n_sim_steps, 3), dtype=np.float64)

    dyn = Remus100Dynamics()
    sim = Remus100Simulator(dyn, dt=sim_cfg.dt_sim)
    sim.reset(
        eta0,
        nu0,
        v_c_inertial=v_c_inertial[0] if use_current else None,
    )

    sim_step_idx = 0

    def _step_with_current(u_control):
        nonlocal sim_step_idx
        idx = min(sim_step_idx, n_sim_steps - 1)
        sim.step(
            u_control,
            v_c_inertial=v_c_inertial[idx] if use_current else None,
        )
        sim_step_idx += 1

    cmd_gen = create_scenario_generator(scenario_type, sim_cfg, seed * 1000)
    for _ in range(int(sim_cfg.warmup_time / sim_cfg.dt_sim)):
        _step_with_current(cmd_gen(sim.time))
        if has_sim_diverged(sim, sim_cfg):
            return None

    return sim.eta[:3].copy()


def _build_ground_truth_payload_from_blocks(spec, sim_cfg, use_current):
    blocks = np.asarray(spec.gt_blocks, dtype=np.float64)
    if blocks.ndim != 3:
        raise ValueError("Heldout trajectory blocks must have shape [num_blocks, points_per_block, state_dim].")
    if blocks.shape[0] < sim_cfg.blocks_per_trajectory:
        raise ValueError(
            f"Heldout trajectory has {blocks.shape[0]} blocks, expected at least {sim_cfg.blocks_per_trajectory}."
        )
    blocks = blocks[:sim_cfg.blocks_per_trajectory]
    if blocks.shape[1] != sim_cfg.points_per_block:
        raise ValueError(
            f"Heldout trajectory has {blocks.shape[1]} points/block, expected {sim_cfg.points_per_block}."
        )

    start_pos = _simulate_warmup_start_position(
        sim_cfg=sim_cfg,
        scenario_type=spec.scenario_type,
        seed=spec.seed,
        eta0=spec.eta0,
        nu0=spec.nu0,
        contract_mode=spec.contract_mode,
        use_current=use_current,
    )
    if start_pos is None:
        return _empty_ground_truth_payload(
            scenario_type=spec.scenario_type,
            seed=spec.seed,
            eta0=spec.eta0,
            nu0=spec.nu0,
            failure_reason="gt_divergence",
        )

    state_dim = blocks.shape[-1]
    depth_extra = state_dim - (27 if use_current else 24)
    if depth_extra not in (0, 1):
        raise ValueError(f"Unsupported heldout state_dim={state_dim} for benchmark payload reconstruction.")
    if use_current:
        u_dim = (state_dim - 21 - depth_extra) // 2
    else:
        u_dim = (state_dim - 18 - depth_extra) // 2
    u_cmd_start = 18 + u_dim
    u_cmd_stop = u_cmd_start + u_dim

    block_u_cmd = blocks[:, 0, u_cmd_start:u_cmd_stop].copy()
    block_v_c = blocks[:, 0, u_cmd_stop:u_cmd_stop + 3].copy() if use_current else None

    abs_pos_chunks = []
    data_chunks = []
    block_start = start_pos.copy()
    for block_idx in range(blocks.shape[0]):
        block = blocks[block_idx]
        block_abs_pos = block_start.reshape(1, 3) + block[:, :3]
        keep = slice(None) if block_idx == blocks.shape[0] - 1 else slice(0, -1)
        abs_pos_chunks.append(block_abs_pos[keep])
        data_chunks.append(block[keep])
        block_start = block_abs_pos[-1]

    traj_data = np.concatenate(data_chunks, axis=0)
    gt_pos = np.concatenate(abs_pos_chunks, axis=0)
    gt_rotation = traj_data[:, 3:12].reshape(-1, 3, 3)
    gt_nu_total = traj_data[:, 12:18].copy()
    gt_model_state = _data_state_to_model_state_array(traj_data, use_current)
    time = np.arange(len(gt_pos), dtype=np.float64) * sim_cfg.dt_state

    return GroundTruthPayload(
        scenario=spec.scenario_type.name,
        seed=int(spec.seed),
        eta0=np.asarray(spec.eta0, dtype=np.float64),
        nu0=np.asarray(spec.nu0, dtype=np.float64),
        gt_pos=gt_pos,
        gt_rotation=gt_rotation,
        gt_nu=gt_model_state[:, 12:18].copy(),
        gt_nu_total=gt_nu_total,
        time=time,
        base_state=gt_model_state[0].copy(),
        pred_origin=start_pos.copy(),
        block_u_cmd=block_u_cmd,
        block_v_c=block_v_c,
        completed_blocks=sim_cfg.blocks_per_trajectory,
        failure_reason="completed",
        completed_time=float(time[-1]) if len(time) else 0.0,
    )


def _simulate_ground_truth_payload(spec, sim_cfg, use_current, u_dim):
    rng_setup = _initialize_rollout_rng(
        seed=spec.seed,
        sim_cfg=sim_cfg,
        scenario_type=spec.scenario_type,
        eta0=spec.eta0,
        nu0=spec.nu0,
        contract_mode=spec.contract_mode,
    )

    total_time = sim_cfg.warmup_time + sim_cfg.blocks_per_trajectory * sim_cfg.dt_ctrl + 5.0
    n_sim_steps = int(total_time / sim_cfg.dt_sim) + 100
    if use_current:
        v_c_inertial, _, _ = _generate_ocean_current(sim_cfg, rng_setup, n_sim_steps, sim_cfg.dt_sim)
    else:
        v_c_inertial = np.zeros((n_sim_steps, 3), dtype=np.float64)

    dyn = Remus100Dynamics()
    sim = Remus100Simulator(dyn, dt=sim_cfg.dt_sim)
    sim.reset(
        spec.eta0,
        spec.nu0,
        v_c_inertial=v_c_inertial[0] if use_current else None,
    )

    sim_step_idx = 0

    def _step_with_current(u_control):
        nonlocal sim_step_idx
        idx = min(sim_step_idx, n_sim_steps - 1)
        sim.step(
            u_control,
            v_c_inertial=v_c_inertial[idx] if use_current else None,
        )
        sim_step_idx += 1

    cmd_gen = create_scenario_generator(spec.scenario_type, sim_cfg, spec.seed * 1000)
    for _ in range(int(sim_cfg.warmup_time / sim_cfg.dt_sim)):
        _step_with_current(cmd_gen(sim.time))
        if has_sim_diverged(sim, sim_cfg):
            return _empty_ground_truth_payload(
                scenario_type=spec.scenario_type,
                seed=spec.seed,
                eta0=spec.eta0,
                nu0=spec.nu0,
                failure_reason="gt_divergence",
            )

    total_points = sim_cfg.blocks_per_trajectory * sim_cfg.state_intervals_per_block + 1
    gt_pos = np.empty((total_points, 3), dtype=np.float64)
    gt_rotation = np.empty((total_points, 3, 3), dtype=np.float64)
    gt_nu = np.empty((total_points, 6), dtype=np.float64)
    gt_nu_total = np.empty((total_points, 6), dtype=np.float64)
    block_u_cmd = np.empty((sim_cfg.blocks_per_trajectory, u_dim), dtype=np.float64)
    block_v_c = (
        np.empty((sim_cfg.blocks_per_trajectory, 3), dtype=np.float64)
        if use_current
        else None
    )

    rotation_0 = rotation_matrix_from_quaternion(sim.quaternion).flatten()
    nu_0 = _exact_nu_r(sim, v_c_inertial, sim_step_idx, n_sim_steps) if use_current else sim.nu.copy()
    u_act_0 = sim.u_actual.copy()
    u_cmd_0 = cmd_gen(sim.time)
    base_state = np.concatenate([np.zeros(3), rotation_0, nu_0, u_act_0, u_cmd_0])
    if use_current:
        base_state = np.concatenate([base_state, v_c_inertial[min(sim_step_idx, n_sim_steps - 1)]])
    if bool(getattr(sim_cfg, "absolute_depth_context", False)):
        base_state = np.concatenate([base_state, np.array([sim.eta[2]], dtype=np.float64)])

    write_idx = 0
    completed_blocks = 0
    failure_reason = "completed"

    for block_idx in range(sim_cfg.blocks_per_trajectory):
        block_start_idx = write_idx
        u_cmd = cmd_gen(sim.time)
        block_u_cmd[block_idx] = u_cmd
        if use_current:
            block_v_c[block_idx] = v_c_inertial[min(sim_step_idx, n_sim_steps - 1)]

        for _ in range(sim_cfg.state_intervals_per_block):
            gt_pos[write_idx] = sim.eta[:3]
            gt_rotation[write_idx] = rotation_matrix_from_quaternion(sim.quaternion)
            gt_nu_total[write_idx] = sim.nu.copy()
            gt_nu[write_idx] = (
                _exact_nu_r(sim, v_c_inertial, sim_step_idx, n_sim_steps)
                if use_current
                else sim.nu.copy()
            )
            write_idx += 1
            for _ in range(sim_cfg.sim_steps_per_state):
                _step_with_current(u_cmd)
                if has_sim_diverged(sim, sim_cfg):
                    failure_reason = "gt_divergence"
                    write_idx = block_start_idx
                    break
            if failure_reason == "gt_divergence":
                break

        if failure_reason == "gt_divergence":
            break

        gt_pos[write_idx] = sim.eta[:3]
        gt_rotation[write_idx] = rotation_matrix_from_quaternion(sim.quaternion)
        gt_nu_total[write_idx] = sim.nu.copy()
        gt_nu[write_idx] = (
            _exact_nu_r(sim, v_c_inertial, sim_step_idx, n_sim_steps)
            if use_current
            else sim.nu.copy()
        )
        write_idx += 1
        completed_blocks += 1

        if block_idx < sim_cfg.blocks_per_trajectory - 1:
            write_idx -= 1

    gt_pos = gt_pos[:write_idx]
    gt_rotation = gt_rotation[:write_idx]
    gt_nu = gt_nu[:write_idx]
    gt_nu_total = gt_nu_total[:write_idx]
    time = np.arange(len(gt_pos), dtype=np.float64) * sim_cfg.dt_state

    return GroundTruthPayload(
        scenario=spec.scenario_type.name,
        seed=int(spec.seed),
        eta0=np.asarray(spec.eta0, dtype=np.float64),
        nu0=np.asarray(spec.nu0, dtype=np.float64),
        gt_pos=gt_pos,
        gt_rotation=gt_rotation,
        gt_nu=gt_nu,
        gt_nu_total=gt_nu_total,
        time=time,
        base_state=base_state if completed_blocks > 0 else None,
        pred_origin=gt_pos[0].copy() if len(gt_pos) else None,
        block_u_cmd=block_u_cmd[:completed_blocks].copy(),
        block_v_c=(block_v_c[:completed_blocks].copy() if block_v_c is not None else None),
        completed_blocks=completed_blocks,
        failure_reason=failure_reason,
        completed_time=float(time[-1]) if len(time) else 0.0,
    )


FIXED_STEP_ODE_SOLVERS = {"euler", "midpoint", "rk4"}


def _supports_batched_rollout(ode_solver):
    return str(ode_solver).lower() in FIXED_STEP_ODE_SOLVERS


def _resolve_ground_truth_payload(spec, sim_cfg, use_current, u_dim, cache_dir=None):
    if spec.gt_blocks is not None:
        return _build_ground_truth_payload_from_blocks(spec, sim_cfg, use_current)

    cache_path = _ground_truth_cache_path(cache_dir, spec, sim_cfg, use_current)
    cached = _load_ground_truth_payload_from_cache(cache_path)
    if cached is not None:
        return cached

    payload = _simulate_ground_truth_payload(spec, sim_cfg, use_current, u_dim)
    _save_ground_truth_payload_to_cache(cache_path, payload)
    return payload


def _integrate_block_with_fallback(model, train_cfg, states, t_eval_node):
    odeint = get_odeint()
    batch_size = int(states.shape[0])
    if batch_size == 0:
        return {}, {}

    try:
        pred = odeint(model, states, t_eval_node, method=train_cfg.ode_solver)
    except (ValueError, RuntimeError):
        if batch_size == 1:
            return {}, {0: "solver_failure"}
        split = batch_size // 2
        left_pred, left_fail = _integrate_block_with_fallback(
            model,
            train_cfg,
            states[:split],
            t_eval_node,
        )
        right_pred, right_fail = _integrate_block_with_fallback(
            model,
            train_cfg,
            states[split:],
            t_eval_node,
        )
        merged_pred = {idx: value for idx, value in left_pred.items()}
        merged_pred.update({split + idx: value for idx, value in right_pred.items()})
        merged_fail = {idx: value for idx, value in left_fail.items()}
        merged_fail.update({split + idx: value for idx, value in right_fail.items()})
        return merged_pred, merged_fail

    return {idx: pred[:, idx:idx + 1].clone() for idx in range(batch_size)}, {}


def _build_rollout_result_from_prediction(
    model,
    payload,
    pred_abs_states,
    pred_nu_total,
    failure_reason,
    device,
):
    pred_len = int(pred_abs_states.shape[0])
    if pred_len == 0:
        return make_empty_rollout_result(
            scenario_type=ScenarioType[payload.scenario],
            seed=payload.seed,
            eta0=payload.eta0,
            nu0=payload.nu0,
            failure_reason=failure_reason,
        )

    gt_pos = payload.gt_pos[:pred_len]
    gt_rotation = payload.gt_rotation[:pred_len]
    gt_nu = payload.gt_nu[:pred_len]
    gt_nu_total = payload.gt_nu_total[:pred_len]
    time = payload.time[:pred_len]

    energy_semantics = getattr(model, "energy_semantics", "not_comparable")
    if hasattr(model, "energy") and energy_semantics == "mechanical_energy":
        torch = get_torch()
        pred_energy = model.energy(
            torch.tensor(pred_abs_states, dtype=torch.float32, device=device)
        ).cpu().numpy()
    else:
        pred_energy = np.full(pred_len, np.nan)

    return RolloutResult(
        scenario=payload.scenario,
        seed=payload.seed,
        eta0=payload.eta0,
        nu0=payload.nu0,
        time=time,
        gt_pos=gt_pos,
        pred_pos=pred_abs_states[:, :3].copy(),
        gt_rotation=gt_rotation,
        pred_rotation_raw=pred_abs_states[:, 3:12].reshape(-1, 3, 3).copy(),
        gt_nu=gt_nu,
        pred_nu=pred_abs_states[:, 12:18].copy(),
        gt_nu_total=gt_nu_total,
        pred_nu_total=pred_nu_total,
        pred_energy=pred_energy,
        failure_reason=failure_reason,
        completed_time=float(time[-1]) if len(time) else 0.0,
    )


def _rollout_model_against_ground_truth_payloads(
    model,
    train_cfg,
    sim_cfg,
    payloads,
    device,
    noise_cfg=None,
    normalizer=None,
    noise_seed=None,
):
    torch = get_torch()
    layout = model.layout
    state_dim = model.STATE_DIM
    t_eval_node = torch.linspace(0.0, sim_cfg.dt_ctrl, sim_cfg.points_per_block, device=device)

    runtimes = []
    for payload in payloads:
        runtimes.append(
            {
                "payload": payload,
                "failure_reason": payload.failure_reason,
                "current_state": (
                    torch.tensor(payload.base_state, dtype=torch.float32, device=device)
                    if payload.base_state is not None and payload.completed_blocks > 0
                    else None
                ),
                "pred_origin": (
                    np.asarray(payload.pred_origin, dtype=np.float64).copy()
                    if payload.pred_origin is not None
                    else None
                ),
                "pred_abs_state_chunks": [],
                "pred_nu_total_chunks": [],
            }
        )

    if noise_cfg is not None and noise_cfg.is_active:
        if normalizer is None:
            raise ValueError("normalizer is required for noisy rollout benchmarking.")
        active_runtimes = [
            runtime
            for runtime in runtimes
            if runtime["current_state"] is not None
        ]
        if active_runtimes:
            init_states = torch.stack(
                [runtime["current_state"] for runtime in active_runtimes],
                dim=0,
            )
            sample_ids = torch.tensor(
                [
                    int(runtime["payload"].seed) + 100000 * (idx + 1)
                    for idx, runtime in enumerate(active_runtimes)
                ],
                dtype=torch.long,
            )
            noisy_states = build_noisy_initial_condition(
                init_states,
                noise_cfg,
                model,
                normalizer,
                epoch=noise_cfg.warmup_epochs + noise_cfg.ramp_epochs,
                sample_ids=sample_ids,
                base_seed=noise_seed,
                state_is_ode=True,
            )
            for runtime, noisy_state in zip(active_runtimes, noisy_states):
                runtime["current_state"] = noisy_state

    max_blocks = max((payload.completed_blocks for payload in payloads), default=0)
    with torch.no_grad():
        for block_idx in range(max_blocks):
            active_indices = [
                idx
                for idx, runtime in enumerate(runtimes)
                if runtime["current_state"] is not None and block_idx < runtime["payload"].completed_blocks
            ]
            if not active_indices:
                continue

            batch_state = torch.stack(
                [runtimes[idx]["current_state"] for idx in active_indices],
                dim=0,
            )
            
            # Convert carried ODE state to data state to preserve total velocity across current updates
            batch_state = model.to_data_state(batch_state)

            for local_idx, runtime_idx in enumerate(active_indices):
                payload = runtimes[runtime_idx]["payload"]
                batch_state[local_idx, layout.u_cmd] = torch.as_tensor(
                    payload.block_u_cmd[block_idx],
                    dtype=torch.float32,
                    device=device,
                )
                if payload.block_v_c is not None:
                    batch_state[local_idx, layout.v_c] = torch.as_tensor(
                        payload.block_v_c[block_idx],
                        dtype=torch.float32,
                        device=device,
                    )
                if getattr(model, "absolute_depth_context", False):
                    batch_state[local_idx, layout.depth_ref] = float(
                        runtimes[runtime_idx]["pred_origin"][2]
                    )
            
            # Convert back to ODE state using the newly assigned current
            batch_state = model.to_ode_state(batch_state)

            pred_map, failure_map = _integrate_block_with_fallback(
                model,
                train_cfg,
                batch_state,
                t_eval_node,
            )
            for local_idx, reason in failure_map.items():
                runtime = runtimes[active_indices[local_idx]]
                runtime["failure_reason"] = reason
                runtime["current_state"] = None

            success_locals = sorted(pred_map)
            if not success_locals:
                continue

            pred_ode = torch.cat([pred_map[local_idx] for local_idx in success_locals], dim=1)
            pred_data = model.to_data_state(pred_ode.reshape(-1, state_dim)).reshape(
                sim_cfg.points_per_block,
                len(success_locals),
                state_dim,
            )
            pred_ode_np = pred_ode.cpu().numpy()
            pred_data_np = pred_data.cpu().numpy()
            origins = np.stack(
                [runtimes[active_indices[local_idx]]["pred_origin"] for local_idx in success_locals],
                axis=0,
            )
            block_abs_pos = origins.reshape(1, len(success_locals), 3) + pred_data_np[:, :, :3]

            for success_idx, local_idx in enumerate(success_locals):
                runtime = runtimes[active_indices[local_idx]]
                payload = runtime["payload"]
                pred_ode_block = pred_ode_np[:, success_idx]
                pred_data_block = pred_data_np[:, success_idx]
                abs_pos_block = block_abs_pos[:, success_idx]

                if np.isnan(pred_ode_block).any() or np.isinf(pred_ode_block).any():
                    runtime["failure_reason"] = "nan_or_inf"
                    runtime["current_state"] = None
                    continue
                if has_prediction_diverged(abs_pos_block, pred_data_block[:, 12:18], sim_cfg):
                    runtime["failure_reason"] = "pred_divergence"
                    runtime["current_state"] = None
                    continue

                keep_count = (
                    sim_cfg.points_per_block
                    if block_idx == payload.completed_blocks - 1
                    else sim_cfg.points_per_block - 1
                )
                runtime["pred_abs_state_chunks"].append(
                    np.concatenate(
                        [abs_pos_block[:keep_count], pred_ode_block[:keep_count, 3:]],
                        axis=1,
                    )
                )
                runtime["pred_nu_total_chunks"].append(pred_data_block[:keep_count, 12:18].copy())

                next_state = pred_ode[-1, success_idx].clone()
                next_state[layout.pos] = 0.0
                if getattr(model, "absolute_depth_context", False):
                    next_state[layout.depth_ref] = float(abs_pos_block[-1, 2])
                runtime["current_state"] = next_state
                runtime["pred_origin"] = abs_pos_block[-1].copy()

    results = []
    for runtime in runtimes:
        payload = runtime["payload"]
        pred_abs_states = (
            np.concatenate(runtime["pred_abs_state_chunks"], axis=0)
            if runtime["pred_abs_state_chunks"]
            else np.empty((0, state_dim), dtype=np.float64)
        )
        pred_nu_total = (
            np.concatenate(runtime["pred_nu_total_chunks"], axis=0)
            if runtime["pred_nu_total_chunks"]
            else _empty_nu()
        )
        results.append(
            _build_rollout_result_from_prediction(
                model=model,
                payload=payload,
                pred_abs_states=pred_abs_states,
                pred_nu_total=pred_nu_total,
                failure_reason=runtime["failure_reason"],
                device=device,
            )
        )
    return results


def rollout_trajectory_batch(
    model,
    train_cfg,
    sim_cfg,
    specs,
    device,
    ground_truth_cache_dir=None,
    noise_cfg=None,
    normalizer=None,
    noise_seed=None,
):
    use_current = getattr(model, "ocean_current", False)
    u_dim = getattr(train_cfg, "u_dim", 3)
    payloads = [
        _resolve_ground_truth_payload(
            spec=spec,
            sim_cfg=sim_cfg,
            use_current=use_current,
            u_dim=u_dim,
            cache_dir=ground_truth_cache_dir,
        )
        for spec in specs
    ]

    if len(payloads) <= 1 or _supports_batched_rollout(train_cfg.ode_solver):
        return _rollout_model_against_ground_truth_payloads(
            model=model,
            train_cfg=train_cfg,
            sim_cfg=sim_cfg,
            payloads=payloads,
            device=device,
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
        )

    results = []
    for payload in payloads:
        results.extend(
            _rollout_model_against_ground_truth_payloads(
                model=model,
                train_cfg=train_cfg,
                sim_cfg=sim_cfg,
                payloads=[payload],
                device=device,
                noise_cfg=noise_cfg,
                normalizer=normalizer,
                noise_seed=noise_seed,
            )
        )
    return results


def rollout_single_trajectory(
    model,
    train_cfg,
    sim_cfg,
    scenario_type,
    seed,
    eta0,
    nu0,
    contract_mode,
    device,
    ground_truth_cache_dir=None,
    noise_cfg=None,
    normalizer=None,
    noise_seed=None,
):
    spec = TrajectorySpec(
        scenario_type=scenario_type,
        seed=int(seed),
        eta0=np.asarray(eta0, dtype=np.float64),
        nu0=np.asarray(nu0, dtype=np.float64),
        contract_mode=contract_mode,
    )
    return rollout_trajectory_batch(
        model=model,
        train_cfg=train_cfg,
        sim_cfg=sim_cfg,
        specs=[spec],
        device=device,
        ground_truth_cache_dir=ground_truth_cache_dir,
        noise_cfg=noise_cfg,
        normalizer=normalizer,
        noise_seed=noise_seed,
    )[0]


# ---------------------------------------------------------------------------
# Metric field definitions
# ---------------------------------------------------------------------------

MODEL_FAILURE_REASONS = {"pred_divergence", "solver_failure", "nan_or_inf"}

METRIC_FIELDS = [
    "final_position_error",
    "final_depth_error",
    "final_rotation_geodesic",
    "final_relative_linear_velocity_error",
    "final_relative_angular_velocity_error",
    "final_total_linear_velocity_error",
    "final_total_angular_velocity_error",
    "mean_position_error",
    "max_position_error",
    "mean_rotation_geodesic",
    "max_rotation_geodesic",
    "mean_relative_linear_velocity_error",
    "mean_relative_angular_velocity_error",
    "mean_total_linear_velocity_error",
    "mean_total_angular_velocity_error",
    "max_so3_det_error",
    "max_so3_orth_error",
    "mean_local_block_position_error",
    "mean_local_block_rotation_geodesic",
    "mean_local_block_relative_linear_velocity_error",
    "mean_local_block_relative_angular_velocity_error",
    "final_pred_energy",
    "energy_span",
    "max_abs_energy_delta",
    "completed_time",
]

RATE_FIELDS = [
    "completed_to_h",
    "failed_by_h",
    "fatal_failure",
    "gt_failed_by_h",
    "model_failed_by_h",
    "gt_divergence_by_h",
    "pred_divergence_by_h",
    "solver_failure_by_h",
    "nan_or_inf_by_h",
    "gt_rollout_failure",
    "model_rollout_failure",
    "any_depth_violation_up_to_h",
    "any_total_velocity_violation_up_to_h",
    "any_relative_velocity_violation_up_to_h",
    "any_so3_violation_up_to_h",
]

METRIC_VALUE_FIELDS = [
    "final_position_error",
    "final_depth_error",
    "final_rotation_geodesic",
    "final_relative_linear_velocity_error",
    "final_relative_angular_velocity_error",
    "final_total_linear_velocity_error",
    "final_total_angular_velocity_error",
    "mean_position_error",
    "max_position_error",
    "mean_rotation_geodesic",
    "max_rotation_geodesic",
    "mean_relative_linear_velocity_error",
    "mean_relative_angular_velocity_error",
    "mean_total_linear_velocity_error",
    "mean_total_angular_velocity_error",
    "max_so3_det_error",
    "max_so3_orth_error",
    "final_pred_energy",
    "energy_span",
    "max_abs_energy_delta",
    "any_depth_violation_up_to_h",
    "any_total_velocity_violation_up_to_h",
    "any_relative_velocity_violation_up_to_h",
    "any_so3_violation_up_to_h",
]

LOCAL_BLOCK_FIELDS = [
    "mean_local_block_position_error",
    "mean_local_block_rotation_geodesic",
    "mean_local_block_relative_linear_velocity_error",
    "mean_local_block_relative_angular_velocity_error",
]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def is_model_failure_reason(reason):
    return reason in MODEL_FAILURE_REASONS


def compute_rotation_metrics(gt_rotations, pred_rotations, tolerance):
    torch = get_torch()
    compute_rotation_matrix, geodesic_distance_so3 = get_rotation_helpers()

    gt_t = torch.tensor(gt_rotations, dtype=torch.float32)
    pred_raw = torch.tensor(pred_rotations, dtype=torch.float32)
    pred_proj = compute_rotation_matrix(pred_raw.reshape(-1, 9)).view(-1, 3, 3)
    geo = geodesic_distance_so3(gt_t, pred_proj).cpu().numpy()

    eye = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    det_err = (torch.det(pred_raw) - 1.0).abs().cpu().numpy()
    orth_err = (
        pred_raw @ pred_raw.transpose(1, 2) - eye
    ).norm(dim=(1, 2)).cpu().numpy()
    violation = (det_err > tolerance) | (orth_err > tolerance)
    return geo, det_err, orth_err, violation


def numeric_stats(values):
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }

    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def build_nan_metric_values():
    return {field: float("nan") for field in METRIC_VALUE_FIELDS}


def build_nan_local_block_values():
    return {field: float("nan") for field in LOCAL_BLOCK_FIELDS}


def build_metric_row_base(rollout, horizon, failed_by_h, completed_to_h):
    gt_rollout_failure = int(rollout.failure_reason == "gt_divergence")
    model_rollout_failure = int(is_model_failure_reason(rollout.failure_reason))

    return {
        "scenario": rollout.scenario,
        "trajectory_id": rollout.trajectory_id,
        "seed": rollout.seed,
        "horizon_s": float(horizon),
        "completed_time": rollout.completed_time,
        "failure_reason": rollout.failure_reason,
        "failed_by_h": failed_by_h,
        "fatal_failure": int(rollout.failure_reason != "completed"),
        "gt_failed_by_h": int(gt_rollout_failure and failed_by_h),
        "model_failed_by_h": int(model_rollout_failure and failed_by_h),
        "gt_divergence_by_h": int(rollout.failure_reason == "gt_divergence" and failed_by_h),
        "pred_divergence_by_h": int(rollout.failure_reason == "pred_divergence" and failed_by_h),
        "solver_failure_by_h": int(rollout.failure_reason == "solver_failure" and failed_by_h),
        "nan_or_inf_by_h": int(rollout.failure_reason == "nan_or_inf" and failed_by_h),
        "gt_rollout_failure": gt_rollout_failure,
        "model_rollout_failure": model_rollout_failure,
        "completed_to_h": completed_to_h,
        "eta0_z": float(rollout.eta0[2]),
        "nu0_u": float(rollout.nu0[0]),
    }


def build_empty_analysis():
    return RolloutAnalysis(
        position_error=_empty_vector(),
        depth_error=_empty_vector(),
        rotation_geodesic=_empty_vector(),
        relative_linear_velocity_error=_empty_vector(),
        relative_angular_velocity_error=_empty_vector(),
        total_linear_velocity_error=_empty_vector(),
        total_angular_velocity_error=_empty_vector(),
        so3_det_error=_empty_vector(),
        so3_orth_error=_empty_vector(),
        so3_violation=_empty_bool_vector(),
        depth_violation=_empty_bool_vector(),
        total_velocity_violation=_empty_bool_vector(),
        relative_velocity_violation=_empty_bool_vector(),
        energy_delta=_empty_vector(),
        block_time_s=_empty_vector(),
        block_position_error=_empty_vector(),
        block_rotation_geodesic=_empty_vector(),
        block_relative_linear_velocity_error=_empty_vector(),
        block_relative_angular_velocity_error=_empty_vector(),
        terminal_position_error=float("nan"),
    )


def compute_block_metrics(rollout, config):
    if len(rollout.time) == 0:
        return {
            "time_s": np.empty((0,), dtype=np.float64),
            "position_error": np.empty((0,), dtype=np.float64),
            "rotation_geodesic": np.empty((0,), dtype=np.float64),
            "relative_linear_velocity_error": np.empty((0,), dtype=np.float64),
            "relative_angular_velocity_error": np.empty((0,), dtype=np.float64),
        }

    block_step = config.state_intervals_per_block
    end_indices = np.arange(block_step, len(rollout.time), block_step, dtype=int)
    if end_indices.size == 0:
        return {
            "time_s": np.empty((0,), dtype=np.float64),
            "position_error": np.empty((0,), dtype=np.float64),
            "rotation_geodesic": np.empty((0,), dtype=np.float64),
            "relative_linear_velocity_error": np.empty((0,), dtype=np.float64),
            "relative_angular_velocity_error": np.empty((0,), dtype=np.float64),
        }

    start_indices = end_indices - block_step
    gt_delta_pos = rollout.gt_pos[end_indices] - rollout.gt_pos[start_indices]
    pred_delta_pos = rollout.pred_pos[end_indices] - rollout.pred_pos[start_indices]
    rotation_geodesic, _, _, _ = compute_rotation_metrics(
        rollout.gt_rotation[end_indices],
        rollout.pred_rotation_raw[end_indices],
        config.rotation_tolerance,
    )

    return {
        "time_s": rollout.time[end_indices],
        "position_error": np.linalg.norm(pred_delta_pos - gt_delta_pos, axis=1),
        "rotation_geodesic": rotation_geodesic,
        "relative_linear_velocity_error": np.linalg.norm(
            rollout.pred_nu[end_indices, :3] - rollout.gt_nu[end_indices, :3],
            axis=1,
        ),
        "relative_angular_velocity_error": np.linalg.norm(
            rollout.pred_nu[end_indices, 3:] - rollout.gt_nu[end_indices, 3:],
            axis=1,
        ),
    }


def analyze_rollout(rollout, config):
    if len(rollout.time) == 0:
        return build_empty_analysis()

    # Observable-space errors: these are the primary rollout metrics.
    position_error = np.linalg.norm(rollout.pred_pos - rollout.gt_pos, axis=1)
    depth_error = np.abs(rollout.pred_pos[:, 2] - rollout.gt_pos[:, 2])
    # Model-space diagnostics: useful when the dynamics are parameterized in nu_r.
    relative_linear_velocity_error = np.linalg.norm(
        rollout.pred_nu[:, :3] - rollout.gt_nu[:, :3],
        axis=1,
    )
    relative_angular_velocity_error = np.linalg.norm(
        rollout.pred_nu[:, 3:] - rollout.gt_nu[:, 3:],
        axis=1,
    )
    total_linear_velocity_error = np.linalg.norm(
        rollout.pred_nu_total[:, :3] - rollout.gt_nu_total[:, :3],
        axis=1,
    )
    total_angular_velocity_error = np.linalg.norm(
        rollout.pred_nu_total[:, 3:] - rollout.gt_nu_total[:, 3:],
        axis=1,
    )
    # Rotation consistency is shared across both conventions.
    rotation_geodesic, so3_det_error, so3_orth_error, so3_violation = compute_rotation_metrics(
        rollout.gt_rotation,
        rollout.pred_rotation_raw,
        config.rotation_tolerance,
    )

    depth_violation = (
        (rollout.pred_pos[:, 2] < config.depth_bounds[0])
        | (rollout.pred_pos[:, 2] > config.depth_bounds[1])
    )
    total_velocity_violation = np.any(
        np.abs(rollout.pred_nu_total) > config.velocity_max.reshape(1, -1),
        axis=1,
    )
    relative_velocity_violation = np.any(
        np.abs(rollout.pred_nu) > config.velocity_max.reshape(1, -1),
        axis=1,
    )
    block_metrics = compute_block_metrics(rollout, config)

    # Energy is a model-internal diagnostic, not a direct observable-space error.
    return RolloutAnalysis(
        position_error=position_error,
        depth_error=depth_error,
        rotation_geodesic=rotation_geodesic,
        relative_linear_velocity_error=relative_linear_velocity_error,
        relative_angular_velocity_error=relative_angular_velocity_error,
        total_linear_velocity_error=total_linear_velocity_error,
        total_angular_velocity_error=total_angular_velocity_error,
        so3_det_error=so3_det_error,
        so3_orth_error=so3_orth_error,
        so3_violation=so3_violation,
        depth_violation=depth_violation,
        total_velocity_violation=total_velocity_violation,
        relative_velocity_violation=relative_velocity_violation,
        energy_delta=rollout.pred_energy - rollout.pred_energy[0],
        block_time_s=block_metrics["time_s"],
        block_position_error=block_metrics["position_error"],
        block_rotation_geodesic=block_metrics["rotation_geodesic"],
        block_relative_linear_velocity_error=block_metrics["relative_linear_velocity_error"],
        block_relative_angular_velocity_error=block_metrics["relative_angular_velocity_error"],
        terminal_position_error=float(position_error[-1]),
    )


def summarize_local_block_metrics(analysis, horizon, completed_to_h):
    if not completed_to_h or analysis.block_time_s.size == 0:
        return build_nan_local_block_values()

    valid_count = int(np.searchsorted(analysis.block_time_s, horizon, side="right"))
    if valid_count <= 0:
        return build_nan_local_block_values()

    return {
        "mean_local_block_position_error": float(
            np.mean(analysis.block_position_error[:valid_count])
        ),
        "mean_local_block_rotation_geodesic": float(
            np.mean(analysis.block_rotation_geodesic[:valid_count])
        ),
        "mean_local_block_relative_linear_velocity_error": float(
            np.mean(analysis.block_relative_linear_velocity_error[:valid_count])
        ),
        "mean_local_block_relative_angular_velocity_error": float(
            np.mean(analysis.block_relative_angular_velocity_error[:valid_count])
        ),
    }


def build_rollout_metric_rows(rollout, analysis, horizons):
    rollout_failed = int(rollout.failure_reason != "completed")

    if len(rollout.time) == 0:
        rows = []
        for horizon in horizons:
            failed_by_h = int(rollout_failed and rollout.completed_time + 1e-9 < horizon)
            rows.append(
                {
                    **build_metric_row_base(
                        rollout=rollout,
                        horizon=horizon,
                        failed_by_h=failed_by_h,
                        completed_to_h=0,
                    ),
                    **build_nan_metric_values(),
                    **build_nan_local_block_values(),
                }
            )
        return rows

    rows = []
    for horizon in horizons:
        completed_to_h = int(rollout.completed_time + 1e-9 >= horizon)
        failed_by_h = int(rollout_failed and not completed_to_h)
        row_base = build_metric_row_base(
            rollout=rollout,
            horizon=horizon,
            failed_by_h=failed_by_h,
            completed_to_h=completed_to_h,
        )

        if completed_to_h:
            idx = min(
                np.searchsorted(rollout.time, horizon, side="right") - 1,
                len(rollout.time) - 1,
            )
            metric_values = {
                "final_position_error": float(analysis.position_error[idx]),
                "final_depth_error": float(analysis.depth_error[idx]),
                "final_rotation_geodesic": float(analysis.rotation_geodesic[idx]),
                "final_relative_linear_velocity_error": float(
                    analysis.relative_linear_velocity_error[idx]
                ),
                "final_relative_angular_velocity_error": float(
                    analysis.relative_angular_velocity_error[idx]
                ),
                "final_total_linear_velocity_error": float(
                    analysis.total_linear_velocity_error[idx]
                ),
                "final_total_angular_velocity_error": float(
                    analysis.total_angular_velocity_error[idx]
                ),
                "mean_position_error": float(np.mean(analysis.position_error[: idx + 1])),
                "max_position_error": float(np.max(analysis.position_error[: idx + 1])),
                "mean_rotation_geodesic": float(np.mean(analysis.rotation_geodesic[: idx + 1])),
                "max_rotation_geodesic": float(np.max(analysis.rotation_geodesic[: idx + 1])),
                "mean_relative_linear_velocity_error": float(
                    np.mean(analysis.relative_linear_velocity_error[: idx + 1])
                ),
                "mean_relative_angular_velocity_error": float(
                    np.mean(analysis.relative_angular_velocity_error[: idx + 1])
                ),
                "mean_total_linear_velocity_error": float(
                    np.mean(analysis.total_linear_velocity_error[: idx + 1])
                ),
                "mean_total_angular_velocity_error": float(
                    np.mean(analysis.total_angular_velocity_error[: idx + 1])
                ),
                "max_so3_det_error": float(np.max(analysis.so3_det_error[: idx + 1])),
                "max_so3_orth_error": float(np.max(analysis.so3_orth_error[: idx + 1])),
                "final_pred_energy": float(rollout.pred_energy[idx]),
                "energy_span": float(
                    np.max(rollout.pred_energy[: idx + 1])
                    - np.min(rollout.pred_energy[: idx + 1])
                ),
                "max_abs_energy_delta": float(
                    np.max(np.abs(analysis.energy_delta[: idx + 1]))
                ),
                "any_depth_violation_up_to_h": int(
                    np.any(analysis.depth_violation[: idx + 1])
                ),
                "any_total_velocity_violation_up_to_h": int(
                    np.any(analysis.total_velocity_violation[: idx + 1])
                ),
                "any_relative_velocity_violation_up_to_h": int(
                    np.any(analysis.relative_velocity_violation[: idx + 1])
                ),
                "any_so3_violation_up_to_h": int(
                    np.any(analysis.so3_violation[: idx + 1])
                ),
            }
        else:
            metric_values = build_nan_metric_values()

        rows.append(
            {
                **row_base,
                **metric_values,
                **summarize_local_block_metrics(analysis, horizon, completed_to_h),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_rollout(rollout, horizons, config):
    analysis = analyze_rollout(rollout, config)
    return RolloutEvaluation(
        rollout=rollout,
        analysis=analysis,
        horizon_rows=build_rollout_metric_rows(rollout, analysis, horizons),
    )


# ---------------------------------------------------------------------------
# Aggregation and summary
# ---------------------------------------------------------------------------

def aggregate_horizon_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["horizon_s"]), []).append(row)
        grouped.setdefault(("ALL", row["horizon_s"]), []).append(row)

    summary_rows = []
    summary_json = {"overall": {}, "by_scenario": {}}

    for (scenario, horizon), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        row = {
            "scenario": scenario,
            "horizon_s": horizon,
            "n_trajectories": len(group),
        }
        metrics = {}
        for field in METRIC_FIELDS:
            stats = numeric_stats([item[field] for item in group])
            metrics[field] = stats
            for stat_name, value in stats.items():
                row[f"{field}_{stat_name}"] = value

        rates = {}
        for field in RATE_FIELDS:
            values = np.asarray([float(item[field]) for item in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            rate = float(np.mean(values)) if values.size else float("nan")
            rates[field] = rate
            row[f"{field}_rate"] = rate

        summary_rows.append(row)

        target = (
            summary_json["overall"]
            if scenario == "ALL"
            else summary_json["by_scenario"].setdefault(scenario, {})
        )
        target[str(horizon)] = {
            "n_trajectories": len(group),
            "metrics": metrics,
            "rates": rates,
        }

    return summary_rows, summary_json


def summarize_rollout_outcomes(rows, scenarios, max_horizon):
    max_rows = [
        row
        for row in rows
        if np.isclose(float(row["horizon_s"]), float(max_horizon))
    ]
    scenario_names = [item.name if hasattr(item, "name") else str(item) for item in scenarios]

    return {
        "overall": _summarize_rollout_subset(max_rows),
        "by_scenario": {
            scenario_name: _summarize_rollout_subset(
                [row for row in max_rows if row["scenario"] == scenario_name]
            )
            for scenario_name in scenario_names
        },
    }


def _summarize_rollout_subset(rows):
    counts = {
        "completed": 0,
        "gt_divergence": 0,
        "pred_divergence": 0,
        "solver_failure": 0,
        "nan_or_inf": 0,
        "other_failure": 0,
    }
    for row in rows:
        reason = row["failure_reason"]
        if reason in counts:
            counts[reason] += 1
        else:
            counts["other_failure"] += 1

    total = len(rows)
    rates = {
        key: (float(value / total) if total else float("nan"))
        for key, value in counts.items()
    }
    return {
        "n_trajectories": total,
        "counts": counts,
        "rates": rates,
    }


def summarize_time_series(evaluations, scenarios, max_time, dt_state):
    time_grid = np.arange(0.0, max_time + 0.5 * dt_state, dt_state)
    rows = []

    def add_rows(label, subset):
        for idx, time_value in enumerate(time_grid):
            pos_values = []
            rot_values = []
            rel_lin_values = []
            rel_ang_values = []
            total_lin_values = []
            total_ang_values = []
            survivors = 0

            for evaluation in subset:
                rollout = evaluation.rollout
                analysis = evaluation.analysis
                if idx < len(rollout.time):
                    survivors += 1
                    pos_values.append(float(analysis.position_error[idx]))
                    rot_values.append(float(analysis.rotation_geodesic[idx]))
                    rel_lin_values.append(float(analysis.relative_linear_velocity_error[idx]))
                    rel_ang_values.append(float(analysis.relative_angular_velocity_error[idx]))
                    total_lin_values.append(float(analysis.total_linear_velocity_error[idx]))
                    total_ang_values.append(float(analysis.total_angular_velocity_error[idx]))

            row = {
                "scenario": label,
                "time_s": float(time_value),
                "n_alive": survivors,
                "survival_rate": float(survivors / len(subset)) if subset else float("nan"),
            }
            for prefix, values in [
                ("position_error", pos_values),
                ("rotation_geodesic", rot_values),
                ("relative_linear_velocity_error", rel_lin_values),
                ("relative_angular_velocity_error", rel_ang_values),
                ("total_linear_velocity_error", total_lin_values),
                ("total_angular_velocity_error", total_ang_values),
            ]:
                stats = numeric_stats(values)
                for stat_name, value in stats.items():
                    row[f"{prefix}_{stat_name}"] = value
            rows.append(row)

    add_rows("ALL", evaluations)
    for scenario in scenarios:
        subset = [
            evaluation
            for evaluation in evaluations
            if evaluation.rollout.scenario == scenario.name
        ]
        add_rows(scenario.name, subset)

    return rows
