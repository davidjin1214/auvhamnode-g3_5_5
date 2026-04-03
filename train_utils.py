"""
Training utilities for AUVHamNODE.

Provides: configuration, data loading, SE(3) loss functions, logging,
checkpointing, and evaluation metrics.
"""

import dataclasses
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import pickle
import json
import logging
import csv
import hashlib
import re


def load_dataset(dataset_path: str) -> Dict:
    with open(dataset_path, "rb") as f:
        return pickle.load(f)


DATASET_TRAINING_DEFAULTS = {
    "noc": {
        "batch_size": 2048,
        "num_epochs": 200,
        "learning_rate": 5e-3,
        "min_learning_rate": 1e-4,
        "warmup_steps": 300,
        "total_steps": 7000,
        "ocean_current": False,
        "dj_current_feature": "none",
        "actuation_current_feature": "current_body",
        "mass_init": "remus",
        "t_actuator_init": None,
        "u_act_scale": None,
    },
    "oc": {
        "batch_size": 4096,
        "num_epochs": 300,
        "learning_rate": 6e-3,
        "min_learning_rate": 1e-4,
        "warmup_steps": 400,
        "total_steps": 5000,
        "ocean_current": True,
        "dj_current_feature": "current_body",
        "actuation_current_feature": "current_body",
        "mass_init": "remus",
        "t_actuator_init": [0.1, 0.1, 1.0],
        "u_act_scale": [1.0, 1.0, 0.001],
    },
}


def infer_dataset_kind_from_path(dataset_path: str) -> str:
    """Infer whether a dataset path corresponds to the `noc` or `oc` recipe."""
    name = Path(dataset_path).name.lower()
    if "noc" in name:
        return "noc"
    if re.search(r"(^|[_-])oc([_-]|$)", name):
        return "oc"
    return "noc"


def get_dataset_training_defaults(
    dataset_path: Optional[str] = None,
    dataset_kind: Optional[str] = None,
) -> Dict:
    """Return a copy of the default training recipe for the dataset kind."""
    kind = dataset_kind or infer_dataset_kind_from_path(dataset_path or "")
    if kind not in DATASET_TRAINING_DEFAULTS:
        raise ValueError(f"Unsupported dataset kind: {kind}")
    return deepcopy(DATASET_TRAINING_DEFAULTS[kind])


def setup_logging(log_dir: Path, name: str = "training") -> logging.Logger:
    """Setup file + console logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt_file = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                 datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt_file)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


@dataclass
class TrainConfig:
    """Training configuration."""

    model_type: str = "phnode_full"
    hidden_dim: int = 128
    coupled_damping: bool = True
    include_depth_in_potential: bool = False

    batch_size: int = 2048
    num_epochs: int = 200
    learning_rate: float = 5e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 300
    total_steps: int = 7000
    weight_decay: float = 1e-4
    ode_solver: str = "rk4"
    so3_regularization_weight: float = 1e-3
    actuator_loss_weight: float = 0.2

    log_interval: int = 1
    save_interval: int = 50
    save_dir: str = "./checkpoints"
    run_name: Optional[str] = None

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Training-time state/observation perturbations
    init_state_noise: bool = False
    init_rot_std: float = 0.005           # rad (small attitude perturbation)
    velocity_noise_std: List[float] = field(
        default_factory=lambda: [0.02, 0.01, 0.01, 0.005, 0.005, 0.005])
    actuator_noise_std: List[float] = field(
        default_factory=lambda: [0.001, 0.001, 5.0])
    observation_noise: bool = False
    observation_noise_scale: float = 0.5  # fraction of init-state noise std
    observation_bias: bool = False        # persistent block-wise velocity bias
    noise_ramp_epochs: int = 100          # ramp from 0 to full over N epochs
    velocity_bias_std: List[float] = field(
        default_factory=lambda: [0.005, 0.003, 0.003, 0.001, 0.001, 0.001])
    current_noise_std: float = 0.05       # m/s

    # Ocean current
    ocean_current: bool = False
    absolute_depth_context: bool = False
    u_dim: int = 3
    dj_current_feature: str = "none"
    actuation_current_feature: str = "current_body"
    mass_init: str = "none"
    mass_init_path: Optional[str] = None
    t_actuator_init: Optional[List[float]] = None
    u_act_scale: Optional[List[float]] = None
    dataset_path: Optional[str] = None
    dataset_id: Optional[str] = None
    dataset_velocity_convention: Optional[str] = None
    dataset_description: Optional[str] = None
    dataset_state_dim: Optional[int] = None
    dataset_generation_config: Optional[Dict] = None

    @classmethod
    def dataset_default_overrides(cls, dataset_kind: str) -> Dict:
        """Return dataset-specific default hyperparameters."""
        return get_dataset_training_defaults(dataset_kind=dataset_kind)

    @classmethod
    def for_dataset_kind(cls, dataset_kind: str, **overrides) -> "TrainConfig":
        """Construct a config from dataset-specific defaults plus explicit overrides."""
        payload = cls.dataset_default_overrides(dataset_kind)
        payload.update(overrides)
        return cls(**payload)

    def _default_run_name(self) -> str:
        """Generate a readable, collision-resistant run name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        lr = f"{self.learning_rate:g}".replace("+", "")

        signature_payload = self.to_dict().copy()
        signature_payload["run_name"] = None
        signature_payload["save_dir"] = None
        config_hash = hashlib.sha1(
            json.dumps(signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]

        tags = []
        if self.init_state_noise:
            tags.append("init-noise")
        if self.observation_noise:
            tags.append("obs-noise")
        if self.ocean_current:
            tags.append("oc")
        tag_suffix = f"_{'-'.join(tags)}" if tags else ""

        return (
            f"{self.model_type}_h{self.hidden_dim}_bs{self.batch_size}"
            f"_lr{lr}_steps{self.total_steps}_seed{self.seed}"
            f"{tag_suffix}_{timestamp}_{config_hash}"
        )

    def get_run_dir(self) -> Path:
        if self.run_name is None:
            self.run_name = self._default_run_name()
        return Path(self.save_dir) / self.run_name

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainConfig":
        valid = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(
                "Unknown TrainConfig fields: "
                + ", ".join(unknown)
            )
        return cls(**data)


@dataclass
class StateNormalizer:
    """
    Per-component standard deviations for balanced loss computation.

    Fields:
      std_pos: [3] position standard deviations
      std_vel: [6] velocity standard deviations [u, v, w, p, q, r]
      std_act: [u_dim] actuator-state standard deviations
    """
    std_pos: torch.Tensor
    std_vel: torch.Tensor
    std_act: Optional[torch.Tensor] = None
    std_vel_data: Optional[torch.Tensor] = None

    @classmethod
    def from_dataset(
        cls,
        data: np.ndarray,
        device: str = "cpu",
        u_dim: int = 3,
    ) -> "StateNormalizer":
        state_dim = data.shape[-1]
        flat = data.reshape(-1, state_dim)
        eps = 1e-8

        vel_data = flat[:, 12:18]
        u_act_slice, _, _ = actuator_slices(u_dim=u_dim, ocean_current=state_dim >= 27)
        act_data = flat[:, u_act_slice]
        return cls(
            std_pos=torch.tensor(
                np.std(flat[:, :3], axis=0) + eps, dtype=torch.float32, device=device),
            std_vel=torch.tensor(
                np.std(vel_data, axis=0) + eps, dtype=torch.float32, device=device),
            std_act=torch.tensor(
                np.std(act_data, axis=0) + eps, dtype=torch.float32, device=device),
            std_vel_data=torch.tensor(
                np.std(vel_data, axis=0) + eps, dtype=torch.float32, device=device),
        )

    def to(self, device: str) -> "StateNormalizer":
        self.std_pos = self.std_pos.to(device)
        self.std_vel = self.std_vel.to(device)
        if self.std_act is not None:
            self.std_act = self.std_act.to(device)
        if self.std_vel_data is not None:
            self.std_vel_data = self.std_vel_data.to(device)
        return self

    def to_dict(self) -> Dict:
        payload = {
            "std_pos": self.std_pos.cpu().tolist(),
            "std_vel": self.std_vel.cpu().tolist(),
        }
        if self.std_act is not None:
            payload["std_act"] = self.std_act.cpu().tolist()
        if self.std_vel_data is not None:
            payload["std_vel_data"] = self.std_vel_data.cpu().tolist()
        return payload

    @classmethod
    def from_dict(cls, d: Dict, device: str = "cpu") -> "StateNormalizer":
        return cls(
            std_pos=torch.tensor(d["std_pos"], dtype=torch.float32, device=device),
            std_vel=torch.tensor(d["std_vel"], dtype=torch.float32, device=device),
            std_act=(
                torch.tensor(d["std_act"], dtype=torch.float32, device=device)
                if "std_act" in d else None
            ),
            std_vel_data=torch.tensor(
                d.get("std_vel_data", d["std_vel"]),
                dtype=torch.float32,
                device=device,
            ),
        )

    def summary(self) -> str:
        p = self.std_pos.cpu().numpy()
        v_ode = self.std_vel.cpu().numpy()
        lines = [
            f"Position std: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]",
            ("Velocity std: "
             f"u={v_ode[0]:.4f}, v={v_ode[1]:.4f}, w={v_ode[2]:.4f}, "
             f"p={v_ode[3]:.4f}, q={v_ode[4]:.4f}, r={v_ode[5]:.4f}"),
        ]
        if self.std_act is not None:
            a = self.std_act.cpu().numpy()
            act_stats = ", ".join(f"a{i}={value:.4f}" for i, value in enumerate(a))
            lines.append(f"Actuator std: {act_stats}")
        return "\n".join(lines)


def actuator_slices(u_dim: int, ocean_current: bool):
    """Return slices for actuator states, commands, and current channels."""
    u_act = slice(18, 18 + u_dim)
    u_cmd = slice(u_act.stop, u_act.stop + u_dim)
    v_c = slice(u_cmd.stop, u_cmd.stop + 3) if ocean_current else None
    return u_act, u_cmd, v_c


def _cfg_value(cfg: Dict, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def dataset_velocity_convention(cfg: Dict) -> str:
    """Return the authoritative stored twist convention of the dataset."""
    return str(_cfg_value(cfg, "velocity_convention", "body_total"))


def validate_dataset_config(cfg: Dict):
    """Reject datasets that do not follow the current storage contract."""
    convention = dataset_velocity_convention(cfg)
    if convention != "body_total":
        raise ValueError(
            "Unsupported dataset velocity convention "
            f"{convention!r}; regenerate the dataset with stored total body velocity."
        )


def validate_depth_conditioning_support(
    dataset_cfg: Optional[Dict],
    include_depth_in_potential: bool,
    context: str,
):
    """Reject depth-conditioned models when the dataset only stores block-relative position."""
    if not include_depth_in_potential:
        return
    if not isinstance(dataset_cfg, dict):
        raise ValueError(
            f"{context}: include_depth_in_potential requires a dataset config with "
            "an explicit absolute-depth contract."
        )

    frame_cfg = dataset_cfg.get("frame_convention", {})
    position_mode = (
        frame_cfg.get("position")
        if isinstance(frame_cfg, dict)
        else None
    )
    has_absolute_depth = bool(dataset_cfg.get("absolute_depth_available", False))
    if position_mode == "inertial_relative" and not has_absolute_depth:
        raise ValueError(
            f"{context}: include_depth_in_potential is incompatible with the current "
            "dataset contract because the state stores block-relative position Δpos, "
            "not absolute depth. Disable depth conditioning or regenerate data with "
            "an explicit absolute-depth field."
        )


def _compute_nu_r_from_total(data: torch.Tensor) -> torch.Tensor:
    """Convert stored total body velocity to model-space relative velocity."""
    shape = data.shape[:-1]
    flat = data.reshape(-1, data.shape[-1])
    R = flat[:, 3:12].reshape(-1, 3, 3)
    v_c_n = flat[:, 24:27]
    v_c_body = torch.bmm(R.transpose(1, 2), v_c_n.unsqueeze(-1)).squeeze(-1)
    result = flat.clone()
    result[:, 12:15] = flat[:, 12:15] - v_c_body
    return result.reshape(*shape, flat.shape[-1])


def adapt_state_tensor_for_model(
    data: torch.Tensor,
    dataset_cfg: Dict,
    ocean_current: bool,
) -> torch.Tensor:
    """Map stored dataset states into the model's internal state convention."""
    validate_dataset_config(dataset_cfg)
    if not ocean_current:
        return data
    return _compute_nu_r_from_total(data)


def adapt_state_array_for_model(
    data: np.ndarray,
    dataset_cfg: Dict,
    ocean_current: bool,
) -> np.ndarray:
    """NumPy wrapper around the canonical state adapter."""
    tensor = torch.tensor(data, dtype=torch.float32)
    return adapt_state_tensor_for_model(tensor, dataset_cfg, ocean_current).numpy()


def resolve_feature_vector(values, expected_dim: int, name: str) -> torch.Tensor:
    """Resolve a scalar/list config field to the expected actuator dimension."""
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.numel() == expected_dim:
        return tensor
    if tensor.numel() == 1:
        return tensor.repeat(expected_dim)
    raise ValueError(f"{name} must have length 1 or u_dim={expected_dim}, got {tensor.numel()}.")


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Batch skew-symmetric matrix from [batch, 3] vectors."""
    z = torch.zeros_like(v[:, 0])
    return torch.stack([
        z,      -v[:, 2],  v[:, 1],
        v[:, 2],  z,      -v[:, 0],
       -v[:, 1],  v[:, 0],  z,
    ], dim=1).view(-1, 3, 3)


def _skew_symmetric_sequence(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrices for [..., 3] vectors."""
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        z,         -v[..., 2],  v[..., 1],
        v[..., 2],  z,         -v[..., 0],
       -v[..., 1],  v[..., 0],  z,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def _correlated_gaussian_noise(
    batch_size: int,
    steps: int,
    dim: int,
    std: torch.Tensor,
    device: torch.device,
    correlation: float = 0.85,
) -> torch.Tensor:
    """Sample zero-mean temporally correlated Gaussian noise."""
    std = std.view(1, 1, dim)
    noise = torch.zeros(batch_size, steps, dim, device=device)
    innovation_scale = float(max(1.0 - correlation ** 2, 1e-6)) ** 0.5

    noise[:, 0] = std[:, 0] * torch.randn(batch_size, dim, device=device)
    for idx in range(1, steps):
        innovation = std[:, 0] * torch.randn(batch_size, dim, device=device)
        noise[:, idx] = correlation * noise[:, idx - 1] + innovation_scale * innovation
    return noise


def _body_current_from_state_rotation(
    state: torch.Tensor,
    v_c_slice: slice,
) -> torch.Tensor:
    """Compute body-frame current R^T v_c^n for flattened state batches."""
    R = state[:, 3:12].reshape(-1, 3, 3)
    v_c_n = state[:, v_c_slice]
    return torch.bmm(R.transpose(1, 2), v_c_n.unsqueeze(-1)).squeeze(-1)


def _apply_velocity_and_current_noise(
    state: torch.Tensor,
    vel_noise: torch.Tensor,
    current_noise: Optional[torch.Tensor],
    *,
    u_dim: int,
    ocean_current: bool,
    state_convention: str,
) -> torch.Tensor:
    """Apply velocity/current noise while preserving the ocean-current kinematic contract."""
    result = state.clone()

    _, _, v_c_slice = actuator_slices(u_dim, ocean_current)
    if state_convention == "data":
        result[..., 12:18] += vel_noise
        if (
            ocean_current
            and v_c_slice is not None
            and result.shape[-1] >= v_c_slice.stop
            and current_noise is not None
        ):
            result[..., v_c_slice] += current_noise
        return result

    if state_convention == "ode":
        flat_state = state.reshape(-1, state.shape[-1])
        flat_result = result.reshape(-1, result.shape[-1])
        vel_noise_flat = vel_noise.reshape(-1, vel_noise.shape[-1])

        if ocean_current and v_c_slice is not None and state.shape[-1] >= v_c_slice.stop:
            old_v_c_body = _body_current_from_state_rotation(flat_state, v_c_slice)
            noisy_total = flat_state[:, 12:18].clone()
            noisy_total[:, :3] += old_v_c_body
            noisy_total += vel_noise_flat

            if current_noise is not None:
                flat_result[:, v_c_slice] += current_noise.reshape(-1, current_noise.shape[-1])
            new_v_c_body = _body_current_from_state_rotation(flat_result, v_c_slice)
            flat_result[:, 12:15] = noisy_total[:, :3] - new_v_c_body
            flat_result[:, 15:18] = noisy_total[:, 3:]
            return flat_result.reshape(result.shape)

        flat_result[:, 12:18] = flat_state[:, 12:18] + vel_noise_flat
        return flat_result.reshape(result.shape)

    raise ValueError(f"Unsupported state_convention: {state_convention!r}")


def apply_initial_condition_noise(
    y0: torch.Tensor,
    config: TrainConfig,
    epoch: int,
) -> torch.Tensor:
    """Add sensor noise to initial conditions for robust training.

    Applies noise only to the initial state (first time step) passed to the
    ODE solver.  The noise level ramps linearly from 0 to full over
    ``config.noise_ramp_epochs`` epochs (curriculum learning).

    Args:
        y0:     [batch, state_dim]  initial conditions
        config: training configuration with noise parameters
        epoch:  current epoch (1-based) for curriculum scaling

    Returns:
        Noisy copy of y0 (does not modify the original tensor).
    """
    if not config.init_state_noise:
        return y0

    # Curriculum scaling: linear ramp from 0 → 1
    scale = min(epoch / max(config.noise_ramp_epochs, 1), 1.0)
    if scale < 1e-8:
        return y0

    y0n = y0.clone()
    device = y0.device

    # Relative block position is defined against the block origin, so adding an
    # arbitrary translation offset to y0 would be inconsistent with the state.

    # --- Rotation noise (small-angle SO(3) perturbation + re-projection) ---
    R = y0n[:, 3:12].view(-1, 3, 3)
    angle_noise = scale * config.init_rot_std * torch.randn(
        y0.shape[0], 3, device=device)
    # Exponential map approximation: R_noisy ≈ (I + [δθ]×) R
    skew = _skew_symmetric(angle_noise)
    R_noisy = (torch.eye(3, device=device).unsqueeze(0) + skew) @ R
    # Re-project to SO(3) via SVD
    U, _, Vt = torch.linalg.svd(R_noisy)
    # Ensure det = +1
    det = torch.det(U @ Vt)
    sign = torch.ones(y0.shape[0], 3, device=device)
    sign[:, -1] = det.sign()
    R_proj = U * sign.unsqueeze(1) @ Vt
    y0n[:, 3:12] = R_proj.reshape(-1, 9)

    # --- Velocity noise ---
    vel_std = torch.tensor(
        config.velocity_noise_std, dtype=torch.float32, device=device)
    vel_noise = scale * vel_std * torch.randn(y0.shape[0], 6, device=device)

    # --- Actuator state noise ---
    u_act_slice, _, v_c_slice = actuator_slices(config.u_dim, config.ocean_current)
    act_std = resolve_feature_vector(
        config.actuator_noise_std, config.u_dim, "actuator_noise_std").to(device)
    y0n[:, u_act_slice] += scale * act_std * torch.randn(
        y0.shape[0], config.u_dim, device=device)

    # --- Velocity bias (constant offset simulating IMU/DVL bias drift) ---
    if config.observation_bias:
        bias_std = torch.tensor(
            config.velocity_bias_std, dtype=torch.float32, device=device)
        vel_noise = vel_noise + scale * bias_std * torch.randn(
            y0.shape[0], 6, device=device)

    # --- Ocean current estimation noise ---
    current_noise = None
    if v_c_slice is not None and y0.shape[1] >= v_c_slice.stop:
        current_noise = scale * config.current_noise_std * torch.randn(
            y0.shape[0], 3, device=device)

    y0n = _apply_velocity_and_current_noise(
        y0n,
        vel_noise,
        current_noise,
        u_dim=config.u_dim,
        ocean_current=config.ocean_current,
        state_convention="ode",
    )

    return y0n


def apply_trajectory_observation_noise(
    target: torch.Tensor,
    config: TrainConfig,
    epoch: int,
    t_eval: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Add observation noise to ground-truth trajectory for robust training.

    Uses a simple navigation-style model:
      - body-frame velocity observations carry temporally correlated noise
      - optional per-block bias stays constant over the short rollout window
      - relative position error is induced by integrating velocity error
      - attitude error is induced by integrating angular-rate error

    This keeps the model lightweight while avoiding per-time-step i.i.d.
    position corruption, which is not well matched to relative-block states.

    Args:
        target: [batch, T, state_dim]
        config: training configuration with noise parameters
        epoch:  current epoch for curriculum scaling
        t_eval: [T] sample times inside one block

    Returns:
        Noisy copy of target.
    """
    if not config.observation_noise:
        return target

    scale = min(epoch / max(config.noise_ramp_epochs, 1), 1.0)
    if scale < 1e-8:
        return target

    target_n = target.clone()
    device = target.device
    B, T = target.shape[0], target.shape[1]
    s = scale * config.observation_noise_scale
    if T <= 1:
        return target_n

    if t_eval is None:
        dt = torch.ones(T - 1, dtype=target.dtype, device=device)
    else:
        dt = torch.diff(t_eval.to(device=device, dtype=target.dtype))

    vel_std = s * torch.tensor(
        config.velocity_noise_std, dtype=target.dtype, device=device)
    vel_noise = _correlated_gaussian_noise(B, T, 6, vel_std, device=device)

    if config.observation_bias:
        bias_std = s * torch.tensor(
            config.velocity_bias_std, dtype=target.dtype, device=device)
        vel_noise = vel_noise + bias_std.view(1, 1, 6) * torch.randn(
            B, 1, 6, device=device)

    _, _, v_c_slice = actuator_slices(config.u_dim, config.ocean_current)
    current_noise = None
    if v_c_slice is not None and target.shape[-1] >= v_c_slice.stop:
        current_std = torch.full((3,), s * config.current_noise_std, dtype=target.dtype, device=device)
        current_noise = _correlated_gaussian_noise(B, T, 3, current_std, device=device)

    target_n = _apply_velocity_and_current_noise(
        target_n,
        vel_noise,
        current_noise,
        u_dim=config.u_dim,
        ocean_current=config.ocean_current,
        state_convention="data",
    )

    # Integrate translational velocity error to obtain relative-position error.
    R_ref = target[..., 3:12].reshape(B, T, 3, 3)
    vel_inertial_noise = torch.matmul(
        R_ref[:, :-1],
        vel_noise[:, :-1, :3].unsqueeze(-1),
    ).squeeze(-1)
    pos_noise = torch.zeros(B, T, 3, dtype=target.dtype, device=device)
    pos_noise[:, 1:] = torch.cumsum(
        dt.view(1, T - 1, 1) * vel_inertial_noise,
        dim=1,
    )
    target_n[..., :3] += pos_noise

    # Integrate angular-rate error to obtain a small-angle attitude mismatch.
    angle_steps = dt.view(1, T - 1, 1) * vel_noise[:, :-1, 3:6]
    angle_error = torch.zeros(B, T, 3, dtype=target.dtype, device=device)
    angle_error[:, 1:] = torch.cumsum(angle_steps, dim=1)
    R_delta = torch.eye(3, dtype=target.dtype, device=device).view(1, 1, 3, 3)
    R_delta = R_delta + _skew_symmetric_sequence(angle_error)
    R_noisy = torch.matmul(R_delta, R_ref)
    U, _, Vh = torch.linalg.svd(R_noisy)
    det = torch.det(torch.matmul(U, Vh))
    sign = torch.ones(B, T, 3, dtype=target.dtype, device=device)
    sign[..., -1] = det.sign()
    R_proj = torch.matmul(U * sign.unsqueeze(-2), Vh)
    target_n[..., 3:12] = R_proj.reshape(B, T, 9)

    return target_n


class AUVDataset(Dataset):
    """PyTorch Dataset for AUV trajectory data."""

    def __init__(
        self,
        data: np.ndarray,
        dataset_cfg: Dict,
    ):
        validate_dataset_config(dataset_cfg)
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_dataloaders_from_dataset(
    dataset: Dict,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Dict]:
    """Create train/test DataLoaders from an in-memory dataset payload."""
    train_blocks = dataset.get("train_blocks", dataset["train_data"])
    test_blocks = dataset.get("test_blocks", dataset["test_data"])
    t_eval = torch.tensor(dataset["t_eval"], dtype=torch.float32)

    cfg = dataset["config"]
    validate_dataset_config(cfg)

    train_loader = DataLoader(
        AUVDataset(train_blocks, dataset_cfg=cfg),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        AUVDataset(test_blocks, dataset_cfg=cfg),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader, t_eval, cfg


def create_dataloaders(
    dataset_path: str,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Dict]:
    """Create train/test DataLoaders from a pickled dataset file."""
    dataset = load_dataset(dataset_path)
    return create_dataloaders_from_dataset(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def get_train_blocks(dataset: Dict) -> np.ndarray:
    return dataset.get("train_blocks", dataset["train_data"])


def get_test_blocks(dataset: Dict) -> np.ndarray:
    return dataset.get("test_blocks", dataset["test_data"])


def get_test_trajectories(dataset: Dict) -> Tuple[np.ndarray, list]:
    if "test_trajectories" in dataset:
        trajectories = dataset["test_trajectories"]
        meta = dataset.get("test_meta", [])
        if len(meta) < len(trajectories):
            meta = list(meta) + [{} for _ in range(len(trajectories) - len(meta))]
        return trajectories, meta

    test_blocks = get_test_blocks(dataset)
    trajectories = test_blocks[:, None, :, :]
    meta = [{} for _ in range(len(trajectories))]
    return trajectories, meta


def compute_rotation_matrix(R_flat: torch.Tensor) -> torch.Tensor:
    """Project a flattened 9D vector onto SO(3) via Gram-Schmidt. [*, 9] -> [*, 3, 3]"""
    R = R_flat.view(-1, 3, 3)
    r1 = F.normalize(R[:, 0, :], dim=1)
    r2 = R[:, 1, :] - (r1 * R[:, 1, :]).sum(dim=1, keepdim=True) * r1
    r2 = F.normalize(r2, dim=1)
    r3 = torch.cross(r1, r2, dim=1)
    return torch.stack([r1, r2, r3], dim=1)


def geodesic_distance_so3(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between rotation matrices. [batch, 3, 3] -> [batch]"""
    R_diff = torch.bmm(R1.transpose(1, 2), R2)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos_angle)


def so3_constraint_terms(R_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw SO(3) regularization loss and diagnostic errors."""
    eye = torch.eye(3, device=R_raw.device, dtype=R_raw.dtype).unsqueeze(0)
    orth_residual = torch.bmm(R_raw, R_raw.transpose(1, 2)) - eye
    det_residual = torch.det(R_raw) - 1.0

    orth_reg = orth_residual.pow(2).sum(dim=(1, 2)).mean()
    det_reg = det_residual.pow(2).mean()
    reg = orth_reg + det_reg

    orth_error = orth_residual.norm(dim=(1, 2)).mean()
    det_error = det_residual.abs().mean()
    return reg, orth_error, det_error


def se3_trajectory_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    normalizer: StateNormalizer,
    target_ode: Optional[torch.Tensor] = None,
    pred_ode: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
    so3_regularization_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    SE(3) trajectory loss with per-component normalization.

    Args:
        target, pred: [batch, T, state_dim]
        normalizer:   StateNormalizer with per-component std
        target_ode, pred_ode: optional trajectories used for velocity loss
        weights:      optional weights (keys: position, rotation, actuator, u, v, w, p, q, r)
        so3_regularization_weight: weight for raw SO(3) constraint regularization

    Returns:
        total_loss, components dict
    """
    w = weights or {}

    def wt(k):
        return w.get(k, 1.0)

    dx = (pred[..., :3] - target[..., :3]) / normalizer.std_pos
    loss_pos = (dx ** 2).mean()

    R_t = target[..., 3:12].reshape(-1, 3, 3)
    R_raw = pred[..., 3:12].reshape(-1, 3, 3)
    R_p = compute_rotation_matrix(pred[..., 3:12].reshape(-1, 9))
    loss_rot = geodesic_distance_so3(R_t, R_p).pow(2).mean()

    so3_reg, so3_orth, so3_det = so3_constraint_terms(R_raw)

    vel_target = target_ode if target_ode is not None else target
    vel_pred = pred_ode if pred_ode is not None else pred
    dv = (vel_pred[..., 12:18] - vel_target[..., 12:18]) / normalizer.std_vel
    vel_mse = (dv ** 2).mean(dim=(0, 1))

    loss_act = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if normalizer.std_act is not None:
        u_dim = int(normalizer.std_act.numel())
        u_act_slice = slice(18, 18 + u_dim)
        du_act = (
            pred[..., u_act_slice] - target[..., u_act_slice]
        ) / normalizer.std_act.view(1, 1, -1)
        loss_act = (du_act ** 2).mean()

    comp_names = ["u", "v", "w", "p", "q", "r"]
    vel_losses = {name: vel_mse[i] for i, name in enumerate(comp_names)}

    total = (wt("position") * loss_pos
             + wt("rotation") * loss_rot
             + wt("actuator") * loss_act
             + sum(wt(k) * vel_losses[k] for k in comp_names)
             + so3_regularization_weight * so3_reg)

    components = {
        "total": total,
        "position": loss_pos,
        "rotation": loss_rot,
        "actuator": loss_act,
        "velocity": vel_mse[:3].mean(),
        "angular": vel_mse[3:].mean(),
        "so3_reg": so3_reg,
        "so3_orth": so3_orth,
        "so3_det": so3_det,
        **vel_losses,
    }
    return total, components


class TrainingLogger:
    """Tracks per-epoch metrics and writes to disk."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, list] = defaultdict(list)

    def log(self, epoch: int, train_m: Dict, test_m: Dict,
            lr: float, epoch_time: float):
        self.history["epoch"].append(epoch)

        for k, v in train_m.items():
            self.history[f"train_{k}"].append(
                v.item() if isinstance(v, torch.Tensor) else v)
        for k, v in test_m.items():
            self.history[f"test_{k}"].append(
                v.item() if isinstance(v, torch.Tensor) else v)

        self.history["lr"].append(lr)
        self.history["epoch_time"].append(epoch_time)

    def save(self, filename: str = "training_history.pkl"):
        with open(self.save_dir / filename, "wb") as f:
            pickle.dump(dict(self.history), f)

    def get_summary(self) -> Dict:
        tl = self.history.get("test_total", [])
        if not tl:
            return {
                "best_test_loss": None,
                "best_epoch": None,
                "total_time": sum(self.history.get("epoch_time", [])),
            }
        best_idx = int(np.argmin(tl))
        epochs = self.history.get("epoch", [])
        return {
            "best_test_loss": min(tl),
            "best_epoch": int(epochs[best_idx]) if best_idx < len(epochs) else best_idx + 1,
            "total_time": sum(self.history["epoch_time"]),
        }


def save_checkpoint(model, optimizer, epoch, loss, config,
                    filepath, normalizer=None):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config.to_dict(),
    }
    if normalizer is not None:
        ckpt["normalizer"] = normalizer.to_dict()
    torch.save(ckpt, filepath)


def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    # Training checkpoints include config/optimizer metadata, not just tensors.
    ckpt = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, ckpt["epoch"], ckpt["loss"]


def _basic_stats(values):
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _safe_mean(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _evaluate_block_batch_with_fallback(
    model: nn.Module,
    batch_np: np.ndarray,
    t_eval: torch.Tensor,
    device: torch.device,
    ode_solver: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, str]]:
    """Evaluate a batch of independent blocks while isolating failing samples."""
    from torchdiffeq import odeint

    batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
    batch_size = int(batch.shape[0])
    if batch_size == 0:
        return {}, {}

    y0 = model.to_ode_state(batch[:, 0])
    model.reset_nfe()
    try:
        pred = odeint(model, y0, t_eval.to(device), method=ode_solver)
    except (ValueError, RuntimeError):
        if batch_size == 1:
            return {}, {0: "solver_failure"}
        split = batch_size // 2
        left_pred, left_fail = _evaluate_block_batch_with_fallback(
            model, batch_np[:split], t_eval, device, ode_solver)
        right_pred, right_fail = _evaluate_block_batch_with_fallback(
            model, batch_np[split:], t_eval, device, ode_solver)
        merged_pred = {idx: value for idx, value in left_pred.items()}
        merged_pred.update({split + idx: value for idx, value in right_pred.items()})
        merged_fail = {idx: value for idx, value in left_fail.items()}
        merged_fail.update({split + idx: value for idx, value in right_fail.items()})
        return merged_pred, merged_fail

    pred = pred.permute(1, 0, 2)
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        if batch_size == 1:
            return {}, {0: "invalid_prediction"}
        split = batch_size // 2
        left_pred, left_fail = _evaluate_block_batch_with_fallback(
            model, batch_np[:split], t_eval, device, ode_solver)
        right_pred, right_fail = _evaluate_block_batch_with_fallback(
            model, batch_np[split:], t_eval, device, ode_solver)
        merged_pred = {idx: value for idx, value in left_pred.items()}
        merged_pred.update({split + idx: value for idx, value in right_pred.items()})
        merged_fail = {idx: value for idx, value in left_fail.items()}
        merged_fail.update({split + idx: value for idx, value in right_fail.items()})
        return merged_pred, merged_fail

    pred = model.to_data_state(pred)
    return {idx: pred[idx:idx + 1].clone() for idx in range(batch_size)}, {}


@torch.no_grad()
def evaluate_heldout_trajectories(
    model: nn.Module,
    dataset: Dict,
    t_eval: torch.Tensor,
    device: torch.device,
    ode_solver: str = "rk4",
    max_trajectories: Optional[int] = None,
) -> Dict:
    """Evaluate short-horizon prediction over held-out trajectories.

    Metrics are reported in data space after converting predictions back from
    the model's internal ODE convention. This keeps held-out evaluation
    comparable across models with different latent velocity conventions.
    """
    model.eval()

    trajectories, meta_list = get_test_trajectories(dataset)
    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]
        meta_list = meta_list[:max_trajectories]

    trajectory_rows = []
    scenario_groups = defaultdict(list)
    trajectory_failures = defaultdict(int)
    block_failures = defaultdict(int)
    for traj_idx, (traj_blocks, meta) in enumerate(zip(trajectories, meta_list)):
        scenario = meta.get("scenario", "UNKNOWN")
        seed = meta.get("seed", traj_idx)

        pred_map, failure_map = _evaluate_block_batch_with_fallback(
            model,
            traj_blocks,
            t_eval,
            device,
            ode_solver,
        )
        successful_indices = sorted(pred_map)
        failed_indices = sorted(failure_map)
        successful_blocks = len(successful_indices)
        n_blocks = int(len(traj_blocks))
        for reason in failure_map.values():
            block_failures[reason] += 1

        if successful_blocks == n_blocks:
            status = "ok"
        elif successful_blocks == 0:
            unique_reasons = sorted(set(failure_map.values()))
            status = unique_reasons[0] if len(unique_reasons) == 1 else "all_blocks_failed"
        else:
            status = "partial_failure"

        row = {
            "trajectory_index": int(traj_idx),
            "seed": int(seed),
            "scenario": scenario,
            "n_blocks": n_blocks,
            "successful_blocks": successful_blocks,
            "failed_blocks": len(failed_indices),
            "success_rate": (successful_blocks / n_blocks) if n_blocks else 0.0,
            "status": status,
            "block_failure_summary": (
                ",".join(f"{idx}:{failure_map[idx]}" for idx in failed_indices)
                if failed_indices else ""
            ),
        }

        if successful_blocks == 0:
            trajectory_failures[status] += 1
            row.update(
                {
                    "position_rmse": float("nan"),
                    "rotation_geodesic": float("nan"),
                    "velocity_rmse": float("nan"),
                    "angular_rmse": float("nan"),
                    "u_rmse": float("nan"),
                    "v_rmse": float("nan"),
                    "w_rmse": float("nan"),
                    "p_rmse": float("nan"),
                    "q_rmse": float("nan"),
                    "r_rmse": float("nan"),
                    "so3_det_mean": float("nan"),
                    "so3_orth_mean": float("nan"),
                }
            )
            trajectory_rows.append(row)
            scenario_groups[scenario].append(row)
            continue

        if successful_blocks < n_blocks:
            trajectory_failures[status] += 1

        pred = torch.cat([pred_map[idx] for idx in successful_indices], dim=0)
        target = torch.tensor(traj_blocks[successful_indices], dtype=torch.float32, device=device)
        pos_rmse = ((target[..., :3] - pred[..., :3]) ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy()

        R_t = target[..., 3:12].reshape(-1, 3, 3)
        R_p = compute_rotation_matrix(pred[..., 3:12].reshape(-1, 9))
        rot_geo = geodesic_distance_so3(R_t, R_p).view(pred.shape[0], pred.shape[1]).mean(dim=1).cpu().numpy()

        vel_rmse = ((target[..., 12:15] - pred[..., 12:15]) ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy()
        ang_rmse = ((target[..., 15:18] - pred[..., 15:18]) ** 2).mean(dim=(1, 2)).sqrt().cpu().numpy()

        component_rmse = {}
        for idx_name, name in enumerate(["u", "v", "w", "p", "q", "r"]):
            offset = 12 + idx_name
            component_rmse[name] = (
                ((target[..., offset] - pred[..., offset]) ** 2).mean(dim=1).sqrt().cpu().numpy()
            )

        R_raw = pred[..., 3:12].reshape(-1, 3, 3)
        det_e = (torch.det(R_raw) - 1.0).abs().view(pred.shape[0], pred.shape[1]).mean(dim=1).cpu().numpy()
        orth_e = (
            (R_raw @ R_raw.transpose(1, 2) - torch.eye(3, device=device))
            .norm(dim=(1, 2))
            .view(pred.shape[0], pred.shape[1])
            .mean(dim=1)
            .cpu()
            .numpy()
        )

        row.update(
            {
                "position_rmse": float(np.mean(pos_rmse)),
                "rotation_geodesic": float(np.mean(rot_geo)),
                "velocity_rmse": float(np.mean(vel_rmse)),
                "angular_rmse": float(np.mean(ang_rmse)),
                "u_rmse": float(np.mean(component_rmse["u"])),
                "v_rmse": float(np.mean(component_rmse["v"])),
                "w_rmse": float(np.mean(component_rmse["w"])),
                "p_rmse": float(np.mean(component_rmse["p"])),
                "q_rmse": float(np.mean(component_rmse["q"])),
                "r_rmse": float(np.mean(component_rmse["r"])),
                "so3_det_mean": float(np.mean(det_e)),
                "so3_orth_mean": float(np.mean(orth_e)),
            }
        )

        trajectory_rows.append(row)
        scenario_groups[scenario].append(row)

    def summarize_rows(rows):
        if not rows:
            return {
                "n_trajectories": 0,
                "success_rate": float("nan"),
                "position_rmse": _basic_stats([]),
                "rotation_geodesic": _basic_stats([]),
                "velocity_rmse": _basic_stats([]),
                "angular_rmse": _basic_stats([]),
                "so3_violation": {"det_mean": float("nan"), "orth_mean": float("nan")},
                "component_rmse": {k: _basic_stats([]) for k in ["u", "v", "w", "p", "q", "r"]},
            }

        success_rate = _safe_mean([row["success_rate"] for row in rows])
        summary = {
            "n_trajectories": len(rows),
            "success_rate": success_rate,
            "position_rmse": _basic_stats([row["position_rmse"] for row in rows]),
            "rotation_geodesic": _basic_stats([row["rotation_geodesic"] for row in rows]),
            "velocity_rmse": _basic_stats([row["velocity_rmse"] for row in rows]),
            "angular_rmse": _basic_stats([row["angular_rmse"] for row in rows]),
            "so3_violation": {
                "det_mean": _safe_mean([row["so3_det_mean"] for row in rows]),
                "orth_mean": _safe_mean([row["so3_orth_mean"] for row in rows]),
            },
            "component_rmse": {
                k: _basic_stats([row[f"{k}_rmse"] for row in rows])
                for k in ["u", "v", "w", "p", "q", "r"]
            },
        }
        return summary

    return {
        "overall": summarize_rows(trajectory_rows),
        "by_scenario": {
            scenario: summarize_rows(rows) for scenario, rows in sorted(scenario_groups.items())
        },
        "failure_counts": dict(trajectory_failures),
        "block_failure_counts": dict(block_failures),
        "trajectory_rows": trajectory_rows,
    }


@torch.no_grad()
def evaluate_trajectory_prediction(
    model: nn.Module,
    test_loader: DataLoader,
    t_eval: torch.Tensor,
    device: torch.device,
    ode_solver: str = "rk4",
    n_samples: int = 100
) -> Dict:
    """Compute per-trajectory RMSE, geodesic error, and SO(3) violation."""
    model.eval()
    t_eval = t_eval.to(device)

    pos_err, rot_err, vel_err, ang_err = [], [], [], []
    comp_err = {k: [] for k in ["u", "v", "w", "p", "q", "r"]}
    so3_viol = []

    solver_failed_batches = 0
    invalid_prediction_batches = 0
    count = 0

    for batch in test_loader:
        if count >= n_samples:
            break

        batch_np = batch.cpu().numpy()
        pred_map, failure_map = _evaluate_block_batch_with_fallback(
            model,
            batch_np,
            t_eval,
            device,
            ode_solver,
        )
        solver_failed_batches += sum(reason == "solver_failure" for reason in failure_map.values())
        invalid_prediction_batches += sum(
            reason == "invalid_prediction" for reason in failure_map.values()
        )

        for sample_idx in sorted(pred_map):
            if count >= n_samples:
                break

            tgt = batch[sample_idx].to(device)
            prd = pred_map[sample_idx][0]

            pos_err.append(((tgt[:, :3] - prd[:, :3]) ** 2).mean().sqrt().item())

            R_t = tgt[:, 3:12].view(-1, 3, 3)
            R_p = compute_rotation_matrix(prd[:, 3:12].reshape(-1, 9))
            rot_err.append(geodesic_distance_so3(R_t, R_p).mean().item())

            vel_err.append(((tgt[:, 12:15] - prd[:, 12:15]) ** 2).mean().sqrt().item())
            ang_err.append(((tgt[:, 15:18] - prd[:, 15:18]) ** 2).mean().sqrt().item())

            for j, k in enumerate(["u", "v", "w"]):
                comp_err[k].append(((tgt[:, 12 + j] - prd[:, 12 + j]) ** 2).mean().sqrt().item())
            for j, k in enumerate(["p", "q", "r"]):
                comp_err[k].append(((tgt[:, 15 + j] - prd[:, 15 + j]) ** 2).mean().sqrt().item())

            R_raw = prd[:, 3:12].view(-1, 3, 3)
            det_e = (torch.det(R_raw) - 1).abs().mean().item()
            orth_e = (R_raw @ R_raw.transpose(1, 2)
                      - torch.eye(3, device=device)).norm(dim=(1, 2)).mean().item()
            so3_viol.append({"det": det_e, "orth": orth_e})

            count += 1

    def stats(values):
        if not values:
            return {"mean": float("nan"), "std": float("nan"), "max": float("nan")}
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "max": float(np.max(arr))}

    component_stats = {}
    for key, values in comp_err.items():
        if values:
            arr = np.asarray(values, dtype=np.float64)
            component_stats[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        else:
            component_stats[key] = {"mean": float("nan"), "std": float("nan")}

    if so3_viol:
        det_mean = float(np.mean([v["det"] for v in so3_viol]))
        orth_mean = float(np.mean([v["orth"] for v in so3_viol]))
    else:
        det_mean = float("nan")
        orth_mean = float("nan")

    return {
        "position_rmse": stats(pos_err),
        "rotation_geodesic": stats(rot_err),
        "velocity_rmse": stats(vel_err),
        "angular_rmse": stats(ang_err),
        "component_rmse": component_stats,
        "so3_violation": {
            "det_mean": det_mean,
            "orth_mean": orth_mean,
        },
        "solver_failed_batches": solver_failed_batches,
        "invalid_prediction_batches": invalid_prediction_batches,
        "n_samples": count,
    }


def print_evaluation_results(results: Dict,
                             logger: Optional[logging.Logger] = None):
    """Pretty-print evaluation results."""
    log = logger.info if logger else print

    log("=" * 50)
    log("Evaluation Results")
    log("=" * 50)

    pr = results["position_rmse"]
    log(f"Position RMSE [m]:   {pr['mean']:.4f} +/- {pr['std']:.4f}  (max {pr['max']:.4f})")

    rg = results["rotation_geodesic"]
    log(f"Rotation geo [rad]:  {rg['mean']:.4f} +/- {rg['std']:.4f}  (max {rg['max']:.4f})")

    vr = results["velocity_rmse"]
    log(f"Lin vel RMSE [m/s]:  {vr['mean']:.4f} +/- {vr['std']:.4f}")

    ar = results["angular_rmse"]
    log(f"Ang vel RMSE [r/s]:  {ar['mean']:.4f} +/- {ar['std']:.4f}")

    cr = results["component_rmse"]
    log(f"Components:  u={cr['u']['mean']:.4f}  v={cr['v']['mean']:.4f}  w={cr['w']['mean']:.4f}  "
        f"p={cr['p']['mean']:.4f}  q={cr['q']['mean']:.4f}  r={cr['r']['mean']:.4f}")

    sv = results["so3_violation"]
    log(f"SO(3) violation:  |det-1|={sv['det_mean']:.2e}  ||RR^T-I||={sv['orth_mean']:.2e}")
    log(f"Solver failures: {results['solver_failed_batches']}  "
        f"Invalid predictions: {results['invalid_prediction_batches']}")
    log(f"Evaluated on {results['n_samples']} samples")


def print_heldout_evaluation_results(results: Dict,
                                     logger: Optional[logging.Logger] = None):
    """Pretty-print held-out trajectory evaluation results."""
    log = logger.info if logger else print

    log("=" * 50)
    log("Held-Out Trajectory Evaluation")
    log("=" * 50)

    overall = results["overall"]
    log(f"Trajectories: {overall['n_trajectories']}  Success rate: {overall['success_rate']:.4f}")
    log(
        f"Position RMSE [m]:   {overall['position_rmse']['mean']:.4f} "
        f"(median {overall['position_rmse']['median']:.4f}, p95 {overall['position_rmse']['p95']:.4f})"
    )
    log(
        f"Rotation geo [rad]:  {overall['rotation_geodesic']['mean']:.4f} "
        f"(median {overall['rotation_geodesic']['median']:.4f}, p95 {overall['rotation_geodesic']['p95']:.4f})"
    )
    log(
        f"Lin vel RMSE [m/s]:  {overall['velocity_rmse']['mean']:.4f}  "
        f"Ang vel RMSE [r/s]:  {overall['angular_rmse']['mean']:.4f}"
    )
    sv = overall["so3_violation"]
    log(f"SO(3) violation:  |det-1|={sv['det_mean']:.2e}  ||RR^T-I||={sv['orth_mean']:.2e}")
    if results["failure_counts"]:
        failures = ", ".join(f"{k}={v}" for k, v in sorted(results["failure_counts"].items()))
        log(f"Trajectory failures: {failures}")
    if results.get("block_failure_counts"):
        failures = ", ".join(
            f"{k}={v}" for k, v in sorted(results["block_failure_counts"].items())
        )
        log(f"Block failures: {failures}")

    for scenario, summary in results["by_scenario"].items():
        log(
            f"[{scenario}] pos={summary['position_rmse']['mean']:.4f} m, "
            f"rot={summary['rotation_geodesic']['mean']:.4f} rad, "
            f"success={summary['success_rate']:.4f}"
        )


def save_heldout_evaluation_results(results: Dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "heldout_evaluation.json", "w") as f:
        json.dump(
            {
                "overall": results["overall"],
                "by_scenario": results["by_scenario"],
                "failure_counts": results["failure_counts"],
                "block_failure_counts": results.get("block_failure_counts", {}),
            },
            f,
            indent=2,
        )

    rows = results["trajectory_rows"]
    if rows:
        with open(output_dir / "heldout_trajectory_metrics.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = []
    overall = results["overall"]
    lines.append("Held-Out Trajectory Evaluation")
    lines.append("=" * 80)
    lines.append(
        f"Trajectories: {overall['n_trajectories']}  Success rate: {overall['success_rate']:.4f}"
    )
    lines.append(
        f"Position RMSE [m]: mean={overall['position_rmse']['mean']:.4f}, "
        f"median={overall['position_rmse']['median']:.4f}, "
        f"p95={overall['position_rmse']['p95']:.4f}"
    )
    lines.append(
        f"Rotation geo [rad]: mean={overall['rotation_geodesic']['mean']:.4f}, "
        f"median={overall['rotation_geodesic']['median']:.4f}, "
        f"p95={overall['rotation_geodesic']['p95']:.4f}"
    )
    lines.append(
        f"Lin vel RMSE [m/s]: {overall['velocity_rmse']['mean']:.4f}  "
        f"Ang vel RMSE [r/s]: {overall['angular_rmse']['mean']:.4f}"
    )
    lines.append(
        f"SO(3) violation: |det-1|={overall['so3_violation']['det_mean']:.2e}  "
        f"||RR^T-I||={overall['so3_violation']['orth_mean']:.2e}"
    )
    if results["failure_counts"]:
        failure_text = ", ".join(f"{k}={v}" for k, v in sorted(results["failure_counts"].items()))
        lines.append(f"Trajectory failures: {failure_text}")
    if results.get("block_failure_counts"):
        failure_text = ", ".join(
            f"{k}={v}" for k, v in sorted(results["block_failure_counts"].items())
        )
        lines.append(f"Block failures: {failure_text}")

    lines.append("")
    lines.append("By Scenario")
    lines.append("-" * 80)
    for scenario, summary in results["by_scenario"].items():
        lines.append(
            f"{scenario}: pos={summary['position_rmse']['mean']:.4f} m, "
            f"rot={summary['rotation_geodesic']['mean']:.4f} rad, "
            f"vel={summary['velocity_rmse']['mean']:.4f} m/s, "
            f"ang={summary['angular_rmse']['mean']:.4f} rad/s, "
            f"success={summary['success_rate']:.4f}"
        )

    with open(output_dir / "heldout_evaluation.txt", "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def save_block_evaluation_results(results: Dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "block_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    lines = []
    lines.append("Block-Level Evaluation")
    lines.append("=" * 80)
    lines.append(
        f"Position RMSE [m]: mean={results['position_rmse']['mean']:.4f}, "
        f"std={results['position_rmse']['std']:.4f}, max={results['position_rmse']['max']:.4f}"
    )
    lines.append(
        f"Rotation geo [rad]: mean={results['rotation_geodesic']['mean']:.4f}, "
        f"std={results['rotation_geodesic']['std']:.4f}, max={results['rotation_geodesic']['max']:.4f}"
    )
    lines.append(
        f"Lin vel RMSE [m/s]: mean={results['velocity_rmse']['mean']:.4f}, "
        f"std={results['velocity_rmse']['std']:.4f}"
    )
    lines.append(
        f"Ang vel RMSE [r/s]: mean={results['angular_rmse']['mean']:.4f}, "
        f"std={results['angular_rmse']['std']:.4f}"
    )
    sv = results["so3_violation"]
    lines.append(
        f"SO(3) violation: |det-1|={sv['det_mean']:.2e}  ||RR^T-I||={sv['orth_mean']:.2e}"
    )
    lines.append(
        f"Solver failures: {results['solver_failed_batches']}  "
        f"Invalid predictions: {results['invalid_prediction_batches']}"
    )
    lines.append(f"Evaluated samples: {results['n_samples']}")

    with open(output_dir / "block_evaluation.txt", "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
