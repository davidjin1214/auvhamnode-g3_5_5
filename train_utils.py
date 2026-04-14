"""
Training utilities for AUVHamNODE.

Provides: configuration, data loading, SE(3) loss functions, logging,
checkpointing, and evaluation metrics.
"""

import dataclasses
from copy import deepcopy
import math

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


@dataclass
class NoiseConfig:
    """Noise configuration aligned with the current IC-only training interface."""

    profile: str = "clean"      # clean / nominal_train / nominal_eval / degraded_eval / heading_biased_eval / current_bias_eval
    reference: str = "remus100_dr"  # remus100_dr / remus100_ins
    scale: float = 1.0          # global magnitude multiplier
    warmup_epochs: int = 20     # fully clean warmup before noisy IC regularization
    ramp_epochs: int = 80       # linear ramp after warmup
    mix_ratio: float = 0.5      # fraction of training samples using noisy IC
    linear_floor_std: float = 0.005   # legacy compatibility field, not used by v3 budgets
    angular_floor_std: float = 0.0015 # legacy compatibility field, not used by v3 budgets

    @property
    def is_active(self) -> bool:
        return self.profile != "clean"

    def epoch_scale(self, epoch: int) -> float:
        if not self.is_active:
            return 0.0
        if epoch <= self.warmup_epochs:
            return 0.0
        ramp_progress = (epoch - self.warmup_epochs) / max(self.ramp_epochs, 1)
        return min(max(ramp_progress, 0.0), 1.0) * self.scale


AVAILABLE_NOISE_PROFILES = (
    "clean",
    "nominal_eval",
    "degraded_eval",
    "heading_biased_eval",
    "current_bias_eval",
)

AVAILABLE_NOISE_REFERENCES = (
    "remus100_dr",
    "remus100_ins",
)


def _base_noise_profile(profile: str) -> str:
    return {
        "heading_biased_eval": "nominal_eval",
        "current_bias_eval": "nominal_eval",
    }.get(profile, profile)


def _resolve_noise_reference(reference: Optional[str]) -> str:
    ref = str(reference or "remus100_dr").strip().lower()
    if ref not in AVAILABLE_NOISE_REFERENCES:
        valid = ", ".join(AVAILABLE_NOISE_REFERENCES)
        raise ValueError(f"Unsupported noise reference {reference!r}. Expected one of: {valid}")
    return ref


def noise_cfg_from_profile(profile_name: str, *, reference: str = "remus100_dr") -> Optional["NoiseConfig"]:
    if profile_name == "clean":
        return None
    valid_profiles = {"nominal_train", *AVAILABLE_NOISE_PROFILES[1:]}
    if profile_name not in valid_profiles:
        valid = ", ".join(["clean", *sorted(valid_profiles)])
        raise ValueError(f"Unsupported noise profile {profile_name!r}. Expected one of: {valid}")
    return NoiseConfig(
        profile=profile_name,
        reference=_resolve_noise_reference(reference),
        warmup_epochs=0,
        ramp_epochs=1,
    )


def resolve_noise_profiles(values, default_profiles=None, allow_none=False, available_profiles=None):
    available_profiles = tuple(available_profiles or AVAILABLE_NOISE_PROFILES)
    if not values:
        return list(default_profiles) if default_profiles else ["clean"]
    resolved = []
    for value in values:
        key = str(value).strip().lower()
        if key == "all":
            for profile in available_profiles:
                if profile not in resolved:
                    resolved.append(profile)
            continue
        if key == "none" and allow_none:
            return []
        if key not in available_profiles:
            extras = ["all", "none"] if allow_none else ["all"]
            valid = ", ".join([*available_profiles, *extras])
            raise ValueError(f"Unsupported noise profile {value!r}. Expected one of: {valid}")
        if key not in resolved:
            resolved.append(key)
    return resolved


def available_eval_noise_profiles(ocean_current: bool) -> List[str]:
    return (
        ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval", "current_bias_eval"]
        if ocean_current else
        ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval"]
    )


def default_eval_noise_profiles(
    *,
    ocean_current: bool,
    noise_reference: str,
    phase: str,
) -> List[str]:
    reference = _resolve_noise_reference(noise_reference)
    if phase not in {"block", "heldout"}:
        raise ValueError(f"Unsupported evaluation phase: {phase}")

    if not ocean_current:
        return (
            ["clean", "nominal_eval"]
            if phase == "block" else
            ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval"]
        )

    if reference == "remus100_ins":
        return (
            ["clean", "nominal_eval", "heading_biased_eval", "current_bias_eval"]
            if phase == "block" else
            ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval", "current_bias_eval"]
        )

    return (
        ["clean", "nominal_eval", "heading_biased_eval"]
        if phase == "block" else
        ["clean", "nominal_eval", "degraded_eval", "heading_biased_eval"]
    )


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

    # Training-time noise injection. ``noise_profile`` is the preferred
    # interface. The remaining noise_* fields are retained for backward
    # compatibility with older configs and scripts.
    noise_profile: Optional[str] = None
    noise_reference: str = "remus100_dr"
    noise_level: int = 0              # legacy alias: 0=clean 1=nominal_train 2=nominal_eval 3=degraded_eval
    noise_scale: float = 1.0          # global noise magnitude multiplier
    noise_ramp_epochs: int = 100      # curriculum ramp: 0 → full over N epochs
    noise_warmup_epochs: int = 20     # clean warmup before enabling noisy IC
    noise_mix_ratio: float = 0.5      # fraction of samples using noisy IC
    noise_linear_floor_std: float = 0.005   # m/s
    noise_angular_floor_std: float = 0.0015 # rad/s
    # Legacy fields kept so old configs still load. The IC-only profile path no
    # longer consumes these values directly.
    noise_vel_lin_std: float = 0.02
    noise_vel_ang_std: float = 0.005
    noise_rot_init_std: float = 0.005
    noise_act_noise_std: float = 0.005
    noise_current_std: float = 0.05
    noise_ar1_corr: float = 0.85
    noise_bias_ratio: float = 0.25
    noise_dropout_prob: float = 0.12
    noise_walk_ratio: float = 0.05
    noise_mult_coeff: float = 0.003

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

    # Legacy checkpoint-only fields from the original observation/init-noise
    # interface. They are ignored on load because the current IC-noise path
    # no longer consumes them, but older checkpoint configs may still store
    # them in config.json.
    _IGNORED_LEGACY_FIELDS = {
        "actuator_noise_std",
        "current_noise_std",
        "init_rot_std",
        "init_state_noise",
        "observation_bias",
        "observation_noise",
        "observation_noise_scale",
        "velocity_bias_std",
        "velocity_noise_std",
    }

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
        profile = self.resolved_noise_profile()
        if profile != "clean":
            tags.append(f"noise-{profile}")
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
        ignored_legacy = set(getattr(cls, "_IGNORED_LEGACY_FIELDS", ()))
        unknown = sorted(set(data) - valid - ignored_legacy)
        if unknown:
            raise ValueError(
                "Unknown TrainConfig fields: "
                + ", ".join(unknown)
            )
        payload = {key: value for key, value in data.items() if key in valid}
        return cls(**payload)

    def get_noise_config(self) -> "NoiseConfig":
        """Extract a NoiseConfig from this TrainConfig's noise_* fields."""
        profile = self.resolved_noise_profile()
        return NoiseConfig(
            profile=profile,
            reference=_resolve_noise_reference(self.noise_reference),
            scale=self.noise_scale,
            warmup_epochs=self.noise_warmup_epochs,
            ramp_epochs=self.noise_ramp_epochs,
            mix_ratio=self.noise_mix_ratio,
            linear_floor_std=self.noise_linear_floor_std,
            angular_floor_std=self.noise_angular_floor_std,
        )

    def resolved_noise_profile(self) -> str:
        if self.noise_profile is not None:
            return self.noise_profile
        legacy_map = {
            0: "clean",
            1: "nominal_train",
            2: "nominal_eval",
            3: "degraded_eval",
        }
        return legacy_map.get(int(self.noise_level), "clean")


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


def _project_to_so3(R: torch.Tensor) -> torch.Tensor:
    """Project batched [..., 3, 3] matrices onto SO(3) with det = +1."""
    orig_shape = R.shape
    flat = R.reshape(-1, 3, 3)
    U, _, Vh = torch.linalg.svd(flat)
    det = torch.det(torch.matmul(U, Vh))
    sign = torch.ones(flat.shape[0], 3, dtype=flat.dtype, device=flat.device)
    sign[:, -1] = det.sign()
    return torch.matmul(U * sign.unsqueeze(-2), Vh).reshape(orig_shape)


def _so3_exp_map(delta_theta: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) exponential map for [..., 3] rotation vectors."""
    return torch.linalg.matrix_exp(_skew_symmetric_sequence(delta_theta))


def _dvl_dropout_freeze(
    delta_nu: torch.Tensor,
    prob: float,
) -> torch.Tensor:
    """Simulate DVL bottom-lock loss by freezing velocity noise at the last
    valid step.  When the DVL loses lock the navigation filter holds the most
    recent good velocity measurement; we model this by repeating the noise
    sample from the previous valid step rather than injecting a fresh one.
    """
    T = delta_nu.shape[1]
    valid = torch.rand(delta_nu.shape[0], T, device=delta_nu.device) > prob
    valid[:, 0] = True   # first step is always valid
    for t in range(1, T):
        delta_nu[:, t] = torch.where(valid[:, t, None], delta_nu[:, t], delta_nu[:, t - 1])
    return delta_nu


def _profile_velocity_spec(
    profile: str,
    reference: str,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    profile = _base_noise_profile(profile)
    _resolve_noise_reference(reference)
    table = {
        "nominal_train": {
            "linear_floor": [0.002, 0.002, 0.003],
            "linear_ratio": [0.002, 0.002, 0.002],
            "angular_std": [0.0006, 0.0006, 0.0010],
        },
        "nominal_eval": {
            "linear_floor": [0.003, 0.003, 0.004],
            "linear_ratio": [0.003, 0.003, 0.003],
            "angular_std": [0.0008, 0.0008, 0.0015],
        },
        "degraded_eval": {
            "linear_floor": [0.006, 0.006, 0.008],
            "linear_ratio": [0.006, 0.006, 0.006],
            "angular_std": [0.0015, 0.0015, 0.0030],
        },
    }
    payload = table.get(profile)
    if payload is None:
        zeros = torch.zeros(3, dtype=dtype, device=device)
        return {
            "linear_floor": zeros,
            "linear_ratio": zeros,
            "angular_std": zeros,
        }
    return {
        key: torch.tensor(values, dtype=dtype, device=device)
        for key, values in payload.items()
    }


def _profile_velocity_std_from_state(
    nu_r_state: torch.Tensor,
    profile: str,
    reference: str,
) -> torch.Tensor:
    spec = _profile_velocity_spec(
        profile,
        reference,
        dtype=nu_r_state.dtype,
        device=nu_r_state.device,
    )
    linear_speed = nu_r_state[:, :3].abs()
    linear_floor = spec["linear_floor"].view(1, 3)
    linear_ratio = spec["linear_ratio"].view(1, 3)
    linear_std = torch.sqrt(linear_floor ** 2 + (linear_ratio * linear_speed) ** 2)
    angular_std = spec["angular_std"].view(1, 3).expand(nu_r_state.shape[0], 3)
    return torch.cat([linear_std, angular_std], dim=1)


def _profile_rotation_std_vector(
    profile: str,
    reference: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    profile = _base_noise_profile(profile)
    reference = _resolve_noise_reference(reference)
    table = {
        "remus100_dr": {
            "nominal_train": [0.0015, 0.0015, 0.0060],
            "nominal_eval": [0.0020, 0.0020, 0.0100],
            "degraded_eval": [0.0040, 0.0040, 0.0250],
        },
        "remus100_ins": {
            "nominal_train": [0.0008, 0.0008, 0.0030],
            "nominal_eval": [0.0010, 0.0010, 0.0040],
            "degraded_eval": [0.0020, 0.0020, 0.0100],
        },
    }
    values = table.get(reference, {}).get(profile)
    if values is None:
        return torch.zeros(3, dtype=dtype, device=device)
    return torch.tensor(values, dtype=dtype, device=device)


def _profile_rotation_bias_vector(
    profile: str,
    reference: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    reference = _resolve_noise_reference(reference)
    values = {
        "remus100_dr": {
            "heading_biased_eval": [0.0, 0.0, 0.0350],
        },
        "remus100_ins": {
            "heading_biased_eval": [0.0, 0.0, 0.0200],
        },
    }.get(reference, {}).get(profile, [0.0, 0.0, 0.0])
    return torch.tensor(values, dtype=dtype, device=device)


def _profile_current_bias_vector(
    profile: str,
    reference: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    reference = _resolve_noise_reference(reference)
    table = {
        "remus100_dr": {
            "current_bias_eval": [0.0500, 0.0500, 0.0200],
        },
        "remus100_ins": {
            "current_bias_eval": [0.0500, 0.0500, 0.0200],
        },
    }
    values = table.get(reference, {}).get(profile, [0.0, 0.0, 0.0])
    return torch.tensor(values, dtype=dtype, device=device)


def _profile_current_std(
    profile: str,
    reference: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    profile = _base_noise_profile(profile)
    reference = _resolve_noise_reference(reference)
    table = {
        "remus100_dr": {
            "nominal_train": [0.0, 0.0, 0.0],
            "nominal_eval": [0.0, 0.0, 0.0],
            "degraded_eval": [0.0, 0.0, 0.0],
        },
        "remus100_ins": {
            "nominal_train": [0.020, 0.020, 0.008],
            "nominal_eval": [0.030, 0.030, 0.010],
            "degraded_eval": [0.060, 0.060, 0.020],
        },
    }
    values = table.get(reference, {}).get(profile, [0.0, 0.0, 0.0])
    return torch.tensor(values, dtype=dtype, device=device)


def _profile_actuator_std(
    profile: str,
    u_dim: int,
    std_act: Optional[torch.Tensor],
    reference: str,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    profile = _base_noise_profile(profile)
    _resolve_noise_reference(reference)
    if u_dim == 3:
        table = {
            "nominal_train": [0.002, 0.002, 3.0],
            "nominal_eval": [0.003, 0.003, 5.0],
            "degraded_eval": [0.004, 0.004, 10.0],
        }
        values = table.get(profile, [0.0, 0.0, 0.0])
        return torch.tensor(values, dtype=dtype, device=device)

    if std_act is None:
        return torch.zeros(u_dim, dtype=dtype, device=device)

    ratio = {
        "nominal_train": 0.015,
        "nominal_eval": 0.025,
        "degraded_eval": 0.05,
    }.get(profile, 0.0)
    return ratio * std_act.to(device=device, dtype=dtype)


def _sample_sign_pattern(
    batch_size: int,
    dim: int = 1,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_ids: Optional[torch.Tensor] = None,
    base_seed: Optional[int] = None,
    stream: int = 0,
) -> torch.Tensor:
    if base_seed is None or sample_ids is None:
        draws = torch.randint(0, 2, (batch_size, dim), device=device)
        return (draws * 2 - 1).to(dtype=dtype)

    sample_ids = torch.as_tensor(sample_ids, dtype=torch.long).view(-1).cpu().tolist()
    out = torch.empty(batch_size, dim, dtype=dtype, device=device)
    for idx, sample_id in enumerate(sample_ids):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(base_seed) + 1000003 * int(sample_id) + 7919 * int(stream))
        draws = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
        out[idx] = (draws * 2 - 1).to(dtype=torch.float32)
    return out


def _sample_scaled_noise(
    std: torch.Tensor,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sample_ids: Optional[torch.Tensor] = None,
    base_seed: Optional[int] = None,
    stream: int = 0,
) -> torch.Tensor:
    std = std.to(device=device, dtype=dtype)
    if std.ndim == 1:
        dim = int(std.numel())
        std_rows = std.view(1, dim).expand(batch_size, dim)
    elif std.ndim == 2:
        if std.shape[0] != batch_size:
            raise ValueError(
                f"Per-sample std must have batch dimension {batch_size}, got {tuple(std.shape)}."
            )
        dim = int(std.shape[1])
        std_rows = std
    else:
        raise ValueError(f"std must be 1D or 2D, got shape {tuple(std.shape)}.")

    if base_seed is None or sample_ids is None:
        return std_rows * torch.randn(batch_size, dim, dtype=dtype, device=device)

    sample_ids = torch.as_tensor(sample_ids, dtype=torch.long).view(-1).cpu().tolist()
    std_cpu = std_rows.detach().cpu().to(dtype=torch.float32)
    out = torch.empty(batch_size, dim, dtype=dtype, device=device)
    for idx, sample_id in enumerate(sample_ids):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(base_seed) + 1000003 * int(sample_id) + 7919 * int(stream))
        draw = torch.randn(dim, generator=gen, dtype=torch.float32)
        out[idx] = (std_cpu[idx] * draw).to(device=device, dtype=dtype)
    return out


def summarize_noise_budget(
    cfg: Optional[NoiseConfig],
    normalizer: StateNormalizer,
    *,
    u_dim: int,
    ocean_current: bool,
    epoch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Dict:
    profile = "clean" if cfg is None else cfg.profile
    base_profile = _base_noise_profile(profile)
    if cfg is None:
        cfg = NoiseConfig(
            profile="clean",
            reference="remus100_dr",
            scale=0.0,
            warmup_epochs=0,
            ramp_epochs=1,
            mix_ratio=0.0,
        )

    report_epoch = (
        int(epoch)
        if epoch is not None
        else int(cfg.warmup_epochs + cfg.ramp_epochs)
    )
    epoch_scale = float(cfg.epoch_scale(report_epoch))
    dev = torch.device(device)
    reference = _resolve_noise_reference(cfg.reference)

    vel_spec = _profile_velocity_spec(profile, reference, dtype=dtype, device=dev)
    nu_r_ref_speed = torch.tensor([1.0, 1.0, 1.0], dtype=dtype, device=dev)
    nu_r_linear_std = torch.sqrt(
        vel_spec["linear_floor"] ** 2 + (vel_spec["linear_ratio"] * nu_r_ref_speed) ** 2
    )
    nu_r_std = epoch_scale * torch.cat([nu_r_linear_std, vel_spec["angular_std"]], dim=0)

    rot_std = epoch_scale * _profile_rotation_std_vector(
        profile,
        reference,
        dtype=dtype,
        device=dev,
    )
    rot_bias = epoch_scale * _profile_rotation_bias_vector(
        profile,
        reference,
        dtype=dtype,
        device=dev,
    )
    current_bias = (
        epoch_scale * _profile_current_bias_vector(
            profile,
            reference,
            dtype=dtype,
            device=dev,
        )
        if ocean_current
        else None
    )
    act_std = epoch_scale * _profile_actuator_std(
        profile,
        u_dim,
        normalizer.std_act,
        reference,
        dtype=dtype,
        device=dev,
    )
    current_std = (
        epoch_scale * _profile_current_std(
            profile,
            reference,
            dtype=dtype,
            device=dev,
        )
        if ocean_current
        else None
    )

    actuator_labels = (
        ["delta_r", "delta_s", "rpm"]
        if u_dim == 3 else
        [f"a{i}" for i in range(u_dim)]
    )

    return {
        "profile": profile,
        "base_profile": base_profile,
        "reference": reference,
        "is_active": bool(cfg.is_active),
        "epoch": report_epoch,
        "epoch_scale": epoch_scale,
        "global_scale": float(cfg.scale),
        "warmup_epochs": int(cfg.warmup_epochs),
        "ramp_epochs": int(cfg.ramp_epochs),
        "mix_ratio": float(cfg.mix_ratio),
        "nu_r_model": "linear=sqrt(floor^2 + (ratio*|nu_r|)^2); angular=fixed",
        "nu_r_reference_speed_mps": 1.0,
        "nu_r_linear_floor_std": {
            name: float(value)
            for name, value in zip(["u", "v", "w"], vel_spec["linear_floor"].cpu().tolist())
        },
        "nu_r_linear_ratio": {
            name: float(value)
            for name, value in zip(["u", "v", "w"], vel_spec["linear_ratio"].cpu().tolist())
        },
        "nu_r_angular_std": {
            name: float(value)
            for name, value in zip(["p", "q", "r"], vel_spec["angular_std"].cpu().tolist())
        },
        "nu_r_std": {
            name: float(value)
            for name, value in zip(["u", "v", "w", "p", "q", "r"], nu_r_std.cpu().tolist())
        },
        "rotation_std": {
            name: float(value)
            for name, value in zip(["roll", "pitch", "yaw"], rot_std.cpu().tolist())
        },
        "rotation_bias_abs": (
            {
                name: float(value)
                for name, value in zip(["roll", "pitch", "yaw"], rot_bias.cpu().tolist())
            }
            if torch.any(rot_bias > 0) else None
        ),
        "rotation_bias_mode": (
            "per-sample fixed-sign bias"
            if torch.any(rot_bias > 0) else None
        ),
        "v_c_bias_abs": (
            {
                name: float(value)
                for name, value in zip(["x", "y", "z"], current_bias.cpu().tolist())
            }
            if current_bias is not None and torch.any(current_bias > 0) else None
        ),
        "v_c_bias_mode": (
            "per-sample fixed-sign bias"
            if current_bias is not None and torch.any(current_bias > 0) else None
        ),
        "u_act_std": {
            name: float(value)
            for name, value in zip(actuator_labels, act_std.cpu().tolist())
        },
        "v_c_std": (
            {
                name: float(value)
                for name, value in zip(["x", "y", "z"], current_std.cpu().tolist())
            }
            if current_std is not None else None
        ),
    }


def format_noise_budget_summary(budget: Optional[Dict]) -> str:
    if not budget:
        return "clean"

    nu_r_floor_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in budget.get("nu_r_linear_floor_std", {}).items()
    )
    nu_r_ratio_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in budget.get("nu_r_linear_ratio", {}).items()
    )
    nu_r_ang_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in budget.get("nu_r_angular_std", {}).items()
    )
    rot_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in budget.get("rotation_std", {}).items()
    )
    rot_bias = budget.get("rotation_bias_abs") or {}
    rot_bias_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in rot_bias.items()
        if abs(value) > 0.0
    )
    current_bias = budget.get("v_c_bias_abs") or {}
    current_bias_text = ", ".join(
        f"{name}={value:.4f}"
        for name, value in current_bias.items()
        if abs(value) > 0.0
    )
    base_profile = budget.get("base_profile")
    reference = budget.get("reference")
    return (
        f"profile={budget.get('profile', 'clean')}"
        f"{f' (base={base_profile})' if base_profile and base_profile != budget.get('profile', 'clean') else ''}"
        f"{f' | ref={reference}' if reference else ''}"
        f" | epoch_scale={budget.get('epoch_scale', 0.0):.3f}"
        f" | mix_ratio={budget.get('mix_ratio', 0.0):.3f}"
        f" | nu_r_lin_floor[{nu_r_floor_text}]"
        f" | nu_r_lin_ratio[{nu_r_ratio_text}]"
        f" | nu_r_ang[{nu_r_ang_text}]"
        f" | rot[{rot_text}]"
        f"{f' | rot_bias[{rot_bias_text}]' if rot_bias_text else ''}"
        f"{f' | v_c_bias[{current_bias_text}]' if current_bias_text else ''}"
    )


def build_noisy_initial_condition(
    clean_state: torch.Tensor,
    cfg: NoiseConfig,
    model: nn.Module,
    normalizer: StateNormalizer,
    epoch: int,
    *,
    sample_ids: Optional[torch.Tensor] = None,
    base_seed: Optional[int] = None,
    state_is_ode: bool = False,
) -> torch.Tensor:
    """Perturb the clean initial state in ODE space using IC-consistent noise."""
    y0 = clean_state.clone() if state_is_ode else model.to_ode_state(clean_state).clone()
    scale = cfg.epoch_scale(epoch)
    if scale < 1e-8:
        return y0

    batch_size = int(y0.shape[0])
    device = y0.device
    dtype = y0.dtype
    layout = model.layout
    reference = _resolve_noise_reference(cfg.reference)

    nu_r_clean = y0[:, layout.nu_r]
    nu_r_std = scale * _profile_velocity_std_from_state(
        nu_r_clean,
        cfg.profile,
        reference,
    )
    y0[:, layout.nu_r] = y0[:, layout.nu_r] + _sample_scaled_noise(
        nu_r_std,
        batch_size,
        device=device,
        dtype=dtype,
        sample_ids=sample_ids,
        base_seed=base_seed,
        stream=11,
    )

    rot_std = scale * _profile_rotation_std_vector(
        cfg.profile,
        reference,
        dtype=dtype,
        device=device,
    )
    rot_bias = scale * _profile_rotation_bias_vector(
        cfg.profile,
        reference,
        dtype=dtype,
        device=device,
    )
    if torch.any(rot_std > 0) or torch.any(rot_bias > 0):
        delta_theta = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        if torch.any(rot_std > 0):
            delta_theta = delta_theta + _sample_scaled_noise(
                rot_std,
                batch_size,
                device=device,
                dtype=dtype,
                sample_ids=sample_ids,
                base_seed=base_seed,
                stream=23,
            )
        if torch.any(rot_bias > 0):
            bias_sign = _sample_sign_pattern(
                batch_size,
                dim=1,
                device=device,
                dtype=dtype,
                sample_ids=sample_ids,
                base_seed=base_seed,
                stream=29,
            )
            delta_theta = delta_theta + bias_sign * rot_bias.view(1, 3)
        R_clean = y0[:, 3:12].reshape(batch_size, 3, 3)
        R_delta = _so3_exp_map(delta_theta)
        y0[:, 3:12] = _project_to_so3(torch.matmul(R_delta, R_clean)).reshape(batch_size, 9)

    u_act_dim = int(layout.u_act.stop - layout.u_act.start)
    act_std = scale * _profile_actuator_std(
        cfg.profile,
        u_act_dim,
        normalizer.std_act,
        reference,
        dtype=dtype,
        device=device,
    )
    if torch.any(act_std > 0):
        y0[:, layout.u_act] = y0[:, layout.u_act] + _sample_scaled_noise(
            act_std,
            batch_size,
            device=device,
            dtype=dtype,
            sample_ids=sample_ids,
            base_seed=base_seed,
            stream=37,
        )

    if getattr(model, "ocean_current", False):
        current_std = scale * _profile_current_std(
            cfg.profile,
            reference,
            dtype=dtype,
            device=device,
        )
        current_bias = scale * _profile_current_bias_vector(
            cfg.profile,
            reference,
            dtype=dtype,
            device=device,
        )
        if torch.any(current_std > 0):
            y0[:, layout.v_c] = y0[:, layout.v_c] + _sample_scaled_noise(
                current_std,
                batch_size,
                device=device,
                dtype=dtype,
                sample_ids=sample_ids,
                base_seed=base_seed,
                stream=53,
            )
        if torch.any(current_bias > 0):
            current_bias_sign = _sample_sign_pattern(
                batch_size,
                dim=3,
                device=device,
                dtype=dtype,
                sample_ids=sample_ids,
                base_seed=base_seed,
                stream=59,
            )
            y0[:, layout.v_c] = y0[:, layout.v_c] + current_bias_sign * current_bias.view(1, 3)

    return y0


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
    frame_weights: Optional[torch.Tensor] = None,
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

    def weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
        if weights is None:
            return values.mean()
        weighted = values * weights
        denom = weights.sum().clamp_min(1.0)
        return weighted.sum() / denom

    weights_bt = None
    if frame_weights is not None:
        weights_bt = frame_weights.to(device=pred.device, dtype=pred.dtype)

    dx = (pred[..., :3] - target[..., :3]) / normalizer.std_pos
    loss_pos = weighted_mean(dx ** 2, None if weights_bt is None else weights_bt.unsqueeze(-1))

    R_t = target[..., 3:12].reshape(-1, 3, 3)
    R_raw = pred[..., 3:12].reshape(-1, 3, 3)
    R_p = compute_rotation_matrix(pred[..., 3:12].reshape(-1, 9))
    rot_sq = geodesic_distance_so3(R_t, R_p).pow(2).reshape(target.shape[0], target.shape[1])
    loss_rot = weighted_mean(rot_sq, weights_bt)

    so3_reg, so3_orth, so3_det = so3_constraint_terms(R_raw)

    vel_target = target_ode if target_ode is not None else target
    vel_pred = pred_ode if pred_ode is not None else pred
    dv = (vel_pred[..., 12:18] - vel_target[..., 12:18]) / normalizer.std_vel
    if weights_bt is None:
        vel_mse = (dv ** 2).mean(dim=(0, 1))
    else:
        vel_num = ((dv ** 2) * weights_bt.unsqueeze(-1)).sum(dim=(0, 1))
        vel_den = weights_bt.sum().clamp_min(1.0)
        vel_mse = vel_num / vel_den

    loss_act = torch.zeros((), dtype=pred.dtype, device=pred.device)
    if normalizer.std_act is not None:
        u_dim = int(normalizer.std_act.numel())
        u_act_slice = slice(18, 18 + u_dim)
        du_act = (
            pred[..., u_act_slice] - target[..., u_act_slice]
        ) / normalizer.std_act.view(1, 1, -1)
        loss_act = weighted_mean(
            du_act ** 2,
            None if weights_bt is None else weights_bt.unsqueeze(-1),
        )

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
    *,
    noise_cfg: Optional[NoiseConfig] = None,
    normalizer: Optional[StateNormalizer] = None,
    noise_seed: Optional[int] = None,
    sample_ids: Optional[np.ndarray] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, str]]:
    """Evaluate a batch of independent blocks while isolating failing samples."""
    from torchdiffeq import odeint

    batch = torch.tensor(batch_np, dtype=torch.float32, device=device)
    batch_size = int(batch.shape[0])
    if batch_size == 0:
        return {}, {}
    if sample_ids is None:
        sample_ids = np.arange(batch_size, dtype=np.int64)

    if noise_cfg is not None and noise_cfg.is_active:
        if normalizer is None:
            raise ValueError("normalizer is required for noisy evaluation.")
        y0 = build_noisy_initial_condition(
            batch[:, 0],
            noise_cfg,
            model,
            normalizer,
            epoch=noise_cfg.warmup_epochs + noise_cfg.ramp_epochs,
            sample_ids=torch.tensor(sample_ids, dtype=torch.long),
            base_seed=noise_seed,
        )
    else:
        y0 = model.to_ode_state(batch[:, 0])
    model.reset_nfe()
    try:
        pred = odeint(model, y0, t_eval.to(device), method=ode_solver)
    except (ValueError, RuntimeError):
        if batch_size == 1:
            return {}, {0: "solver_failure"}
        split = batch_size // 2
        left_pred, left_fail = _evaluate_block_batch_with_fallback(
            model,
            batch_np[:split],
            t_eval,
            device,
            ode_solver,
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
            sample_ids=sample_ids[:split],
        )
        right_pred, right_fail = _evaluate_block_batch_with_fallback(
            model,
            batch_np[split:],
            t_eval,
            device,
            ode_solver,
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
            sample_ids=sample_ids[split:],
        )
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
            model,
            batch_np[:split],
            t_eval,
            device,
            ode_solver,
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
            sample_ids=sample_ids[:split],
        )
        right_pred, right_fail = _evaluate_block_batch_with_fallback(
            model,
            batch_np[split:],
            t_eval,
            device,
            ode_solver,
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
            sample_ids=sample_ids[split:],
        )
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
    noise_cfg: Optional[NoiseConfig] = None,
    normalizer: Optional[StateNormalizer] = None,
    noise_seed: Optional[int] = None,
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
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
            sample_ids=np.asarray(
                [traj_idx * 100000 + block_idx for block_idx in range(len(traj_blocks))],
                dtype=np.int64,
            ),
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
    n_samples: int = 100,
    noise_cfg: Optional[NoiseConfig] = None,
    normalizer: Optional[StateNormalizer] = None,
    noise_seed: Optional[int] = None,
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
    sample_offset = 0

    for batch in test_loader:
        if count >= n_samples:
            break

        batch_np = batch.cpu().numpy()
        sample_ids = np.arange(sample_offset, sample_offset + len(batch_np), dtype=np.int64)
        pred_map, failure_map = _evaluate_block_batch_with_fallback(
            model,
            batch_np,
            t_eval,
            device,
            ode_solver,
            noise_cfg=noise_cfg,
            normalizer=normalizer,
            noise_seed=noise_seed,
            sample_ids=sample_ids,
        )
        sample_offset += len(batch_np)
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


def _comparison_entry(value: float, clean_value: float, *, higher_is_better: bool = False) -> Dict:
    value = float(value)
    clean_value = float(clean_value)
    if not np.isfinite(value) or not np.isfinite(clean_value):
        return {
            "value": value,
            "clean_value": clean_value,
            "absolute_delta": float("nan"),
            "ratio_to_clean": float("nan"),
            "degradation_pct": float("nan"),
        }

    absolute_delta = value - clean_value
    if abs(clean_value) < 1e-12:
        ratio_to_clean = 1.0 if abs(value) < 1e-12 else float("inf")
        degradation_pct = 0.0 if abs(value) < 1e-12 else float("inf")
    else:
        ratio_to_clean = value / clean_value
        if higher_is_better:
            degradation_pct = (clean_value - value) / abs(clean_value) * 100.0
        else:
            degradation_pct = (value - clean_value) / abs(clean_value) * 100.0

    return {
        "value": value,
        "clean_value": clean_value,
        "absolute_delta": float(absolute_delta),
        "ratio_to_clean": float(ratio_to_clean),
        "degradation_pct": float(degradation_pct),
    }


def _format_ratio_value(value: float) -> str:
    if np.isposinf(value):
        return "inf"
    if np.isneginf(value):
        return "-inf"
    if np.isfinite(value):
        return f"{value:.3f}"
    return "nan"


def _format_pct_value(value: float) -> str:
    if np.isposinf(value):
        return "+inf%"
    if np.isneginf(value):
        return "-inf%"
    if np.isfinite(value):
        return f"{value:+.1f}%"
    return "nan"


def _block_failure_rate(results: Dict) -> float:
    failures = int(results.get("solver_failed_batches", 0)) + int(
        results.get("invalid_prediction_batches", 0)
    )
    total = int(results.get("n_samples", 0)) + failures
    return float(failures / total) if total > 0 else float("nan")


def _heldout_trajectory_failure_rate(results: Dict) -> float:
    total = int(results.get("overall", {}).get("n_trajectories", 0))
    failures = sum(int(v) for v in results.get("failure_counts", {}).values())
    return float(failures / total) if total > 0 else float("nan")


def attach_relative_to_clean_block(results_by_profile: Dict[str, Dict]) -> Dict[str, Dict]:
    clean = results_by_profile.get("clean")
    if clean is None:
        return {}

    comparisons = {}
    clean_failure_rate = _block_failure_rate(clean)
    for profile, profile_results in results_by_profile.items():
        if profile == "clean":
            continue
        failure_rate = _block_failure_rate(profile_results)
        profile_comparison = {
            "position_rmse": _comparison_entry(
                profile_results["position_rmse"]["mean"],
                clean["position_rmse"]["mean"],
            ),
            "rotation_geodesic": _comparison_entry(
                profile_results["rotation_geodesic"]["mean"],
                clean["rotation_geodesic"]["mean"],
            ),
            "velocity_rmse": _comparison_entry(
                profile_results["velocity_rmse"]["mean"],
                clean["velocity_rmse"]["mean"],
            ),
            "angular_rmse": _comparison_entry(
                profile_results["angular_rmse"]["mean"],
                clean["angular_rmse"]["mean"],
            ),
            "failure_rate": _comparison_entry(
                failure_rate,
                clean_failure_rate,
            ),
        }
        profile_results["relative_to_clean"] = profile_comparison
        comparisons[profile] = profile_comparison
    return comparisons


def attach_relative_to_clean_heldout(results_by_profile: Dict[str, Dict]) -> Dict[str, Dict]:
    clean = results_by_profile.get("clean")
    if clean is None:
        return {}

    comparisons = {}
    clean_failure_rate = _heldout_trajectory_failure_rate(clean)
    for profile, profile_results in results_by_profile.items():
        if profile == "clean":
            continue
        failure_rate = _heldout_trajectory_failure_rate(profile_results)
        profile_comparison = {
            "position_rmse": _comparison_entry(
                profile_results["overall"]["position_rmse"]["mean"],
                clean["overall"]["position_rmse"]["mean"],
            ),
            "rotation_geodesic": _comparison_entry(
                profile_results["overall"]["rotation_geodesic"]["mean"],
                clean["overall"]["rotation_geodesic"]["mean"],
            ),
            "velocity_rmse": _comparison_entry(
                profile_results["overall"]["velocity_rmse"]["mean"],
                clean["overall"]["velocity_rmse"]["mean"],
            ),
            "angular_rmse": _comparison_entry(
                profile_results["overall"]["angular_rmse"]["mean"],
                clean["overall"]["angular_rmse"]["mean"],
            ),
            "success_rate": _comparison_entry(
                profile_results["overall"]["success_rate"],
                clean["overall"]["success_rate"],
                higher_is_better=True,
            ),
            "trajectory_failure_rate": _comparison_entry(
                failure_rate,
                clean_failure_rate,
            ),
        }
        profile_results["relative_to_clean"] = profile_comparison
        comparisons[profile] = profile_comparison
    return comparisons


def print_relative_to_clean_summary(
    comparisons: Dict[str, Dict],
    *,
    title: str,
    logger: Optional[logging.Logger] = None,
):
    if not comparisons:
        return

    log = logger.info if logger else print
    log("=" * 50)
    log(title)
    log("=" * 50)
    for profile, profile_comparison in comparisons.items():
        log(f"[{profile}]")
        for metric_name, entry in profile_comparison.items():
            ratio_text = _format_ratio_value(entry["ratio_to_clean"])
            degradation_text = _format_pct_value(entry["degradation_pct"])
            log(f"{metric_name}: ratio={ratio_text}  degradation={degradation_text}")


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
    if results.get("noise_budget") is not None:
        log(f"Noise budget: {format_noise_budget_summary(results['noise_budget'])}")


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
    if results.get("noise_budget") is not None:
        log(f"Noise budget: {format_noise_budget_summary(results['noise_budget'])}")

    for scenario, summary in results["by_scenario"].items():
        log(
            f"[{scenario}] pos={summary['position_rmse']['mean']:.4f} m, "
            f"rot={summary['rotation_geodesic']['mean']:.4f} rad, "
            f"success={summary['success_rate']:.4f}"
        )


def save_heldout_evaluation_results(results: Dict, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "overall" not in results:
        payload = {}
        for profile, profile_results in results.items():
            entry = {
                "overall": profile_results["overall"],
                "by_scenario": profile_results["by_scenario"],
                "failure_counts": profile_results["failure_counts"],
                "block_failure_counts": profile_results.get("block_failure_counts", {}),
            }
            if profile_results.get("noise_budget") is not None:
                entry["noise_budget"] = profile_results["noise_budget"]
            if profile_results.get("relative_to_clean") is not None:
                entry["relative_to_clean"] = profile_results["relative_to_clean"]
            payload[profile] = entry
        with open(output_dir / "heldout_evaluation.json", "w") as f:
            json.dump(payload, f, indent=2)

        summary_lines = ["Held-Out Trajectory Evaluation", "=" * 80]
        for profile, profile_results in results.items():
            overall = profile_results["overall"]
            summary_lines.append(f"[{profile}]")
            summary_lines.append(
                f"Trajectories: {overall['n_trajectories']}  Success rate: {overall['success_rate']:.4f}"
            )
            summary_lines.append(
                f"Position RMSE [m]: mean={overall['position_rmse']['mean']:.4f}, "
                f"median={overall['position_rmse']['median']:.4f}, "
                f"p95={overall['position_rmse']['p95']:.4f}"
            )
            summary_lines.append(
                f"Rotation geo [rad]: mean={overall['rotation_geodesic']['mean']:.4f}, "
                f"median={overall['rotation_geodesic']['median']:.4f}, "
                f"p95={overall['rotation_geodesic']['p95']:.4f}"
            )
            summary_lines.append(
                f"Lin vel RMSE [m/s]: {overall['velocity_rmse']['mean']:.4f}  "
                f"Ang vel RMSE [r/s]: {overall['angular_rmse']['mean']:.4f}"
            )
            if profile_results.get("noise_budget") is not None:
                summary_lines.append(
                    f"Noise budget: {format_noise_budget_summary(profile_results['noise_budget'])}"
                )
            if profile_results.get("relative_to_clean") is not None:
                summary_lines.append("Relative to clean:")
                for metric_name, entry in profile_results["relative_to_clean"].items():
                    ratio_text = _format_ratio_value(entry["ratio_to_clean"])
                    degradation_text = _format_pct_value(entry["degradation_pct"])
                    summary_lines.append(
                        f"  {metric_name}: ratio={ratio_text}, degradation={degradation_text}"
                    )
            summary_lines.append("")

            rows = profile_results.get("trajectory_rows", [])
            if rows:
                with open(output_dir / f"heldout_trajectory_metrics_{profile}.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

        with open(output_dir / "heldout_evaluation.txt", "w") as f:
            f.write("\n".join(summary_lines).rstrip() + "\n")
        return

    with open(output_dir / "heldout_evaluation.json", "w") as f:
        json.dump(
            {
                "overall": results["overall"],
                "by_scenario": results["by_scenario"],
                "failure_counts": results["failure_counts"],
                "block_failure_counts": results.get("block_failure_counts", {}),
                "noise_budget": results.get("noise_budget"),
                "relative_to_clean": results.get("relative_to_clean"),
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
    if results.get("noise_budget") is not None:
        lines.append(f"Noise budget: {format_noise_budget_summary(results['noise_budget'])}")
    if results.get("relative_to_clean") is not None:
        lines.append("Relative to clean:")
        for metric_name, entry in results["relative_to_clean"].items():
            ratio_text = _format_ratio_value(entry["ratio_to_clean"])
            degradation_text = _format_pct_value(entry["degradation_pct"])
            lines.append(
                f"{metric_name}: ratio={ratio_text}, degradation={degradation_text}"
            )

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

    if "position_rmse" not in results:
        with open(output_dir / "block_evaluation.json", "w") as f:
            json.dump(results, f, indent=2)

        lines = ["Block-Level Evaluation", "=" * 80]
        for profile, profile_results in results.items():
            lines.append(f"[{profile}]")
            lines.append(
                f"Position RMSE [m]: mean={profile_results['position_rmse']['mean']:.4f}, "
                f"std={profile_results['position_rmse']['std']:.4f}, "
                f"max={profile_results['position_rmse']['max']:.4f}"
            )
            lines.append(
                f"Rotation geo [rad]: mean={profile_results['rotation_geodesic']['mean']:.4f}, "
                f"std={profile_results['rotation_geodesic']['std']:.4f}, "
                f"max={profile_results['rotation_geodesic']['max']:.4f}"
            )
            lines.append(
                f"Lin vel RMSE [m/s]: mean={profile_results['velocity_rmse']['mean']:.4f}, "
                f"std={profile_results['velocity_rmse']['std']:.4f}"
            )
            lines.append(
                f"Ang vel RMSE [r/s]: mean={profile_results['angular_rmse']['mean']:.4f}, "
                f"std={profile_results['angular_rmse']['std']:.4f}"
            )
            if profile_results.get("noise_budget") is not None:
                lines.append(
                    f"Noise budget: {format_noise_budget_summary(profile_results['noise_budget'])}"
                )
            if profile_results.get("relative_to_clean") is not None:
                lines.append("Relative to clean:")
                for metric_name, entry in profile_results["relative_to_clean"].items():
                    ratio_text = _format_ratio_value(entry["ratio_to_clean"])
                    degradation_text = _format_pct_value(entry["degradation_pct"])
                    lines.append(
                        f"  {metric_name}: ratio={ratio_text}, degradation={degradation_text}"
                    )
            lines.append("")

        with open(output_dir / "block_evaluation.txt", "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        return

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
    if results.get("noise_budget") is not None:
        lines.append(f"Noise budget: {format_noise_budget_summary(results['noise_budget'])}")
    if results.get("relative_to_clean") is not None:
        lines.append("Relative to clean:")
        for metric_name, entry in results["relative_to_clean"].items():
            ratio_text = _format_ratio_value(entry["ratio_to_clean"])
            degradation_text = _format_pct_value(entry["degradation_pct"])
            lines.append(f"{metric_name}: ratio={ratio_text}, degradation={degradation_text}")

    with open(output_dir / "block_evaluation.txt", "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
