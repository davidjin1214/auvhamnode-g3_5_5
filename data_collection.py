#!/usr/bin/env python3
"""
AUVHamNODE Data Collection

Generates training data for the port-Hamiltonian Neural ODE model.
Each sample is a control block with state snapshots at dt_state intervals.

State vector without current (24D/25D): [Δpos(3), R(9), ν(6), u_actual(3), u_cmd(3), z_ref?(1)]
State vector with current    (27D/28D): [Δpos(3), R(9), ν(6), u_actual(3), u_cmd(3), v_c^n(3), z_ref?(1)]
  Δpos     -- position relative to block start (inertial frame)
  R        -- rotation matrix (body -> inertial), row-major flattened
  ν        -- total body-frame velocity [v(3), ω(3)] (NOT relative to water)
  u_actual -- actuator states [δ_r, δ_s, n] at each snapshot
  u_cmd    -- actuator commands [δ_r_c, δ_s_c, n_c] (constant per block)
  z_ref    -- optional absolute block-start depth, carried as context so
              depth-aware models can reconstruct absolute depth

In both cases the raw total body-frame velocity ν is stored as the dataset
contract. When ocean current is enabled, the model-space relative velocity is
obtained from the shared kinematic relation
  ν_r = ν - R^T v_c^n
using the stored rotation matrix and inertial-frame current.

The ODE model integrates actuator dynamics (first-order lag from u_actual
toward u_cmd) alongside rigid-body dynamics, so the force is computed at
every solver step rather than held constant.
"""

import numpy as np
import math
import pickle
import json
import hashlib
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum, auto
from datetime import datetime
from collections import defaultdict

from remus100_core import (
    Remus100Dynamics, Remus100Simulator,
    rotation_matrix_from_quaternion, euler_to_quaternion,
)


# =============================================================================
# Configuration
# =============================================================================

class ScenarioType(Enum):
    PRBS = auto()
    CHIRP = auto()
    OU = auto()


@dataclass
class Config:
    """
    Timing hierarchy: dt_sim (0.01s) -> dt_state (0.05s) -> dt_ctrl (0.2s)
    Each training sample spans one control interval with 5 state snapshots.
    """
    # Dataset size
    num_trajectories: int = 500
    blocks_per_trajectory: int = 150      # 150 x 0.2s = 30s per trajectory
    train_ratio: float = 0.8

    # Timing
    dt_sim: float = 0.01
    dt_state: float = 0.05
    dt_ctrl: float = 0.2
    warmup_time: float = 3.0             # >= 3 x T_n for propeller convergence

    @property
    def sim_steps_per_state(self) -> int:
        return int(round(self.dt_state / self.dt_sim))        # 5

    @property
    def state_intervals_per_block(self) -> int:
        return int(round(self.dt_ctrl / self.dt_state))       # 4

    @property
    def points_per_block(self) -> int:
        return self.state_intervals_per_block + 1             # 5

    @property
    def state_dim(self) -> int:
        dim = 24 + (3 if self.ocean_current else 0)
        if self.absolute_depth_context:
            dim += 1
        return dim

    # Scenario weights (must sum to 1)
    scenario_weights: Dict[ScenarioType, float] = field(default_factory=lambda: {
        ScenarioType.PRBS:  0.40,
        ScenarioType.CHIRP: 0.35,
        ScenarioType.OU:    0.25,
    })

    # Initial conditions
    init_depth: Tuple[float, float] = (5.0, 40.0)
    init_surge: Tuple[float, float] = (0.8, 2.5)
    init_sway_std: float = 0.25
    init_heave_std: float = 0.15
    init_angular_std: float = 0.15

    # Actuator limits
    delta_max: float = 15.0 * np.pi / 180                    # rad
    rpm_range: Tuple[float, float] = (400, 1400)

    # PRBS
    prbs_min_hold: int = 3                                    # sim steps
    prbs_max_hold: int = 20
    prbs_smooth_width: int = 5

    # Chirp frequency ranges (Hz) for rudder and stern plane
    chirp_freqs_rudder: list = field(
        default_factory=lambda: [(0.01, 0.15), (0.05, 0.3), (0.1, 0.5)]
    )
    chirp_freqs_stern: list = field(
        default_factory=lambda: [(0.02, 0.2), (0.08, 0.4)]
    )

    # Ornstein-Uhlenbeck
    ou_theta: float = 0.3                                     # mean-reversion rate
    ou_sigma: float = 0.8                                     # volatility
    ou_impulse_count: Tuple[int, int] = (5, 15)

    # RPM variation
    rpm_ou_theta: float = 0.1
    rpm_ou_sigma_frac: float = 0.6                            # fraction of rpm half-range

    # Divergence limits
    max_attitude: float = 1.2                                 # rad (~69 deg)
    depth_bounds: Tuple[float, float] = (0.5, 100.0)
    velocity_max: np.ndarray = field(
        default_factory=lambda: np.array([4.0, 0.8, 0.8, 0.5, 0.5, 0.5])
    )

    # Quality filtering
    enable_filtering: bool = True
    min_excitation_percentile: float = 30.0                   # keep top 70%
    rotation_tolerance: float = 1e-4

    # Ocean current
    ocean_current: bool = False
    absolute_depth_context: bool = False
    current_speed_range: Tuple[float, float] = (0.0, 0.5)    # m/s
    current_direction_range: Tuple[float, float] = (-np.pi, np.pi)  # rad
    current_vertical_std: float = 0.05                        # m/s
    current_drift_sigma: float = 0.02                         # slow OU drift rate
    current_drift_theta: float = 0.005                        # OU mean reversion

    # Output
    seed: int = 42
    save_dir: str = './data'
    num_workers: int = 8
    save_figures: bool = True


def _scenario_weights_to_payload(weights: Dict[ScenarioType, float]) -> Dict[str, float]:
    """Serialize scenario weights with stable enum-name keys."""
    return {scenario.name: float(weight) for scenario, weight in weights.items()}


def _scenario_weights_from_payload(payload: Dict[str, float]) -> Dict[ScenarioType, float]:
    """Restore scenario weights from a JSON-safe payload."""
    return {
        ScenarioType[str(name).upper()]: float(weight)
        for name, weight in payload.items()
    }


def serialize_config(config: Config) -> Dict:
    """Convert the full generation config to a JSON-safe dictionary."""
    return {
        "num_trajectories": int(config.num_trajectories),
        "blocks_per_trajectory": int(config.blocks_per_trajectory),
        "train_ratio": float(config.train_ratio),
        "dt_sim": float(config.dt_sim),
        "dt_state": float(config.dt_state),
        "dt_ctrl": float(config.dt_ctrl),
        "warmup_time": float(config.warmup_time),
        "scenario_weights": _scenario_weights_to_payload(config.scenario_weights),
        "init_depth": list(map(float, config.init_depth)),
        "init_surge": list(map(float, config.init_surge)),
        "init_sway_std": float(config.init_sway_std),
        "init_heave_std": float(config.init_heave_std),
        "init_angular_std": float(config.init_angular_std),
        "delta_max": float(config.delta_max),
        "rpm_range": list(map(float, config.rpm_range)),
        "prbs_min_hold": int(config.prbs_min_hold),
        "prbs_max_hold": int(config.prbs_max_hold),
        "prbs_smooth_width": int(config.prbs_smooth_width),
        "chirp_freqs_rudder": [list(map(float, pair)) for pair in config.chirp_freqs_rudder],
        "chirp_freqs_stern": [list(map(float, pair)) for pair in config.chirp_freqs_stern],
        "ou_theta": float(config.ou_theta),
        "ou_sigma": float(config.ou_sigma),
        "ou_impulse_count": list(map(int, config.ou_impulse_count)),
        "rpm_ou_theta": float(config.rpm_ou_theta),
        "rpm_ou_sigma_frac": float(config.rpm_ou_sigma_frac),
        "max_attitude": float(config.max_attitude),
        "depth_bounds": list(map(float, config.depth_bounds)),
        "velocity_max": np.asarray(config.velocity_max, dtype=float).tolist(),
        "enable_filtering": bool(config.enable_filtering),
        "min_excitation_percentile": float(config.min_excitation_percentile),
        "rotation_tolerance": float(config.rotation_tolerance),
        "ocean_current": bool(config.ocean_current),
        "absolute_depth_context": bool(config.absolute_depth_context),
        "current_speed_range": list(map(float, config.current_speed_range)),
        "current_direction_range": list(map(float, config.current_direction_range)),
        "current_vertical_std": float(config.current_vertical_std),
        "current_drift_sigma": float(config.current_drift_sigma),
        "current_drift_theta": float(config.current_drift_theta),
        "seed": int(config.seed),
        "save_dir": str(config.save_dir),
        "num_workers": int(config.num_workers),
        "save_figures": bool(config.save_figures),
    }


def deserialize_config(payload: Dict) -> Config:
    """Restore a Config instance from a JSON-safe dictionary."""
    cfg = dict(payload)
    cfg["scenario_weights"] = _scenario_weights_from_payload(cfg["scenario_weights"])
    cfg["init_depth"] = tuple(map(float, cfg["init_depth"]))
    cfg["init_surge"] = tuple(map(float, cfg["init_surge"]))
    cfg["rpm_range"] = tuple(map(float, cfg["rpm_range"]))
    cfg["ou_impulse_count"] = tuple(map(int, cfg["ou_impulse_count"]))
    cfg["depth_bounds"] = tuple(map(float, cfg["depth_bounds"]))
    cfg["velocity_max"] = np.asarray(cfg["velocity_max"], dtype=float)
    cfg["current_speed_range"] = tuple(map(float, cfg["current_speed_range"]))
    cfg["current_direction_range"] = tuple(map(float, cfg["current_direction_range"]))
    cfg["chirp_freqs_rudder"] = [tuple(map(float, pair)) for pair in cfg["chirp_freqs_rudder"]]
    cfg["chirp_freqs_stern"] = [tuple(map(float, pair)) for pair in cfg["chirp_freqs_stern"]]
    return Config(**cfg)


def sample_initial_state(rng, config):
    """Sample the trajectory initial condition using the dataset contract."""
    z0 = rng.uniform(*config.init_depth)
    psi0 = rng.uniform(-np.pi, np.pi)
    phi0 = rng.normal(0.0, 0.05)
    theta0 = rng.normal(0.0, 0.05)

    eta0 = np.array([0.0, 0.0, z0, phi0, theta0, psi0], dtype=np.float64)
    nu0 = np.array(
        [
            rng.uniform(*config.init_surge),
            rng.normal(0.0, config.init_sway_std),
            rng.normal(0.0, config.init_heave_std),
            rng.normal(0.0, config.init_angular_std),
            rng.normal(0.0, config.init_angular_std),
            rng.normal(0.0, config.init_angular_std),
        ],
        dtype=np.float64,
    )
    return eta0, nu0


def sample_scenario_type(rng, config):
    """Sample the trajectory excitation scenario using the dataset contract."""
    types = list(config.scenario_weights.keys())
    probs = [config.scenario_weights[scenario] for scenario in types]
    return types[rng.choice(len(types), p=probs)]


def replay_trajectory_setup(seed, config):
    """Replay the dataset RNG contract up to the current-generation stage."""
    rng = np.random.default_rng(seed)
    eta0, nu0 = sample_initial_state(rng, config)
    scenario_type = sample_scenario_type(rng, config)
    return rng, eta0, nu0, scenario_type


# =============================================================================
# Signal Generators
# =============================================================================

def prbs(n, dt, amplitude, min_hold, max_hold, rng):
    """Pseudo-random binary sequence with variable hold times."""
    signal = np.zeros(n)
    idx = 0
    val = amplitude * (2 * rng.integers(2) - 1)
    while idx < n:
        hold = rng.integers(min_hold, max_hold + 1)
        signal[idx:min(idx + hold, n)] = val
        val = -val
        idx += hold
    return signal


def multi_chirp(n, dt, freq_ranges, rng):
    """Superposition of chirps with equal amplitude and random phases."""
    t = np.arange(n) * dt
    T = n * dt
    amp = 1.0 / len(freq_ranges)
    signal = np.zeros(n)
    for f0, f1 in freq_ranges:
        k = (f1 - f0) / T
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2) + phase)
    return signal


def ornstein_uhlenbeck(n, dt, mean, sigma, theta, rng):
    """Ornstein-Uhlenbeck process (mean-reverting stochastic signal)."""
    x = np.zeros(n)
    x[0] = mean + rng.standard_normal() * sigma / np.sqrt(2 * theta)
    sqrt_dt = np.sqrt(dt)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mean - x[i - 1]) * dt \
               + sigma * sqrt_dt * rng.standard_normal()
    return x


# =============================================================================
# Scenario Factory
# =============================================================================

def create_scenario_generator(scenario_type, config, seed):
    """
    Build an actuator command generator for the given scenario type.
    Returns a callable: time (float) -> np.array([delta_r, delta_s, rpm]).
    """
    rng = np.random.default_rng(seed)

    total_time = config.warmup_time + config.blocks_per_trajectory * config.dt_ctrl + 5.0
    n_pts = int(total_time / config.dt_sim) + 100
    dt = config.dt_sim

    d_max = config.delta_max
    rpm_lo, rpm_hi = config.rpm_range
    rpm_mid = (rpm_lo + rpm_hi) / 2
    rpm_amp = (rpm_hi - rpm_lo) / 2 * config.rpm_ou_sigma_frac

    if scenario_type == ScenarioType.PRBS:
        dr_raw = prbs(n_pts, dt, 1.0, config.prbs_min_hold, config.prbs_max_hold, rng)
        ds_raw = prbs(n_pts, dt, 1.0, config.prbs_min_hold, config.prbs_max_hold, rng)
        kernel = np.ones(config.prbs_smooth_width) / config.prbs_smooth_width
        dr = np.convolve(dr_raw, kernel, mode='same') * d_max * 0.9
        ds = np.convolve(ds_raw, kernel, mode='same') * d_max * 0.9
        rpm = ornstein_uhlenbeck(n_pts, dt, rpm_mid, rpm_amp, config.rpm_ou_theta, rng)

    elif scenario_type == ScenarioType.CHIRP:
        dr = multi_chirp(n_pts, dt, config.chirp_freqs_rudder, rng) * d_max * 0.85
        ds = multi_chirp(n_pts, dt, config.chirp_freqs_stern, rng) * d_max * 0.85
        dr = np.clip(dr, -d_max, d_max)
        ds = np.clip(ds, -d_max, d_max)
        phase = rng.uniform(0, 2 * np.pi)
        rpm = rpm_mid + rpm_amp * 0.5 * np.sin(
            2 * np.pi * 0.02 * np.arange(n_pts) * dt + phase
        )

    else:  # OU
        bias_r = rng.uniform(-0.3, 0.3)
        bias_s = rng.uniform(-0.3, 0.3)
        dr = ornstein_uhlenbeck(n_pts, dt, bias_r, config.ou_sigma, config.ou_theta, rng)
        ds = ornstein_uhlenbeck(n_pts, dt, bias_s, config.ou_sigma, config.ou_theta, rng)
        n_imp = rng.integers(*config.ou_impulse_count)
        for _ in range(n_imp):
            pos = rng.integers(0, n_pts - 30)
            width = rng.integers(10, 30)
            amp = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
            dr[pos:pos + width] += amp
            ds[pos:pos + width] += amp * rng.uniform(-0.5, 0.5)
        dr = np.clip(dr * d_max, -d_max, d_max)
        ds = np.clip(ds * d_max, -d_max, d_max)
        rpm = ornstein_uhlenbeck(n_pts, dt, rpm_mid, rpm_amp * 0.8,
                                 config.rpm_ou_theta + 0.05, rng)

    rpm = np.clip(rpm, rpm_lo, rpm_hi)

    def generator(t):
        idx = min(int(t / dt), n_pts - 1)
        return np.array([dr[idx], ds[idx], rpm[idx]])

    return generator


# =============================================================================
# Trajectory Simulation
# =============================================================================

def _generate_ocean_current(config, rng, n_steps, dt):
    """Generate a slowly time-varying ocean current trajectory.

    Returns:
        v_c_inertial: (n_steps, 3) inertial-frame current velocity [m/s]
        V_c_seq:      (n_steps,)   current speed for simulator
        beta_c_seq:   (n_steps,)   current direction for simulator
    """
    V_c0 = rng.uniform(*config.current_speed_range)
    beta_c0 = rng.uniform(*config.current_direction_range)
    v_z0 = rng.normal(0, config.current_vertical_std)

    # Slowly varying via Ornstein-Uhlenbeck drift
    V_c = np.zeros(n_steps)
    beta_c = np.zeros(n_steps)
    v_z = np.zeros(n_steps)
    V_c[0], beta_c[0], v_z[0] = V_c0, beta_c0, v_z0
    sqrt_dt = np.sqrt(dt)
    theta = config.current_drift_theta
    sigma = config.current_drift_sigma

    for i in range(1, n_steps):
        V_c[i] = V_c[i-1] + theta * (V_c0 - V_c[i-1]) * dt \
                 + sigma * sqrt_dt * rng.standard_normal()
        beta_c[i] = beta_c[i-1] + theta * 0.1 * (beta_c0 - beta_c[i-1]) * dt \
                    + sigma * 0.5 * sqrt_dt * rng.standard_normal()
        v_z[i] = v_z[i-1] + theta * (v_z0 - v_z[i-1]) * dt \
                 + sigma * 0.3 * sqrt_dt * rng.standard_normal()

    V_c = np.clip(V_c, 0, config.current_speed_range[1] * 1.5)

    # Inertial-frame current: [V_c*cos(beta_c), V_c*sin(beta_c), v_z]
    v_c_inertial = np.stack([
        V_c * np.cos(beta_c),
        V_c * np.sin(beta_c),
        v_z,
    ], axis=1)

    return v_c_inertial, V_c, beta_c


def simulate_trajectory(seed, config):
    """
    Simulate one trajectory and extract training blocks.

    Returns a trajectory record or None if diverged.
    """
    rng, eta0, nu0, scenario_type = replay_trajectory_setup(seed, config)

    cmd_gen = create_scenario_generator(scenario_type, config, seed * 1000)

    # --- Ocean current ---
    total_time = config.warmup_time + config.blocks_per_trajectory * config.dt_ctrl + 5.0
    n_sim_steps = int(total_time / config.dt_sim) + 100
    if config.ocean_current:
        v_c_inertial, V_c_seq, beta_c_seq = _generate_ocean_current(
            config, rng, n_sim_steps, config.dt_sim)
    else:
        v_c_inertial = np.zeros((n_sim_steps, 3))
        V_c_seq = np.zeros(n_sim_steps)
        beta_c_seq = np.zeros(n_sim_steps)

    dyn = Remus100Dynamics()
    sim = Remus100Simulator(dyn, dt=config.dt_sim)
    sim.reset(
        eta0, nu0,
        v_c_inertial=v_c_inertial[0] if config.ocean_current else None,
    )

    sim_step_idx = 0

    def _step_with_current(u_control):
        nonlocal sim_step_idx
        idx = min(sim_step_idx, n_sim_steps - 1)
        result = sim.step(
            u_control,
            v_c_inertial=v_c_inertial[idx] if config.ocean_current else None,
        )
        sim_step_idx += 1
        return result

    # --- Warmup (actuators converge to initial commands) ---
    warmup_steps = int(config.warmup_time / config.dt_sim)
    for _ in range(warmup_steps):
        _step_with_current(cmd_gen(sim.time))
        if _check_divergence(sim, config):
            return None

    # --- Data collection ---
    blocks = []
    all_velocities = []

    for _ in range(config.blocks_per_trajectory):
        pos_ref = sim.eta[:3].copy()
        depth_ref = float(pos_ref[2]) if config.absolute_depth_context else None

        # Command for this block (held constant for dt_ctrl)
        u_cmd = cmd_gen(sim.time)

        # Current velocity at block start (inertial frame)
        vc_idx = min(sim_step_idx, n_sim_steps - 1)
        v_c_block = v_c_inertial[vc_idx]

        # Snapshot 0: actuators still reflect previous command's steady-state
        snapshots = [_build_snapshot(sim, pos_ref, u_cmd,
                                     v_c=v_c_block if config.ocean_current else None,
                                     depth_ref=depth_ref)]
        all_velocities.append(sim.nu.copy())

        # Simulate state_intervals_per_block intervals of dt_state each
        for _ in range(config.state_intervals_per_block):
            for _ in range(config.sim_steps_per_state):
                _step_with_current(u_cmd)
                if _check_divergence(sim, config):
                    return None
            vc_idx = min(sim_step_idx, n_sim_steps - 1)
            v_c_snap = v_c_inertial[vc_idx]
            snapshots.append(_build_snapshot(sim, pos_ref, u_cmd,
                                             v_c=v_c_snap if config.ocean_current else None,
                                             depth_ref=depth_ref))
            all_velocities.append(sim.nu.copy())

        blocks.append(np.array(snapshots))

    vel_array = np.array(all_velocities)
    blocks_array = np.array(blocks)
    traj_stats = {
        'scenario': scenario_type.name,
        'vel_std': np.std(vel_array, axis=0),
        'vel_range': np.ptp(vel_array, axis=0),
    }
    meta = {
        'seed': int(seed),
        'scenario': scenario_type.name,
        'eta0': eta0.copy(),
        'nu0': nu0.copy(),
        'ocean_current': config.ocean_current,
    }
    return {
        'blocks': blocks_array,
        'traj_stats': traj_stats,
        'meta': meta,
    }


def _build_snapshot(sim, pos_ref, u_cmd, v_c=None, depth_ref=None):
    """Build state snapshot.

    Without current (24D/25D): [Dpos(3), R_flat(9), nu(6), u_actual(3), u_cmd(3), z_ref?(1)]
    With current    (27D/28D): [Dpos(3), R_flat(9), nu(6), u_actual(3), u_cmd(3), v_c^n(3), z_ref?(1)]

    Always stores the total body-frame velocity nu. When ocean current is
    enabled, training and rollout evaluation convert it to the model-space
    relative velocity nu_r = nu - R^T v_c^n using the same stored rotation
    matrix and current vector.
    """
    dpos = sim.eta[:3] - pos_ref
    R_flat = rotation_matrix_from_quaternion(sim.quaternion).flatten()
    base = np.concatenate([dpos, R_flat, sim.nu, sim.u_actual, u_cmd])
    if v_c is not None:
        base = np.concatenate([base, v_c])
    if depth_ref is not None:
        base = np.concatenate([base, np.array([depth_ref], dtype=np.float64)])
    return base


def _check_divergence(sim, config):
    """Detect simulation blow-up: attitude, depth, velocity, or NaN."""
    if np.any(np.isnan(sim.nu)) or np.any(np.isnan(sim.eta)):
        return True
    if abs(sim.eta[3]) > config.max_attitude or abs(sim.eta[4]) > config.max_attitude:
        return True
    depth = sim.eta[2]
    if depth < config.depth_bounds[0] or depth > config.depth_bounds[1]:
        return True
    if np.any(np.abs(sim.nu) > config.velocity_max * 1.5):
        return True
    return False


# =============================================================================
# Quality Filtering (trajectory-level)
# =============================================================================

def _excitation_score(traj_stats):
    """
    Trajectory excitation quality based on velocity statistics.
    Higher = better 6-DOF coverage.  Lateral and angular DOFs receive
    higher weights because they are harder to excite on torpedo-shaped AUVs.
    """
    scales = np.array([0.3, 0.1, 0.1, 0.03, 0.08, 0.05])
    weights = np.array([1.0, 3.0, 3.0, 4.0, 2.0, 3.0])

    norm_std = traj_stats['vel_std'] / scales
    norm_range = traj_stats['vel_range'] / (2 * scales)
    return float(np.sum(weights * norm_std) + 0.5 * np.sum(weights * norm_range))


def _validate_trajectory(blocks, config):
    """Check rotation matrix orthogonality and velocity bounds."""
    for block in blocks:
        R_flat = block[:, 3:12]
        for row in R_flat:
            R = row.reshape(3, 3)
            if abs(np.linalg.det(R) - 1.0) > config.rotation_tolerance:
                return False
            if np.linalg.norm(R @ R.T - np.eye(3)) > config.rotation_tolerance:
                return False
        if np.any(np.abs(block[:, 12:18]) > config.velocity_max):
            return False
    return True


def filter_trajectories(results, config, verbose=True):
    """
    Two-stage trajectory-level filter:
      1. Physical validity (SO(3) + velocity bounds)
      2. Excitation quality percentile threshold

    Returns: list of kept trajectory records
    """
    # Stage 1: physical validity
    valid = [record for record in results if _validate_trajectory(record['blocks'], config)]
    if verbose:
        print(f"  Validity: {len(valid)}/{len(results)} trajectories passed")

    if not valid:
        return results

    # Stage 2: excitation quality
    scores = np.array([_excitation_score(record['traj_stats']) for record in valid])
    threshold = np.percentile(scores, config.min_excitation_percentile)

    kept = [record for record, sc in zip(valid, scores) if sc >= threshold]
    if verbose:
        pct = 100 - config.min_excitation_percentile
        print(f"  Excitation: {len(kept)}/{len(valid)} trajectories passed (top {pct:.0f}%)")
        print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}], "
              f"threshold: {threshold:.2f}")

    return kept


def _empty_trajectory_array(config):
    return np.empty(
        (0, config.blocks_per_trajectory, config.points_per_block, config.state_dim),
        dtype=np.float64,
    )


def _stack_trajectories(records, config):
    if not records:
        return _empty_trajectory_array(config)
    return np.stack([record['blocks'] for record in records], axis=0)


# =============================================================================
# Statistics
# =============================================================================

def print_statistics(data, title="Dataset Statistics"):
    """Print velocity and actuator statistics for the dataset."""
    state_dim = data.shape[-1]
    flat = data.reshape(-1, state_dim)
    vel = flat[:, 12:18]
    u_act = flat[:, 18:21]
    u_cmd = flat[:, 21:24]

    print(f"\n{'=' * 60}")
    print(title)
    print('=' * 60)
    print(f"Samples: {len(data)}, total state points: {len(flat)}")

    vel_labels = ['u(surge)', 'v(sway)', 'w(heave)',
                  'p(roll)', 'q(pitch)', 'r(yaw)']
    print(f"\n{'Velocity':<12} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print('-' * 52)
    for i, lbl in enumerate(vel_labels):
        v = vel[:, i]
        print(f"{lbl:<12} {v.mean():>9.4f} {v.std():>9.4f} "
              f"{v.min():>9.4f} {v.max():>9.4f}")

    stds = vel.std(axis=0)
    ref = max(stds[0], 1e-8)
    print("\nExcitation ratios (surge_std / DOF_std):")
    for i, lbl in enumerate(vel_labels[1:], 1):
        ratio = ref / max(stds[i], 1e-8)
        tag = "ok" if ratio < 20 else ("low" if ratio < 50 else "poor")
        print(f"  {lbl:<12} {ratio:>6.1f}x  [{tag}]")

    act_labels = ['delta_r(rad)', 'delta_s(rad)', 'n(RPM)']
    print(f"\n{'Actuator':<14} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print('-' * 56)
    for i, lbl in enumerate(act_labels):
        a = u_act[:, i]
        print(f"{lbl:<14} {a.mean():>9.4f} {a.std():>9.4f} "
              f"{a.min():>9.4f} {a.max():>9.4f}")

    cmd_labels = ['delta_r_c', 'delta_s_c', 'n_c(RPM)']
    print(f"\n{'Command':<14} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
    print('-' * 56)
    for i, lbl in enumerate(cmd_labels):
        c = u_cmd[:, i]
        print(f"{lbl:<14} {c.mean():>9.4f} {c.std():>9.4f} "
              f"{c.min():>9.4f} {c.max():>9.4f}")

    has_current = state_dim in {27, 28}
    has_depth_context = state_dim in {25, 28}

    if has_current:
        v_c = flat[:, 24:27]
        vc_labels = ['v_c_x(m/s)', 'v_c_y(m/s)', 'v_c_z(m/s)']
        print(f"\n{'Current':<14} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        print('-' * 56)
        for i, lbl in enumerate(vc_labels):
            vc = v_c[:, i]
            print(f"{lbl:<14} {vc.mean():>9.4f} {vc.std():>9.4f} "
                  f"{vc.min():>9.4f} {vc.max():>9.4f}")
        speed = np.linalg.norm(v_c, axis=1)
        print(f"{'|v_c|(m/s)':<14} {speed.mean():>9.4f} {speed.std():>9.4f} "
              f"{speed.min():>9.4f} {speed.max():>9.4f}")

    if has_depth_context:
        z_ref = flat[:, -1]
        print(f"\n{'Depth ref':<14} {'Mean':>9} {'Std':>9} {'Min':>9} {'Max':>9}")
        print('-' * 56)
        print(f"{'z_ref(m)':<14} {z_ref.mean():>9.4f} {z_ref.std():>9.4f} "
              f"{z_ref.min():>9.4f} {z_ref.max():>9.4f}")


# =============================================================================
# Persistent Statistics
# =============================================================================

POS_LABELS = ['x_rel_n', 'y_rel_n', 'z_rel_n']
EULER_LABELS = ['roll', 'pitch', 'yaw']
G_BODY_LABELS = ['g_bx', 'g_by', 'g_bz']
TWIST_LABELS = ['u', 'v', 'w', 'p', 'q', 'r']
ACTUATOR_LABELS = ['delta_r', 'delta_s', 'rpm']
COMMAND_LABELS = ['delta_r_cmd', 'delta_s_cmd', 'rpm_cmd']
CURRENT_LABELS = ['v_cx_n', 'v_cy_n', 'v_cz_n']
CURRENT_BODY_LABELS = ['v_cx_b', 'v_cy_b', 'v_cz_b']
NORM_LABELS = ['linear_speed', 'angular_speed']


def _scalar_stats(values):
    """Compact scalar distribution summary for paper-ready reporting."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'min': float('nan'),
            'max': float('nan'),
            'median': float('nan'),
            'p05': float('nan'),
            'p25': float('nan'),
            'p75': float('nan'),
            'p95': float('nan'),
            'p99': float('nan'),
            'rms': float('nan'),
            'abs_max': float('nan'),
        }

    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'p05': float(np.percentile(arr, 5)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'rms': float(np.sqrt(np.mean(arr ** 2))),
        'abs_max': float(np.max(np.abs(arr))),
    }


def _matrix_stats(values, labels):
    """Per-channel stats for a 2D array of shape [N, D]."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    return {label: _scalar_stats(arr[:, idx]) for idx, label in enumerate(labels)}


def _safe_corrcoef(values):
    """Correlation matrix with NaNs collapsed to zero for constant channels."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] < 2:
        return np.full((arr.shape[1], arr.shape[1]), np.nan, dtype=np.float64)
    corr = np.corrcoef(arr, rowvar=False)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_cross_corrcoef(lhs, rhs):
    """Cross-correlation block between two multivariate signals."""
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.ndim == 1:
        lhs = lhs[:, None]
    if rhs.ndim == 1:
        rhs = rhs[:, None]
    if lhs.shape[0] < 2 or rhs.shape[0] < 2:
        return np.full((lhs.shape[1], rhs.shape[1]), np.nan, dtype=np.float64)
    merged = np.concatenate([lhs, rhs], axis=1)
    corr = np.corrcoef(merged, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr[:lhs.shape[1], lhs.shape[1]:]


def _rotation_matrix_to_euler_batch(R):
    """ZYX Euler angles from rotation matrices (body -> inertial)."""
    R = np.asarray(R, dtype=np.float64)
    theta = np.arcsin(np.clip(-R[:, 2, 0], -1.0, 1.0))
    ctheta = np.cos(theta)
    phi = np.empty(len(R), dtype=np.float64)
    psi = np.empty(len(R), dtype=np.float64)

    regular = np.abs(ctheta) > 1e-8
    phi[regular] = np.arctan2(R[regular, 2, 1], R[regular, 2, 2])
    psi[regular] = np.arctan2(R[regular, 1, 0], R[regular, 0, 0])

    phi[~regular] = np.arctan2(-R[~regular, 1, 2], R[~regular, 1, 1])
    psi[~regular] = 0.0
    return np.stack([phi, theta, psi], axis=1)


def _scenario_counts(meta_list):
    counts = defaultdict(int)
    for meta in meta_list:
        counts[meta.get('scenario', 'UNKNOWN')] += 1
    return dict(sorted(counts.items()))


def _coverage_vs_limits(values, limits, labels):
    """Fraction of samples exceeding reference fractions of the configured limits."""
    thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    arr = np.asarray(values, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    payload = {}
    for idx, label in enumerate(labels):
        abs_values = np.abs(arr[:, idx])
        limit = max(abs(float(limits[idx])), 1e-8)
        payload[label] = {
            f'gt_{int(frac * 100):02d}pct_limit': float(np.mean(abs_values > frac * limit))
            for frac in thresholds
        }
    return payload


def _add_histograms(npz_payload, prefix, values, labels, bins=64):
    """Store per-channel histograms in a compressed NPZ sidecar."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]

    counts = []
    edges = []
    for idx in range(arr.shape[1]):
        hist, edge = np.histogram(arr[:, idx], bins=bins)
        counts.append(hist.astype(np.int64))
        edges.append(edge.astype(np.float64))

    npz_payload[f'{prefix}_labels'] = np.asarray(labels)
    npz_payload[f'{prefix}_counts'] = np.stack(counts, axis=0)
    npz_payload[f'{prefix}_edges'] = np.stack(edges, axis=0)


def _compute_split_statistics(blocks, config, split_name):
    """Compute persistent summary statistics for one block split."""
    blocks = np.asarray(blocks, dtype=np.float64)
    if blocks.size == 0:
        return {
            'split': split_name,
            'n_blocks': 0,
            'n_points': 0,
        }, {}

    flat = blocks.reshape(-1, blocks.shape[-1])
    R = flat[:, 3:12].reshape(-1, 3, 3)
    euler = _rotation_matrix_to_euler_batch(R)
    g_body = np.transpose(R, (0, 2, 1))[:, :, 2]

    nu_b = flat[:, 12:18]
    u_actual = flat[:, 18:21]
    u_cmd = flat[:, 21:24]
    actuator_tracking_error = u_cmd - u_actual

    linear_speed = np.linalg.norm(nu_b[:, :3], axis=1)
    angular_speed = np.linalg.norm(nu_b[:, 3:6], axis=1)
    block_dpos = blocks[:, -1, :3] - blocks[:, 0, :3]
    block_dnu_b = blocks[:, -1, 12:18] - blocks[:, 0, 12:18]
    block_du_actual = blocks[:, -1, 18:21] - blocks[:, 0, 18:21]

    payload = {
        'split': split_name,
        'n_blocks': int(len(blocks)),
        'n_points': int(len(flat)),
        'position_rel': _matrix_stats(flat[:, :3], POS_LABELS),
        'euler': _matrix_stats(euler, EULER_LABELS),
        'g_body': _matrix_stats(g_body, G_BODY_LABELS),
        'nu_b': _matrix_stats(nu_b, TWIST_LABELS),
        'u_actual': _matrix_stats(u_actual, ACTUATOR_LABELS),
        'u_cmd': _matrix_stats(u_cmd, COMMAND_LABELS),
        'u_tracking_error': _matrix_stats(actuator_tracking_error, ACTUATOR_LABELS),
        'speed_norms': {
            'linear_body': _scalar_stats(linear_speed),
            'angular_body': _scalar_stats(angular_speed),
        },
        'block_delta': {
            'position_rel': _matrix_stats(block_dpos, POS_LABELS),
            'position_rel_norm': _scalar_stats(np.linalg.norm(block_dpos, axis=1)),
            'nu_b': _matrix_stats(block_dnu_b, TWIST_LABELS),
            'u_actual': _matrix_stats(block_du_actual, ACTUATOR_LABELS),
        },
        'coverage_vs_velocity_limits': {
            'nu_b': _coverage_vs_limits(nu_b, config.velocity_max, TWIST_LABELS),
        },
        'correlations': {
            'nu_b': {
                'labels': TWIST_LABELS,
                'matrix': _safe_corrcoef(nu_b).tolist(),
            },
            'u_cmd_to_nu_b': {
                'row_labels': COMMAND_LABELS,
                'col_labels': TWIST_LABELS,
                'matrix': _safe_cross_corrcoef(u_cmd, nu_b).tolist(),
            },
            'u_actual_to_nu_b': {
                'row_labels': ACTUATOR_LABELS,
                'col_labels': TWIST_LABELS,
                'matrix': _safe_cross_corrcoef(u_actual, nu_b).tolist(),
            },
        },
    }

    npz_payload = {}
    _add_histograms(npz_payload, f'{split_name}_position_rel', flat[:, :3], POS_LABELS)
    _add_histograms(npz_payload, f'{split_name}_euler', euler, EULER_LABELS)
    _add_histograms(npz_payload, f'{split_name}_nu_b', nu_b, TWIST_LABELS)
    _add_histograms(npz_payload, f'{split_name}_u_actual', u_actual, ACTUATOR_LABELS)
    _add_histograms(npz_payload, f'{split_name}_u_cmd', u_cmd, COMMAND_LABELS)
    _add_histograms(npz_payload, f'{split_name}_u_tracking_error', actuator_tracking_error, ACTUATOR_LABELS)
    _add_histograms(npz_payload, f'{split_name}_speed_norms', np.stack([linear_speed, angular_speed], axis=1), NORM_LABELS)

    npz_payload[f'{split_name}_corr_nu_b'] = _safe_corrcoef(nu_b)
    npz_payload[f'{split_name}_corr_u_cmd_to_nu_b'] = _safe_cross_corrcoef(u_cmd, nu_b)
    npz_payload[f'{split_name}_corr_u_actual_to_nu_b'] = _safe_cross_corrcoef(u_actual, nu_b)

    if config.absolute_depth_context:
        z_ref = flat[:, -1]
        payload['z_ref'] = _scalar_stats(z_ref)
        _add_histograms(npz_payload, f'{split_name}_z_ref', z_ref, ['z_ref'])

    if config.ocean_current:
        v_c_n = flat[:, 24:27]
        v_c_b = np.einsum('nij,nj->ni', np.transpose(R, (0, 2, 1)), v_c_n)
        nu_r = nu_b.copy()
        nu_r[:, :3] -= v_c_b
        current_speed = np.linalg.norm(v_c_n, axis=1)

        payload['nu_r'] = _matrix_stats(nu_r, TWIST_LABELS)
        payload['v_c_n'] = _matrix_stats(v_c_n, CURRENT_LABELS)
        payload['v_c_b'] = _matrix_stats(v_c_b, CURRENT_BODY_LABELS)
        payload['speed_norms']['current_inertial'] = _scalar_stats(current_speed)
        payload['coverage_vs_velocity_limits']['nu_r'] = _coverage_vs_limits(
            nu_r, config.velocity_max, TWIST_LABELS,
        )
        payload['correlations']['nu_r'] = {
            'labels': TWIST_LABELS,
            'matrix': _safe_corrcoef(nu_r).tolist(),
        }
        payload['correlations']['u_cmd_to_nu_r'] = {
            'row_labels': COMMAND_LABELS,
            'col_labels': TWIST_LABELS,
            'matrix': _safe_cross_corrcoef(u_cmd, nu_r).tolist(),
        }
        payload['correlations']['u_actual_to_nu_r'] = {
            'row_labels': ACTUATOR_LABELS,
            'col_labels': TWIST_LABELS,
            'matrix': _safe_cross_corrcoef(u_actual, nu_r).tolist(),
        }

        _add_histograms(npz_payload, f'{split_name}_nu_r', nu_r, TWIST_LABELS)
        _add_histograms(npz_payload, f'{split_name}_v_c_n', v_c_n, CURRENT_LABELS)
        _add_histograms(npz_payload, f'{split_name}_v_c_b', v_c_b, CURRENT_BODY_LABELS)
        _add_histograms(npz_payload, f'{split_name}_current_speed', current_speed, ['current_speed'])
        npz_payload[f'{split_name}_corr_nu_r'] = _safe_corrcoef(nu_r)
        npz_payload[f'{split_name}_corr_u_cmd_to_nu_r'] = _safe_cross_corrcoef(u_cmd, nu_r)
        npz_payload[f'{split_name}_corr_u_actual_to_nu_r'] = _safe_cross_corrcoef(u_actual, nu_r)

    return payload, npz_payload


def _build_dataset_statistics(dataset: Dict, config: Config, metadata: Dict):
    """Build persistent dataset statistics for paper reporting."""
    train_blocks = dataset['train_blocks']
    test_blocks = dataset['test_blocks']
    if len(train_blocks) and len(test_blocks):
        full_blocks = np.concatenate([train_blocks, test_blocks], axis=0)
    elif len(train_blocks):
        full_blocks = train_blocks
    else:
        full_blocks = test_blocks

    stats_json = {
        'dataset_id': metadata['dataset_id'],
        'schema': metadata['schema'],
        'velocity_convention': metadata['velocity_convention'],
        'frame_convention': metadata['frame_convention'],
        'derived_fields': metadata['derived_fields'],
        'scenario_counts': {
            'train': _scenario_counts(dataset.get('train_meta', [])),
            'test': _scenario_counts(dataset.get('test_meta', [])),
        },
        'splits': {},
        'by_scenario': {
            'train': {},
            'test': {},
        },
    }
    full_counts = defaultdict(int)
    for split_name in ['train', 'test']:
        for scenario_name, count in stats_json['scenario_counts'][split_name].items():
            full_counts[scenario_name] += int(count)
    stats_json['scenario_counts']['full'] = dict(sorted(full_counts.items()))

    npz_payload = {}
    for split_name, blocks in [
        ('full', full_blocks),
        ('train', train_blocks),
        ('test', test_blocks),
    ]:
        split_stats, split_npz = _compute_split_statistics(blocks, config, split_name)
        stats_json['splits'][split_name] = split_stats
        npz_payload.update(split_npz)

    for split_name, trajectories_key, meta_key in [
        ('train', 'train_trajectories', 'train_meta'),
        ('test', 'test_trajectories', 'test_meta'),
    ]:
        trajectories = dataset.get(trajectories_key)
        meta_list = dataset.get(meta_key, [])
        if trajectories is None or len(trajectories) == 0:
            continue

        scenario_to_indices = defaultdict(list)
        for idx, meta in enumerate(meta_list):
            scenario_to_indices[meta.get('scenario', 'UNKNOWN')].append(idx)

        for scenario_name, indices in sorted(scenario_to_indices.items()):
            scenario_blocks = trajectories[np.asarray(indices, dtype=np.int64)].reshape(
                -1, config.points_per_block, trajectories.shape[-1]
            )
            scenario_stats, _ = _compute_split_statistics(
                scenario_blocks, config, f'{split_name}_{scenario_name.lower()}'
            )
            stats_json['by_scenario'][split_name][scenario_name] = scenario_stats

    full_stats = stats_json['splits']['full']
    stats_json['paper_summary'] = {
        'n_train_blocks': int(metadata['num_train_blocks']),
        'n_test_blocks': int(metadata['num_test_blocks']),
        'n_total_points': int(full_stats.get('n_points', 0)),
        'scenario_counts_full': stats_json['scenario_counts']['full'],
        'linear_speed_body': full_stats.get('speed_norms', {}).get('linear_body', {}),
        'angular_speed_body': full_stats.get('speed_norms', {}).get('angular_body', {}),
        'nu_b_std': {
            label: full_stats.get('nu_b', {}).get(label, {}).get('std', float('nan'))
            for label in TWIST_LABELS
        },
    }
    if config.ocean_current:
        stats_json['paper_summary']['current_speed_inertial'] = full_stats.get(
            'speed_norms', {}
        ).get('current_inertial', {})
        stats_json['paper_summary']['nu_r_std'] = {
            label: full_stats.get('nu_r', {}).get(label, {}).get('std', float('nan'))
            for label in TWIST_LABELS
        }

    return stats_json, npz_payload


# =============================================================================
# Plotting
# =============================================================================

def _get_plot_module():
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = Path.home() / ".matplotlib"
        if not (cache_dir.exists() and os.access(cache_dir, os.W_OK)):
            fallback = Path(tempfile.gettempdir()) / "matplotlib-cache"
            fallback.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(fallback)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    return plt


def _mm_to_inches(mm):
    return float(mm) / 25.4


def _publication_rc_params():
    """Elsevier-compatible typography with restrained Nature-like styling."""
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.0,
        'axes.titlesize': 7.5,
        'axes.labelsize': 7.0,
        'xtick.labelsize': 6.0,
        'ytick.labelsize': 6.0,
        'legend.fontsize': 6.0,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.linewidth': 0.6,
        'lines.linewidth': 1.2,
        'lines.markersize': 3.0,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
    }


def _save_publication_figure(fig, output_path):
    """Save a figure as both vector PDF and high-resolution PNG."""
    pdf_path = output_path.with_suffix('.pdf')
    png_path = output_path.with_suffix('.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=500)
    return [str(pdf_path), str(png_path)]


def _render_hist_figure(npz_payload, prefix, labels, title, output_path):
    """Render a multi-panel histogram figure from cached histogram arrays."""
    plt = _get_plot_module()
    if plt is None:
        return False

    counts = np.asarray(npz_payload.get(f'{prefix}_counts'))
    edges = np.asarray(npz_payload.get(f'{prefix}_edges'))
    if counts.size == 0 or edges.size == 0:
        return False

    n_channels = len(labels)
    ncols = 3 if n_channels > 3 else n_channels
    nrows = int(np.ceil(n_channels / ncols))
    with plt.rc_context(_publication_rc_params()):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(_mm_to_inches(190), _mm_to_inches(38 * nrows)),
            squeeze=False,
        )
        flat_axes = axes.reshape(-1)

        for idx, label in enumerate(labels):
            ax = flat_axes[idx]
            ax.stairs(counts[idx], edges[idx], color='#4477AA', linewidth=1.2, fill=True, alpha=0.24)
            ax.set_title(label)
            ax.set_ylabel('count')
            ax.grid(True, alpha=0.18, linewidth=0.4)

        for idx in range(n_channels, len(flat_axes)):
            flat_axes[idx].axis('off')

        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


def _render_matrix_heatmap(ax, matrix, row_labels, col_labels, title, cmap='coolwarm', vmin=-1.0, vmax=1.0):
    """Draw a labeled heatmap with numeric annotations."""
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = 'white' if np.isfinite(value) and abs(value) > 0.55 else 'black'
            ax.text(col, row, f'{value:.2f}', ha='center', va='center', color=text_color, fontsize=5.5)
    return im


def _render_correlation_figure(stats_json, npz_payload, output_path):
    """Render key correlation heatmaps for velocity and actuation channels."""
    plt = _get_plot_module()
    if plt is None:
        return False

    current_enabled = 'full_corr_nu_r' in npz_payload
    ncols = 2
    nrows = 2 if current_enabled else 1
    with plt.rc_context(_publication_rc_params()):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(_mm_to_inches(190), _mm_to_inches(46 * nrows)),
            squeeze=False,
        )

        im = _render_matrix_heatmap(
            axes[0, 0],
            np.asarray(npz_payload['full_corr_nu_b']),
            TWIST_LABELS,
            TWIST_LABELS,
            'nu_b correlation',
        )
        _render_matrix_heatmap(
            axes[0, 1],
            np.asarray(npz_payload['full_corr_u_cmd_to_nu_b']),
            COMMAND_LABELS,
            TWIST_LABELS,
            'u_cmd vs nu_b',
        )

        if current_enabled:
            _render_matrix_heatmap(
                axes[1, 0],
                np.asarray(npz_payload['full_corr_nu_r']),
                TWIST_LABELS,
                TWIST_LABELS,
                'nu_r correlation',
            )
            _render_matrix_heatmap(
                axes[1, 1],
                np.asarray(npz_payload['full_corr_u_cmd_to_nu_r']),
                COMMAND_LABELS,
                TWIST_LABELS,
                'u_cmd vs nu_r',
            )

        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.02, label='corr.')
        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


def _render_scenario_summary_figure(stats_json, output_path):
    """Render scenario counts plus per-scenario DOF coverage heatmaps."""
    plt = _get_plot_module()
    if plt is None:
        return False

    train_counts = stats_json['scenario_counts'].get('train', {})
    test_counts = stats_json['scenario_counts'].get('test', {})
    scenario_names = sorted(set(train_counts) | set(test_counts))
    if not scenario_names:
        return False

    current_enabled = bool(stats_json.get('splits', {}).get('full', {}).get('nu_r'))
    with plt.rc_context(_publication_rc_params()):
        fig, axes = plt.subplots(
            1,
            3 if current_enabled else 2,
            figsize=(_mm_to_inches(190), _mm_to_inches(54)),
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes])

        x = np.arange(len(scenario_names))
        width = 0.38
        axes[0].bar(x - width / 2, [train_counts.get(name, 0) for name in scenario_names], width=width, label='train', color='#4477AA')
        axes[0].bar(x + width / 2, [test_counts.get(name, 0) for name in scenario_names], width=width, label='test', color='#CC6677')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(scenario_names)
        axes[0].set_ylabel('trajectory count')
        axes[0].set_title('scenario counts')
        axes[0].legend(frameon=False)
        axes[0].grid(True, axis='y', alpha=0.18, linewidth=0.4)

        def _scenario_std_matrix(split_name, field_name):
            split_payload = stats_json['by_scenario'].get(split_name, {})
            labels = sorted(split_payload)
            matrix = np.array([
                [split_payload[scenario][field_name][label]['std'] for label in TWIST_LABELS]
                for scenario in labels
            ], dtype=np.float64)
            return labels, matrix

        train_labels, train_nu_b = _scenario_std_matrix('train', 'nu_b')
        im = _render_matrix_heatmap(
            axes[1],
            train_nu_b,
            train_labels,
            TWIST_LABELS,
            'train scenario nu_b std',
            cmap='viridis',
            vmin=float(np.nanmin(train_nu_b)),
            vmax=float(np.nanmax(train_nu_b)),
        )
        if current_enabled:
            _, train_nu_r = _scenario_std_matrix('train', 'nu_r')
            _render_matrix_heatmap(
                axes[2],
                train_nu_r,
                train_labels,
                TWIST_LABELS,
                'train scenario nu_r std',
                cmap='viridis',
                vmin=float(np.nanmin(train_nu_r)),
                vmax=float(np.nanmax(train_nu_r)),
            )

        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.02, label='std')
        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


PAPER_COLORS = {
    'stored': '#4477AA',
    'model': '#228833',
    'current': '#66CCEE',
    'command': '#CC6677',
    'actuation': '#AA3377',
    'tracking': '#EE7733',
}


def _add_panel_header(ax, panel_label, title):
    ax.text(
        -0.22,
        1.18,
        panel_label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight='bold',
        va='bottom',
        ha='left',
    )
    ax.text(
        -0.02,
        1.18,
        title,
        transform=ax.transAxes,
        fontsize=7.5,
        fontweight='bold',
        va='bottom',
        ha='left',
    )


def _draw_hist_group(fig, parent_spec, npz_payload, prefix, labels, panel_label, title, color):
    """Draw a grouped histogram panel with an internal subplot grid."""
    counts = np.asarray(npz_payload.get(f'{prefix}_counts'))
    edges = np.asarray(npz_payload.get(f'{prefix}_edges'))
    if counts.size == 0 or edges.size == 0:
        return []

    n_channels = len(labels)
    if n_channels <= 3:
        nrows, ncols = 1, n_channels
    else:
        nrows, ncols = 2, 3

    subgrid = parent_spec.subgridspec(nrows, ncols, hspace=0.42, wspace=0.28)
    axes = []
    for idx, label in enumerate(labels):
        ax = fig.add_subplot(subgrid[idx // ncols, idx % ncols])
        ax.stairs(counts[idx], edges[idx], color=color, linewidth=1.1, fill=True, alpha=0.25)
        ax.set_title(label, pad=1.5)
        ax.grid(True, alpha=0.18, linewidth=0.4)
        if idx % ncols == 0:
            ax.set_ylabel('count')
        if idx // ncols == nrows - 1:
            ax.set_xlabel('value')
        axes.append(ax)

    for idx in range(n_channels, nrows * ncols):
        ax = fig.add_subplot(subgrid[idx // ncols, idx % ncols])
        ax.axis('off')

    _add_panel_header(axes[0], panel_label, title)
    return axes


def _render_paper_velocity_figure(stats_json, npz_payload, output_path):
    """Main paper figure for stored/model velocity spaces and current."""
    plt = _get_plot_module()
    if plt is None:
        return []

    current_enabled = 'full_nu_r_counts' in npz_payload
    with plt.rc_context(_publication_rc_params()):
        width = 190
        height = 86 if current_enabled else 72
        fig = plt.figure(figsize=(_mm_to_inches(width), _mm_to_inches(height)))
        outer = fig.add_gridspec(
            1,
            3 if current_enabled else 2,
            width_ratios=[2.2, 2.2, 1.3] if current_enabled else [2.6, 1.2],
            wspace=0.28,
        )

        _draw_hist_group(
            fig, outer[0], npz_payload, 'full_nu_b', TWIST_LABELS,
            panel_label='A', title='Stored body velocity $\\nu_b$', color=PAPER_COLORS['stored'],
        )

        if current_enabled:
            _draw_hist_group(
                fig, outer[1], npz_payload, 'full_nu_r', TWIST_LABELS,
                panel_label='B', title='Model-space relative velocity $\\nu_r$', color=PAPER_COLORS['model'],
            )
            _draw_hist_group(
                fig, outer[2], npz_payload, 'full_v_c_n', CURRENT_LABELS,
                panel_label='C', title='Inertial current $v_c^n$', color=PAPER_COLORS['current'],
            )
        else:
            _draw_hist_group(
                fig, outer[1], npz_payload, 'full_speed_norms', NORM_LABELS,
                panel_label='B', title='Body speed norms', color=PAPER_COLORS['stored'],
            )

        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


def _render_paper_controls_figure(stats_json, npz_payload, output_path):
    """Main paper figure for actuation and command channels."""
    plt = _get_plot_module()
    if plt is None:
        return []

    with plt.rc_context(_publication_rc_params()):
        fig = plt.figure(figsize=(_mm_to_inches(190), _mm_to_inches(56)))
        outer = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.32)

        _draw_hist_group(
            fig, outer[0], npz_payload, 'full_u_actual', ACTUATOR_LABELS,
            panel_label='A', title='Actuator state $u_{act}$', color=PAPER_COLORS['actuation'],
        )
        _draw_hist_group(
            fig, outer[1], npz_payload, 'full_u_cmd', COMMAND_LABELS,
            panel_label='B', title='Command channel $u_{cmd}$', color=PAPER_COLORS['command'],
        )
        _draw_hist_group(
            fig, outer[2], npz_payload, 'full_u_tracking_error', ACTUATOR_LABELS,
            panel_label='C', title='Tracking error $u_{cmd} - u_{act}$', color=PAPER_COLORS['tracking'],
        )

        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


def _render_paper_correlation_figure(stats_json, npz_payload, output_path):
    """Main paper figure for correlation structure and control-response coupling."""
    plt = _get_plot_module()
    if plt is None:
        return []

    current_enabled = 'full_corr_nu_r' in npz_payload
    nrows = 2 if current_enabled else 1
    with plt.rc_context(_publication_rc_params()):
        fig, axes = plt.subplots(
            nrows, 2,
            figsize=(_mm_to_inches(190), _mm_to_inches(44 * nrows))
        )
        axes = np.asarray(axes).reshape(nrows, 2)

        im = _render_matrix_heatmap(
            axes[0, 0], np.asarray(npz_payload['full_corr_nu_b']),
            TWIST_LABELS, TWIST_LABELS, 'Stored velocity correlation'
        )
        _add_panel_header(axes[0, 0], 'A', 'Stored velocity structure')
        _render_matrix_heatmap(
            axes[0, 1], np.asarray(npz_payload['full_corr_u_cmd_to_nu_b']),
            COMMAND_LABELS, TWIST_LABELS, 'Command-response correlation'
        )
        _add_panel_header(axes[0, 1], 'B', 'Control-response in stored space')

        if current_enabled:
            _render_matrix_heatmap(
                axes[1, 0], np.asarray(npz_payload['full_corr_nu_r']),
                TWIST_LABELS, TWIST_LABELS, 'Model-space velocity correlation'
            )
            _add_panel_header(axes[1, 0], 'C', 'Model-space velocity structure')
            _render_matrix_heatmap(
                axes[1, 1], np.asarray(npz_payload['full_corr_u_cmd_to_nu_r']),
                COMMAND_LABELS, TWIST_LABELS, 'Command-response correlation'
            )
            _add_panel_header(axes[1, 1], 'D', 'Control-response in model space')

        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.02, label='corr.')
        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


def _render_paper_coverage_figure(stats_json, output_path):
    """Main paper figure for scenario balance and excitation coverage."""
    plt = _get_plot_module()
    if plt is None:
        return []

    train_counts = stats_json['scenario_counts'].get('train', {})
    test_counts = stats_json['scenario_counts'].get('test', {})
    scenario_names = sorted(set(train_counts) | set(test_counts))
    if not scenario_names:
        return []

    current_enabled = bool(stats_json.get('splits', {}).get('full', {}).get('nu_r'))
    with plt.rc_context(_publication_rc_params()):
        fig, axes = plt.subplots(
            1, 3 if current_enabled else 2,
            figsize=(_mm_to_inches(190), _mm_to_inches(56))
        )
        axes = np.asarray(axes).reshape(-1)

        x = np.arange(len(scenario_names))
        width = 0.38
        axes[0].bar(x - width / 2, [train_counts.get(name, 0) for name in scenario_names], width=width, label='train', color=PAPER_COLORS['stored'])
        axes[0].bar(x + width / 2, [test_counts.get(name, 0) for name in scenario_names], width=width, label='test', color=PAPER_COLORS['command'])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(scenario_names)
        axes[0].set_ylabel('trajectory count')
        axes[0].legend(frameon=False)
        axes[0].grid(True, axis='y', alpha=0.18, linewidth=0.4)
        _add_panel_header(axes[0], 'A', 'Scenario balance')

        def _scenario_std_matrix(split_name, field_name):
            split_payload = stats_json['by_scenario'].get(split_name, {})
            labels = sorted(split_payload)
            matrix = np.array([
                [split_payload[scenario][field_name][label]['std'] for label in TWIST_LABELS]
                for scenario in labels
            ], dtype=np.float64)
            return labels, matrix

        train_labels, train_nu_b = _scenario_std_matrix('train', 'nu_b')
        im = _render_matrix_heatmap(
            axes[1], train_nu_b, train_labels, TWIST_LABELS,
            'Train split std'
        )
        _add_panel_header(axes[1], 'B', 'Stored-space excitation')

        if current_enabled:
            _, train_nu_r = _scenario_std_matrix('train', 'nu_r')
            _render_matrix_heatmap(
                axes[2], train_nu_r, train_labels, TWIST_LABELS,
                'Train split std'
            )
            _add_panel_header(axes[2], 'C', 'Model-space excitation')

        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.024, pad=0.02, label='std')
        paths = _save_publication_figure(fig, output_path)
        plt.close(fig)
    return paths


def _paper_figure_manifest(stats_json, output_dir, figure_records):
    """Build figure manifest and caption draft for the paper figure pack."""
    current_enabled = bool(stats_json.get('splits', {}).get('full', {}).get('nu_r'))
    captions = {
        'fig01_velocity_spaces': (
            "Distribution of the velocity variables used in the dataset and in the model. "
            "Panel A shows the stored body-frame total velocity $\\nu_b$. "
            + (
                "Panel B shows the model-space relative velocity $\\nu_r$ obtained from the exact transform "
                "$\\nu_r = \\nu_b - [R^T v_c^n; 0]$, and Panel C shows the inertial current vector $v_c^n$. "
                if current_enabled else
                "Panel B shows body-speed norms derived from $\\nu_b$. "
            )
            + "These panels make the difference between storage-space and model-space statistics explicit."
        ),
        'fig02_controls_actuation': (
            "Distribution of actuation-related channels. "
            "Panel A shows the actuator state $u_{act}$, Panel B shows the commanded input $u_{cmd}$, "
            "and Panel C shows the tracking error $u_{cmd} - u_{act}$. "
            "These panels quantify command coverage and the scale of first-order actuator lag present in the training data."
        ),
        'fig03_correlation_structure': (
            "Correlation structure of the dataset and command-response coupling. "
            "Stored-space correlations are shown in the top row. "
            + (
                "Model-space correlations after transforming to $\\nu_r$ are shown in the bottom row. "
                if current_enabled else
                ""
            )
            + "This figure highlights both cross-DOF coupling and how control signals align with the dominant response channels."
        ),
        'fig04_scenario_coverage': (
            "Scenario balance and excitation coverage across training scenarios. "
            "Panel A shows trajectory counts for train/test splits. "
            "The remaining panels summarize per-scenario standard deviations across the six body DOFs"
            + (
                " in stored space and model space, respectively."
                if current_enabled else
                "."
            )
        ),
    }

    manifest = []
    for stem, title in [
        ('fig01_velocity_spaces', 'Velocity spaces'),
        ('fig02_controls_actuation', 'Controls and actuation'),
        ('fig03_correlation_structure', 'Correlation structure'),
        ('fig04_scenario_coverage', 'Scenario coverage'),
    ]:
        manifest.append({
            'id': stem,
            'title': title,
            'files': figure_records.get(stem, []),
            'caption': captions[stem],
        })
    return manifest


def _write_paper_caption_files(output_dir, manifest):
    """Write a manifest JSON and a Markdown caption draft."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / 'figure_manifest.json'
    captions_path = output_dir / 'captions.md'

    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    lines = ["# Dataset Figure Captions", ""]
    for item in manifest:
        lines.append(f"## {item['id']} {item['title']}")
        lines.append("")
        lines.append(item['caption'])
        lines.append("")
        lines.append("Files:")
        for path in item['files']:
            lines.append(f"- {path}")
        lines.append("")

    with open(captions_path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return [str(manifest_path), str(captions_path)]


def _save_dataset_figures(stats_json, npz_payload, output_dir: Path):
    """Render the final paper-oriented figure pack and caption draft."""
    plt = _get_plot_module()
    if plt is None:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_records = {
        'fig01_velocity_spaces': _render_paper_velocity_figure(stats_json, npz_payload, output_dir / 'fig01_velocity_spaces'),
        'fig02_controls_actuation': _render_paper_controls_figure(stats_json, npz_payload, output_dir / 'fig02_controls_actuation'),
        'fig03_correlation_structure': _render_paper_correlation_figure(stats_json, npz_payload, output_dir / 'fig03_correlation_structure'),
        'fig04_scenario_coverage': _render_paper_coverage_figure(stats_json, output_dir / 'fig04_scenario_coverage'),
    }
    manifest = _paper_figure_manifest(stats_json, output_dir, figure_records)
    written = []
    for paths in figure_records.values():
        written.extend(paths)
    written.extend(_write_paper_caption_files(output_dir, manifest))
    return written


def export_dataset_figures_from_stats(
    stats_json_path,
    stats_npz_path=None,
    output_dir=None,
):
    """Export publication-style figures from existing dataset statistics sidecars."""
    stats_json_path = Path(stats_json_path)
    stem = stats_json_path.name
    if stem.endswith('.stats.json'):
        stem = stem[:-len('.stats.json')]
    else:
        stem = stats_json_path.stem
    if stats_npz_path is None:
        stats_npz_path = stats_json_path.parent / f'{stem}.stats.npz'
    stats_npz_path = Path(stats_npz_path)

    with open(stats_json_path, 'r', encoding='utf-8') as handle:
        stats_json = json.load(handle)

    npz_file = np.load(stats_npz_path, allow_pickle=False)
    npz_payload = {key: npz_file[key] for key in npz_file.files}
    npz_file.close()

    if output_dir is None:
        output_dir = stats_json_path.parent / f'{stem}_figures'
    output_dir = Path(output_dir)

    written = _save_dataset_figures(stats_json, npz_payload, output_dir)
    return {
        'stats_json': str(stats_json_path),
        'stats_npz': str(stats_npz_path),
        'output_dir': str(output_dir),
        'files': written,
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def generate_dataset(config=None, verbose=True):
    """
    Main dataset generation pipeline.

    Returns dict with keys:
      train_trajectories -- ndarray (N_train_traj, B, 5, 24)
      test_trajectories  -- ndarray (N_test_traj,  B, 5, 24)
      train_blocks       -- ndarray (N_train_blocks, 5, 24)
      test_blocks        -- ndarray (N_test_blocks,  5, 24)
      t_eval             -- ndarray (5,) time points within a block
      config             -- metadata dict
    """
    if config is None:
        config = Config()
    np.random.seed(config.seed)

    if verbose:
        print("=" * 60)
        print("AUVHamNODE Data Collection")
        print("=" * 60)
        print(f"Trajectories: {config.num_trajectories}, "
              f"blocks/traj: {config.blocks_per_trajectory}")
        print(f"Timing: dt_sim={config.dt_sim}s, dt_state={config.dt_state}s, "
              f"dt_ctrl={config.dt_ctrl}s")
        print(f"Points/block: {config.points_per_block}")
        print("Generating trajectories...")

    target = config.num_trajectories
    results_by_seed = {}

    # Submit 2x target to account for diverged trajectories
    seeds = list(range(config.seed, config.seed + target * 2))

    if config.num_workers <= 1:
        for seed in seeds:
            try:
                result = simulate_trajectory(seed, config)
            except Exception:
                result = None
            results_by_seed[seed] = result
            valid_count = sum(item is not None for item in results_by_seed.values())
            if verbose and valid_count > 0 and valid_count % 50 == 0:
                print(f"  Progress: {valid_count}/{target}")
            if valid_count >= target:
                break
    else:
        executor = ProcessPoolExecutor(max_workers=config.num_workers)
        futures = {executor.submit(simulate_trajectory, s, config): s
                   for s in seeds}

        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
            except Exception:
                result = None
            results_by_seed[seed] = result
            valid_count = sum(item is not None for item in results_by_seed.values())
            if verbose and valid_count > 0 and valid_count % 50 == 0:
                print(f"  Progress: {valid_count}/{target}")

        executor.shutdown(wait=False, cancel_futures=True)

    ordered_results = [results_by_seed.get(seed) for seed in seeds]
    results = [result for result in ordered_results if result is not None][:target]

    if verbose:
        achieved = len(results)
        print(f"  Completed: {achieved} trajectories"
              + (f" (target was {target})" if achieved < target else ""))

    if not results:
        raise RuntimeError("No valid trajectories were generated.")

    # Filter
    if config.enable_filtering and results:
        kept_records = filter_trajectories(results, config, verbose)
    else:
        kept_records = results

    if not kept_records:
        raise RuntimeError("All generated trajectories were filtered out.")

    # Split by trajectory before flattening to blocks.
    idx = np.random.permutation(len(kept_records))
    shuffled_records = [kept_records[i] for i in idx]
    split = int(len(shuffled_records) * config.train_ratio)
    train_records = shuffled_records[:split]
    test_records = shuffled_records[split:]

    dataset_id = _dataset_id(config, train_records, test_records)

    train_trajectories = _stack_trajectories(train_records, config)
    test_trajectories = _stack_trajectories(test_records, config)

    state_dim = config.state_dim
    train_blocks = train_trajectories.reshape(-1, config.points_per_block, state_dim)
    test_blocks = test_trajectories.reshape(-1, config.points_per_block, state_dim)

    t_eval = np.linspace(0, config.dt_ctrl, config.points_per_block)

    dataset = {
        'train_trajectories': train_trajectories,
        'test_trajectories': test_trajectories,
        'train_blocks': train_blocks,
        'test_blocks': test_blocks,
        'train_data': train_blocks,
        'test_data': test_blocks,
        'train_meta': [record['meta'] for record in train_records],
        'test_meta': [record['meta'] for record in test_records],
        't_eval': t_eval,
        'config': {
            'dataset_id': dataset_id,
            'schema': 'auv_dataset_v3',
            'dt_sim': config.dt_sim,
            'dt_state': config.dt_state,
            'dt_ctrl': config.dt_ctrl,
            'state_dim': state_dim,
            'u_dim': 3,
            'points_per_block': config.points_per_block,
            'blocks_per_trajectory': config.blocks_per_trajectory,
            'split_level': 'trajectory',
            'num_train_trajectories': len(train_records),
            'num_test_trajectories': len(test_records),
            'train_seeds': _record_seed_list(train_records),
            'test_seeds': _record_seed_list(test_records),
            'ocean_current': config.ocean_current,
            'absolute_depth_available': bool(config.absolute_depth_context),
            'depth_context_convention': (
                'block_start_depth' if config.absolute_depth_context else None
            ),
            'velocity_convention': 'body_total',
            'frame_convention': {
                'rotation': 'body_to_inertial',
                'position': 'inertial_relative',
                'twist': 'body_total',
                'current': 'inertial',
                'absolute_depth': (
                    'carried_block_start_depth' if config.absolute_depth_context else None
                ),
            },
            'derived_fields': (
                {
                    'nu_model': 'nu_r = nu_b - [R^T v_c^n; 0]',
                    **(
                        {'absolute_depth': 'z_abs = z_ref + Δz'}
                        if config.absolute_depth_context else {}
                    ),
                }
                if config.ocean_current else
                {
                    'nu_model': 'nu_b',
                    **(
                        {'absolute_depth': 'z_abs = z_ref + Δz'}
                        if config.absolute_depth_context else {}
                    ),
                }
            ),
            'description': (
                'State: [Dpos(3), R(9), nu_b(6), u_actual(3), u_cmd(3), v_c^n(3), z_ref(1)]'
                if config.ocean_current and config.absolute_depth_context else
                'State: [Dpos(3), R(9), nu_b(6), u_actual(3), u_cmd(3), v_c^n(3)]'
                if config.ocean_current else
                'State: [Dpos(3), R(9), nu_b(6), u_actual(3), u_cmd(3), z_ref(1)]'
                if config.absolute_depth_context else
                'State: [Dpos(3), R(9), nu_b(6), u_actual(3), u_cmd(3)]'
            ),
            'generation_config': serialize_config(config),
        },
    }

    if verbose:
        print(
            f"\nDataset: {len(train_records)} train traj / {len(test_records)} test traj"
        )
        print(f"Blocks: {len(train_blocks)} train / {len(test_blocks)} test")
        print(f"State dim: {state_dim}, time points/sample: {config.points_per_block}")
        print_statistics(train_blocks, "Training Data")

    return dataset


def _dataset_identity_payload(config: Config) -> Dict:
    """Return the stable configuration payload used to derive a dataset ID."""
    return {
        "schema": "auv_dataset_v3",
        "ocean_current": bool(config.ocean_current),
        "num_trajectories": int(config.num_trajectories),
        "blocks_per_trajectory": int(config.blocks_per_trajectory),
        "train_ratio": float(config.train_ratio),
        "dt_sim": float(config.dt_sim),
        "dt_state": float(config.dt_state),
        "dt_ctrl": float(config.dt_ctrl),
        "warmup_time": float(config.warmup_time),
        "enable_filtering": bool(config.enable_filtering),
        "min_excitation_percentile": float(config.min_excitation_percentile),
        "rotation_tolerance": float(config.rotation_tolerance),
        "absolute_depth_context": bool(config.absolute_depth_context),
        "current_speed_range": list(map(float, config.current_speed_range)),
        "current_direction_range": list(map(float, config.current_direction_range)),
        "current_vertical_std": float(config.current_vertical_std),
        "current_drift_sigma": float(config.current_drift_sigma),
        "current_drift_theta": float(config.current_drift_theta),
        "seed": int(config.seed),
    }


def _record_seed_list(records) -> list[int]:
    """Extract the realized trajectory seed list from dataset records."""
    return [int(record["meta"]["seed"]) for record in records]


def _dataset_realization_payload(
    config: Config,
    train_records=None,
    test_records=None,
) -> Dict:
    """Return the full payload that uniquely identifies a realized dataset."""
    payload = {
        "identity": _dataset_identity_payload(config),
    }
    if train_records is not None or test_records is not None:
        train_records = train_records or []
        test_records = test_records or []
        payload["train_seeds"] = _record_seed_list(train_records)
        payload["test_seeds"] = _record_seed_list(test_records)
    return payload


def _dataset_id(config: Config, train_records=None, test_records=None) -> str:
    """Build a stable short ID from config plus the realized trajectory seeds."""
    payload = _dataset_realization_payload(config, train_records, test_records)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:8]


def _dataset_stem(config: Config, dataset_id: Optional[str] = None) -> str:
    """Build a readable default dataset stem from key generation settings."""
    current_tag = "oc" if config.ocean_current else "noc"
    dataset_id = dataset_id or _dataset_id(config)
    return (
        f"auv_{current_tag}"
        f"_traj{config.num_trajectories}"
        f"_blk{config.blocks_per_trajectory}"
        f"_s{config.seed}"
        f"_{dataset_id}"
    )


def _build_dataset_metadata(dataset: Dict, config: Config, dataset_path: Path) -> Dict:
    """Build machine-readable metadata saved alongside the pickle file."""
    cfg = dataset["config"]
    train_blocks = dataset["train_blocks"]
    test_blocks = dataset["test_blocks"]
    train_meta = dataset.get("train_meta", [])
    test_meta = dataset.get("test_meta", [])
    dataset_id = str(cfg["dataset_id"])

    metadata = {
        "dataset_id": dataset_id,
        "dataset_file": dataset_path.name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "save_dir": str(dataset_path.parent),
        "seed": int(config.seed),
        "num_trajectories_requested": int(config.num_trajectories),
        "num_train_trajectories": int(cfg["num_train_trajectories"]),
        "num_test_trajectories": int(cfg["num_test_trajectories"]),
        "num_train_blocks": int(len(train_blocks)),
        "num_test_blocks": int(len(test_blocks)),
        "blocks_per_trajectory": int(config.blocks_per_trajectory),
        "points_per_block": int(cfg["points_per_block"]),
        "state_dim": int(cfg["state_dim"]),
        "u_dim": int(cfg["u_dim"]),
        "ocean_current": bool(cfg["ocean_current"]),
        "absolute_depth_available": bool(cfg.get("absolute_depth_available", False)),
        "depth_context_convention": cfg.get("depth_context_convention"),
        "schema": cfg["schema"],
        "velocity_convention": cfg["velocity_convention"],
        "frame_convention": cfg["frame_convention"],
        "derived_fields": cfg["derived_fields"],
        "description": cfg["description"],
        "dt_sim": float(cfg["dt_sim"]),
        "dt_state": float(cfg["dt_state"]),
        "dt_ctrl": float(cfg["dt_ctrl"]),
        "warmup_time": float(config.warmup_time),
        "enable_filtering": bool(config.enable_filtering),
        "min_excitation_percentile": float(config.min_excitation_percentile),
        "rotation_tolerance": float(config.rotation_tolerance),
        "train_ratio": float(config.train_ratio),
        "current_speed_range": list(map(float, config.current_speed_range)),
        "current_direction_range": list(map(float, config.current_direction_range)),
        "current_vertical_std": float(config.current_vertical_std),
        "current_drift_sigma": float(config.current_drift_sigma),
        "current_drift_theta": float(config.current_drift_theta),
        "num_workers": int(config.num_workers),
        "train_scenarios": sorted({meta.get("scenario", "UNKNOWN") for meta in train_meta}),
        "test_scenarios": sorted({meta.get("scenario", "UNKNOWN") for meta in test_meta}),
        "identity_payload": _dataset_identity_payload(config),
        "realization_payload": _dataset_realization_payload(
            config,
            train_records=[{"meta": meta} for meta in train_meta],
            test_records=[{"meta": meta} for meta in test_meta],
        ),
        "generation_config": cfg.get("generation_config", serialize_config(config)),
    }
    return metadata


def _build_dataset_summary_text(metadata: Dict, stats: Optional[Dict] = None) -> str:
    """Build a concise human-readable dataset summary."""
    lines = [
        "AUV Dataset Summary",
        "=" * 80,
        f"Dataset ID: {metadata['dataset_id']}",
        f"Schema: {metadata['schema']}",
        f"File: {metadata['dataset_file']}",
        f"Generated at: {metadata['generated_at']}",
        f"Ocean current: {metadata['ocean_current']}",
        f"Absolute depth context: {metadata['absolute_depth_available']}",
        f"Velocity convention: {metadata['velocity_convention']}",
        f"State dim: {metadata['state_dim']}",
        f"Actuator dim: {metadata['u_dim']}",
        f"Description: {metadata['description']}",
        f"Frame convention: {metadata['frame_convention']}",
        f"Derived fields: {metadata['derived_fields']}",
        "",
        "Dataset Size",
        "-" * 80,
        f"Requested trajectories: {metadata['num_trajectories_requested']}",
        f"Train trajectories: {metadata['num_train_trajectories']}",
        f"Test trajectories: {metadata['num_test_trajectories']}",
        f"Train blocks: {metadata['num_train_blocks']}",
        f"Test blocks: {metadata['num_test_blocks']}",
        f"Blocks per trajectory: {metadata['blocks_per_trajectory']}",
        f"Points per block: {metadata['points_per_block']}",
        "",
        "Timing",
        "-" * 80,
        f"dt_sim: {metadata['dt_sim']:.4f} s",
        f"dt_state: {metadata['dt_state']:.4f} s",
        f"dt_ctrl: {metadata['dt_ctrl']:.4f} s",
        f"warmup_time: {metadata['warmup_time']:.4f} s",
        "",
        "Generation Settings",
        "-" * 80,
        f"Seed: {metadata['seed']}",
        f"Filtering enabled: {metadata['enable_filtering']}",
        f"Min excitation percentile: {metadata['min_excitation_percentile']}",
        f"Rotation tolerance: {metadata['rotation_tolerance']}",
        f"Train ratio: {metadata['train_ratio']:.2f}",
        f"Workers: {metadata['num_workers']}",
        "",
        "Ocean Current Settings",
        "-" * 80,
        f"Current speed range: {metadata['current_speed_range']}",
        f"Current direction range: {metadata['current_direction_range']}",
        f"Current vertical std: {metadata['current_vertical_std']:.4f}",
        f"Current drift sigma: {metadata['current_drift_sigma']:.4f}",
        f"Current drift theta: {metadata['current_drift_theta']:.4f}",
        "",
        "Scenario Coverage",
        "-" * 80,
        f"Train scenarios: {', '.join(metadata['train_scenarios']) or 'n/a'}",
        f"Test scenarios: {', '.join(metadata['test_scenarios']) or 'n/a'}",
    ]
    if stats is not None:
        paper = stats.get('paper_summary', {})
        lines.extend([
            "",
            "Headline Statistics",
            "-" * 80,
            f"Total points: {paper.get('n_total_points', 0)}",
            f"Scenario counts (full): {paper.get('scenario_counts_full', {})}",
            f"Linear speed |nu_b[:3]|: {paper.get('linear_speed_body', {})}",
            f"Angular speed |nu_b[3:6]|: {paper.get('angular_speed_body', {})}",
            f"nu_b std by DOF: {paper.get('nu_b_std', {})}",
        ])
        if 'current_speed_inertial' in paper:
            lines.append(f"Current speed |v_c^n|: {paper.get('current_speed_inertial', {})}")
        if 'nu_r_std' in paper:
            lines.append(f"nu_r std by DOF: {paper.get('nu_r_std', {})}")
    figure_files = metadata.get('figure_files', [])
    if figure_files:
        lines.extend([
            "",
            "Figure Files",
            "-" * 80,
            *figure_files,
        ])
    return "\n".join(lines) + "\n"


def save_dataset(dataset, config=None, filename=None):
    """Save dataset pickle plus machine- and human-readable metadata."""
    if config is None:
        config = Config()
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = str(dataset.get("config", {}).get("dataset_id", _dataset_id(config)))
    path = save_dir / (filename or f"{_dataset_stem(config, dataset_id=dataset_id)}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)
    metadata = _build_dataset_metadata(dataset, config, path)
    stats_json, stats_npz = _build_dataset_statistics(dataset, config, metadata)
    figure_dir = path.parent / f"{path.stem}_figures"
    figure_files = _save_dataset_figures(stats_json, stats_npz, figure_dir) if config.save_figures else []
    metadata['figure_files'] = figure_files
    meta_path = path.with_suffix(".meta.json")
    stats_path = path.with_suffix(".stats.json")
    stats_npz_path = path.with_suffix(".stats.npz")
    summary_path = path.with_suffix(".summary.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)
    np.savez_compressed(stats_npz_path, **stats_npz)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(_build_dataset_summary_text(metadata, stats_json))
    print(f"Saved dataset to: {path}")
    print(f"Saved metadata to: {meta_path}")
    print(f"Saved statistics to: {stats_path}")
    print(f"Saved histogram/correlation arrays to: {stats_npz_path}")
    if figure_files:
        print(f"Saved figures to: {figure_dir}")
    print(f"Saved summary to: {summary_path}")
    return path


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate AUVHamNODE training data')
    parser.add_argument('--num_traj', type=int, default=500)
    parser.add_argument('--blocks', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--ocean_current', action='store_true',
                        help='Include ocean current in simulation (27D state)')
    parser.add_argument('--absolute_depth_context', action='store_true',
                        help='Append block-start absolute depth z_ref as a carried state channel')
    parser.add_argument('--current_speed_max', type=float, default=0.5,
                        help='Max ocean current speed [m/s]')
    parser.add_argument('--no_figures', action='store_true',
                        help='Skip exporting paper-ready dataset figures')

    args = parser.parse_args()
    cfg = Config(
        num_trajectories=args.num_traj,
        blocks_per_trajectory=args.blocks,
        seed=args.seed,
        save_dir=args.save_dir,
        enable_filtering=not args.no_filter,
        num_workers=args.workers,
        ocean_current=args.ocean_current,
        absolute_depth_context=args.absolute_depth_context,
        current_speed_range=(0.0, args.current_speed_max),
        save_figures=not args.no_figures,
    )
    ds = generate_dataset(cfg)
    save_dataset(ds, cfg)
