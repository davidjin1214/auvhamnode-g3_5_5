"""Validate the v4_lite noisy-initial-condition protocol.

This script is a protocol-level smoke check. It verifies that:

1. the same `(trajectory, block, epoch, seed)` request is reproducible
2. the same trajectory does not get re-sampled when batches are reordered
3. different epochs produce different trajectory-level realizations
4. heading-bias evaluation keeps the v4_lite per-trajectory sign contract

The script does not require trained weights. It only needs a run config
(`config.json`) plus the referenced dataset so it can reconstruct the model
layout and noise semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auv_model_registry import instantiate_model
from train_utils import (
    TrainConfig,
    StateNormalizer,
    _profile_rotation_bias_vector,
    _sample_sign_pattern,
    _sample_trajectory_sign_pattern,
    build_noisy_initial_condition,
    get_train_blocks,
    load_dataset,
    noise_cfg_from_profile,
)


def _load_mass_init(source: str, path: str | None):
    """Load the optional mass-matrix prior used by the training config."""
    source = (source or "none").lower()
    if source == "none":
        return None
    if source == "remus":
        from remus100_core import Remus100Dynamics

        return Remus100Dynamics().M
    if source == "file":
        if not path:
            raise ValueError("--mass_init_path is required when mass_init='file'.")
        payload = np.load(path)
        if isinstance(payload, np.ndarray):
            matrix = payload
        else:
            if "M" in payload:
                matrix = payload["M"]
            elif payload.files:
                matrix = payload[payload.files[0]]
            else:
                raise ValueError(f"No arrays found in mass init file: {path}")
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (6, 6):
            raise ValueError(f"Mass init must have shape (6, 6), got {matrix.shape}.")
        return matrix
    raise ValueError(f"Unsupported mass init source: {source}")


def _build_model(config: TrainConfig, device: torch.device):
    """Instantiate the model skeleton needed by the noise builder."""
    m_init = _load_mass_init(config.mass_init, config.mass_init_path)
    model = instantiate_model(
        config.model_type,
        device=device,
        hidden_dim=config.hidden_dim,
        M_init=m_init,
        coupled_damping=config.coupled_damping,
        include_depth_in_potential=config.include_depth_in_potential,
        ocean_current=config.ocean_current,
        actuation_current_feature=config.actuation_current_feature,
        dj_current_feature=config.dj_current_feature,
        t_actuator_init=config.t_actuator_init,
        u_act_scale=config.u_act_scale,
        u_dim=config.u_dim,
        absolute_depth_context=config.absolute_depth_context,
    )
    model.eval()
    return model


def _select_samples(
    train_blocks: np.ndarray,
    *,
    blocks_per_trajectory: int,
    max_trajectories: int,
    max_blocks_per_trajectory: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
    """Pick a small deterministic set of training samples for protocol checks."""
    if train_blocks.ndim != 3:
        raise ValueError(f"Expected train blocks with shape [N, T, D], got {train_blocks.shape}.")
    if blocks_per_trajectory <= 0:
        raise ValueError(f"blocks_per_trajectory must be positive, got {blocks_per_trajectory}.")
    if train_blocks.shape[0] % blocks_per_trajectory != 0:
        raise ValueError(
            "Flattened training blocks are not divisible by blocks_per_trajectory; "
            "trajectory indexing would be ambiguous."
        )

    n_trajectories = train_blocks.shape[0] // blocks_per_trajectory
    selected_trajs = list(range(min(max_trajectories, n_trajectories)))
    selected_blocks = list(range(min(max_blocks_per_trajectory, blocks_per_trajectory)))
    if not selected_trajs or not selected_blocks:
        raise ValueError("Need at least one trajectory and one block for validation.")

    flat_indices: List[int] = []
    traj_ids: List[int] = []
    block_indices: List[int] = []
    for traj_id in selected_trajs:
        for block_idx in selected_blocks:
            flat_indices.append(traj_id * blocks_per_trajectory + block_idx)
            traj_ids.append(traj_id)
            block_indices.append(block_idx)
    return (
        np.asarray(flat_indices, dtype=np.int64),
        np.asarray(traj_ids, dtype=np.int64),
        np.asarray(block_indices, dtype=np.int64),
        selected_trajs,
        selected_blocks,
    )


def _gather_initial_states(
    train_blocks: np.ndarray,
    flat_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Return clean initial states for the selected blocks."""
    y0_clean = train_blocks[flat_indices, 0]
    return torch.tensor(y0_clean, dtype=torch.float32, device=device)


def _build_noisy_states(
    clean_states: torch.Tensor,
    *,
    noise_cfg,
    model,
    normalizer: StateNormalizer,
    epoch: int,
    traj_ids: Sequence[int],
    block_indices: Sequence[int],
    base_seed: int,
) -> torch.Tensor:
    return build_noisy_initial_condition(
        clean_states,
        noise_cfg,
        model,
        normalizer,
        epoch=epoch,
        traj_ids=torch.tensor(traj_ids, dtype=torch.long, device=clean_states.device),
        block_indices=torch.tensor(block_indices, dtype=torch.long, device=clean_states.device),
        base_seed=base_seed,
    )


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=perm.device)
    return inv


def _max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item()) if lhs.numel() else 0.0


def _mutable_columns(model) -> List[int]:
    columns = list(range(3, 12))
    columns.extend(range(model.layout.nu_r.start, model.layout.nu_r.stop))
    columns.extend(range(model.layout.u_act.start, model.layout.u_act.stop))
    if getattr(model, "ocean_current", False):
        columns.extend(range(model.layout.v_c.start, model.layout.v_c.stop))
    return sorted(set(columns))


def _non_rotation_columns(state_dim: int) -> List[int]:
    return [idx for idx in range(state_dim) if idx < 3 or idx >= 12]


def _group_unique_rows(
    rows: torch.Tensor,
    traj_ids: Sequence[int],
) -> Dict[str, List[List[float]]]:
    """Collect unique row patterns per trajectory for JSON reporting."""
    payload: Dict[str, List[List[float]]] = {}
    for traj_id in sorted({int(x) for x in traj_ids}):
        mask = [idx for idx, value in enumerate(traj_ids) if int(value) == traj_id]
        group = rows[mask]
        unique = torch.unique(group, dim=0)
        payload[str(traj_id)] = [[float(v) for v in row.tolist()] for row in unique]
    return payload


def _per_traj_constant(unique_rows: Dict[str, List[List[float]]]) -> bool:
    return all(len(rows) == 1 for rows in unique_rows.values())


def _validate_same_epoch_stability(
    clean_states: torch.Tensor,
    *,
    noise_cfg,
    model,
    normalizer: StateNormalizer,
    epoch: int,
    traj_ids: np.ndarray,
    block_indices: np.ndarray,
    base_seed: int,
    tolerance: float,
) -> Dict:
    full = _build_noisy_states(
        clean_states,
        noise_cfg=noise_cfg,
        model=model,
        normalizer=normalizer,
        epoch=epoch,
        traj_ids=traj_ids,
        block_indices=block_indices,
        base_seed=base_seed,
    )

    perm = torch.arange(full.shape[0] - 1, -1, -1, device=full.device)
    permuted = _build_noisy_states(
        clean_states[perm],
        noise_cfg=noise_cfg,
        model=model,
        normalizer=normalizer,
        epoch=epoch,
        traj_ids=traj_ids[perm.cpu().numpy()],
        block_indices=block_indices[perm.cpu().numpy()],
        base_seed=base_seed,
    )
    permuted_back = permuted[_inverse_permutation(perm)]

    singleton_rows = []
    for row_idx in range(full.shape[0]):
        singleton_rows.append(
            _build_noisy_states(
                clean_states[row_idx : row_idx + 1],
                noise_cfg=noise_cfg,
                model=model,
                normalizer=normalizer,
                epoch=epoch,
                traj_ids=[int(traj_ids[row_idx])],
                block_indices=[int(block_indices[row_idx])],
                base_seed=base_seed,
            )
        )
    singleton = torch.cat(singleton_rows, dim=0)

    perm_diff = _max_abs_diff(full, permuted_back)
    singleton_diff = _max_abs_diff(full, singleton)
    return {
        "passed": bool(perm_diff <= tolerance and singleton_diff <= tolerance),
        "epoch": int(epoch),
        "base_seed": int(base_seed),
        "max_abs_diff_full_vs_reordered": perm_diff,
        "max_abs_diff_full_vs_singleton": singleton_diff,
        "tolerance": float(tolerance),
    }


def _validate_epoch_resampling(
    clean_states: torch.Tensor,
    *,
    noise_cfg,
    model,
    normalizer: StateNormalizer,
    traj_ids: np.ndarray,
    block_indices: np.ndarray,
    base_seed: int,
    epoch_a: int,
    epoch_b: int,
    min_changed_fraction: float,
    min_max_abs_diff: float,
) -> Dict:
    a = _build_noisy_states(
        clean_states,
        noise_cfg=noise_cfg,
        model=model,
        normalizer=normalizer,
        epoch=epoch_a,
        traj_ids=traj_ids,
        block_indices=block_indices,
        base_seed=base_seed,
    )
    b = _build_noisy_states(
        clean_states,
        noise_cfg=noise_cfg,
        model=model,
        normalizer=normalizer,
        epoch=epoch_b,
        traj_ids=traj_ids,
        block_indices=block_indices,
        base_seed=base_seed,
    )

    mutable = torch.tensor(_mutable_columns(model), dtype=torch.long, device=a.device)
    delta = (a[:, mutable] - b[:, mutable]).abs()
    changed_fraction = float((delta > 1e-7).float().mean().item()) if delta.numel() else 0.0
    max_abs_diff = float(delta.max().item()) if delta.numel() else 0.0
    mean_abs_diff = float(delta.mean().item()) if delta.numel() else 0.0
    return {
        "passed": bool(
            changed_fraction >= min_changed_fraction and max_abs_diff >= min_max_abs_diff
        ),
        "epoch_a": int(epoch_a),
        "epoch_b": int(epoch_b),
        "base_seed": int(base_seed),
        "mutable_changed_fraction": changed_fraction,
        "mutable_max_abs_diff": max_abs_diff,
        "mutable_mean_abs_diff": mean_abs_diff,
        "min_changed_fraction": float(min_changed_fraction),
        "min_max_abs_diff": float(min_max_abs_diff),
    }


def _validate_heading_bias_overlay(
    clean_states: torch.Tensor,
    *,
    model,
    normalizer: StateNormalizer,
    reference: str,
    trajectory_correlation: float,
    traj_ids: np.ndarray,
    block_indices: np.ndarray,
    flat_indices: np.ndarray,
    base_seed: int,
    epoch: int,
    tolerance: float,
) -> Dict:
    nominal_cfg = noise_cfg_from_profile(
        "nominal_eval",
        reference=reference,
        protocol="v4_lite",
        trajectory_correlation=trajectory_correlation,
    )
    heading_cfg = noise_cfg_from_profile(
        "heading_biased_eval",
        reference=reference,
        protocol="v4_lite",
        trajectory_correlation=trajectory_correlation,
    )
    nominal = _build_noisy_states(
        clean_states,
        noise_cfg=nominal_cfg,
        model=model,
        normalizer=normalizer,
        epoch=epoch,
        traj_ids=traj_ids,
        block_indices=block_indices,
        base_seed=base_seed,
    )
    heading = _build_noisy_states(
        clean_states,
        noise_cfg=heading_cfg,
        model=model,
        normalizer=normalizer,
        epoch=epoch,
        traj_ids=traj_ids,
        block_indices=block_indices,
        base_seed=base_seed,
    )

    non_rot = torch.tensor(_non_rotation_columns(nominal.shape[1]), dtype=torch.long, device=nominal.device)
    non_rot_diff = _max_abs_diff(nominal[:, non_rot], heading[:, non_rot])

    v4_signs = _sample_trajectory_sign_pattern(
        torch.tensor(traj_ids, dtype=torch.long),
        dim=1,
        device=nominal.device,
        dtype=nominal.dtype,
        base_seed=base_seed,
        epoch=epoch,
        stream=29,
    )
    iid_signs = _sample_sign_pattern(
        len(flat_indices),
        dim=1,
        device=nominal.device,
        dtype=nominal.dtype,
        sample_ids=torch.tensor(flat_indices, dtype=torch.long),
        base_seed=base_seed,
        stream=29,
    )
    v4_unique = _group_unique_rows(v4_signs, traj_ids)
    iid_unique = _group_unique_rows(iid_signs, traj_ids)
    rot_bias = _profile_rotation_bias_vector(
        "heading_biased_eval",
        reference,
        dtype=nominal.dtype,
        device=nominal.device,
    )

    return {
        "passed": bool(
            non_rot_diff <= tolerance
            and torch.any(rot_bias > 0)
            and _per_traj_constant(v4_unique)
        ),
        "epoch": int(epoch),
        "base_seed": int(base_seed),
        "non_rotation_max_abs_diff_vs_nominal_eval": non_rot_diff,
        "tolerance": float(tolerance),
        "rotation_bias_vector": [float(value) for value in rot_bias.cpu().tolist()],
        "v4_unique_sign_patterns_by_trajectory": v4_unique,
        "iid_unique_sign_patterns_by_trajectory": iid_unique,
        "v4_sign_constant_within_trajectory": _per_traj_constant(v4_unique),
        "iid_sign_constant_within_trajectory": _per_traj_constant(iid_unique),
    }


def _resolve_output_path(
    output: str | None,
    *,
    run_dir: Path | None,
    config_path: Path,
) -> Path:
    if output:
        return Path(output)
    base = run_dir if run_dir is not None else config_path.parent
    return base / "v4_lite_protocol_validation.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the v4_lite protocol contract.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=str, help="Run directory containing config.json")
    source.add_argument("--config", type=str, help="Path to a config.json file")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path from config.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--traj-count", type=int, default=3)
    parser.add_argument("--blocks-per-traj-check", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--consistency-profile", type=str, default="nominal_eval")
    parser.add_argument("--epoch-a", type=int, default=1)
    parser.add_argument("--epoch-b", type=int, default=2)
    parser.add_argument("--heading-epoch", type=int, default=1)
    parser.add_argument("--stability-tol", type=float, default=1e-6)
    parser.add_argument("--resample-min-changed-fraction", type=float, default=0.05)
    parser.add_argument("--resample-min-max-abs-diff", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = _parse_args()

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    config_path = (
        (run_dir / "config.json") if run_dir is not None else Path(args.config)
    ).resolve()
    config = TrainConfig.load(str(config_path))
    dataset_path = Path(args.dataset or config.dataset_path).resolve()
    output_path = _resolve_output_path(args.output, run_dir=run_dir, config_path=config_path)

    dataset = load_dataset(str(dataset_path))
    train_blocks = get_train_blocks(dataset)
    dataset_cfg = dataset["config"]
    blocks_per_trajectory = int(dataset_cfg.get("blocks_per_trajectory", 1))
    device = torch.device(args.device)
    normalizer = StateNormalizer.from_dataset(
        train_blocks,
        device=args.device,
        u_dim=config.u_dim,
    )
    model = _build_model(config, device)

    flat_indices, traj_ids, block_indices, selected_trajs, selected_blocks = _select_samples(
        train_blocks,
        blocks_per_trajectory=blocks_per_trajectory,
        max_trajectories=args.traj_count,
        max_blocks_per_trajectory=args.blocks_per_traj_check,
    )
    clean_states = _gather_initial_states(train_blocks, flat_indices, device)

    consistency_cfg = noise_cfg_from_profile(
        args.consistency_profile,
        reference=config.noise_reference,
        protocol="v4_lite",
        trajectory_correlation=config.noise_ar1_corr,
    )
    base_seed = (
        int(args.base_seed)
        if args.base_seed is not None
        else int(config.seed) + 4000
    )

    checks = {
        "same_epoch_stability": _validate_same_epoch_stability(
            clean_states,
            noise_cfg=consistency_cfg,
            model=model,
            normalizer=normalizer,
            epoch=args.epoch_a,
            traj_ids=traj_ids,
            block_indices=block_indices,
            base_seed=base_seed,
            tolerance=args.stability_tol,
        ),
        "epoch_resampling": _validate_epoch_resampling(
            clean_states,
            noise_cfg=consistency_cfg,
            model=model,
            normalizer=normalizer,
            traj_ids=traj_ids,
            block_indices=block_indices,
            base_seed=base_seed,
            epoch_a=args.epoch_a,
            epoch_b=args.epoch_b,
            min_changed_fraction=args.resample_min_changed_fraction,
            min_max_abs_diff=args.resample_min_max_abs_diff,
        ),
        "heading_bias_overlay": _validate_heading_bias_overlay(
            clean_states,
            model=model,
            normalizer=normalizer,
            reference=config.noise_reference,
            trajectory_correlation=config.noise_ar1_corr,
            traj_ids=traj_ids,
            block_indices=block_indices,
            flat_indices=flat_indices,
            base_seed=base_seed,
            epoch=args.heading_epoch,
            tolerance=args.stability_tol,
        ),
    }
    overall_passed = all(item["passed"] for item in checks.values())

    payload = {
        "overall_passed": bool(overall_passed),
        "context": {
            "config_path": str(config_path),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "dataset_path": str(dataset_path),
            "model_type": config.model_type,
            "training_noise_profile": config.resolved_noise_profile(),
            "training_noise_protocol": config.resolved_noise_protocol(),
            "noise_reference": config.noise_reference,
            "trajectory_correlation": float(config.noise_ar1_corr),
            "device": str(device),
        },
        "selection": {
            "selected_trajectory_ids": selected_trajs,
            "selected_block_indices": selected_blocks,
            "flat_indices": [int(value) for value in flat_indices.tolist()],
            "blocks_per_trajectory": int(blocks_per_trajectory),
            "consistency_profile": args.consistency_profile,
        },
        "checks": checks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"v4_lite protocol validation saved to: {output_path}")
    print(f"overall_passed={payload['overall_passed']}")
    for name, result in checks.items():
        print(f"{name}: passed={result['passed']}")


if __name__ == "__main__":
    main()
