"""
AUV dynamics model training pipeline.

Supports all models in the ablation chain:
  phnode_full            -- AUVHamNODE (structured open pH core, decomposed D/J/V/B)
  phnode_merged_force    -- pH core with merged non-conservative force branch
  phnode_qforce          -- structured pH model with generalized q-force instead of scalar V(q)
  se3_momentum_blackbox  -- exact SE(3) + constant mass matrix in momentum coordinates
  se3_accel_blackbox     -- exact SE(3) kinematics + black-box acceleration
  blackbox_fullstate     -- fully unstructured (even kinematics learned)
  ablate_no_mass_prior   -- AUVHamNODE without physics-based mass initialization
  ablate_diag_damping    -- AUVHamNODE with diagonal damping only
  ablate_no_lift         -- AUVHamNODE without learned skew-symmetric lift
  ablate_bu_only         -- AUVHamNODE with B(u) only

Usage:
    python train_auv_hamnode.py --dataset ./data/auv_dataset.pkl
    python train_auv_hamnode.py --dataset ./data/dataset.pkl --model_type se3_accel_blackbox
    python train_auv_hamnode.py --dataset ./data/dataset.pkl --model_type blackbox_fullstate --batch_size 2048 --total_steps 7000 --epochs 200
    python train_auv_hamnode.py --dataset ./data/dataset.pkl --noise_level 2
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import Dict, Optional, Tuple
from collections import defaultdict
from pathlib import Path
from copy import deepcopy
import pickle
import time
import argparse

from torchdiffeq import odeint

from train_utils import (
    TrainConfig, StateNormalizer, TrainingLogger,
    se3_trajectory_loss,
    save_checkpoint, evaluate_trajectory_prediction,
    print_evaluation_results, setup_logging,
    load_dataset, get_train_blocks, evaluate_heldout_trajectories,
    print_heldout_evaluation_results, save_heldout_evaluation_results,
    save_block_evaluation_results, build_noisy_training_pair,
    adapt_state_array_for_model,
    validate_depth_conditioning_support,
    create_dataloaders_from_dataset,
    get_dataset_training_defaults,
    infer_dataset_kind_from_path,
)
from auv_model_registry import (
    canonicalize_model_type,
    format_model_type_help,
    get_model_spec,
    instantiate_model,
)


def _load_mass_init(source: str, path: Optional[str]):
    """Load an optional mass-matrix prior from a named source."""
    source = (source or "none").lower()
    if source == "none":
        return None
    if source == "remus":
        from remus100_core import Remus100Dynamics
        return Remus100Dynamics().M
    if source == "file":
        if not path:
            raise ValueError("--mass_init_path is required when --mass_init file.")
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


def _build_model(model_type, device, hidden_dim=128, M_init=None, **kwargs):
    """Instantiate the requested model type via the centralized model registry."""
    return instantiate_model(
        model_type,
        device,
        hidden_dim=hidden_dim,
        M_init=M_init,
        **kwargs,
    )


class AUVHamNODETrainer:
    """Handles model creation, training loop, validation, and checkpointing."""

    def __init__(self, config: TrainConfig,
                 model_class=None,
                 normalizer: Optional[StateNormalizer] = None,
                 M_init=None):
        self.config = config
        self.config.model_type = canonicalize_model_type(self.config.model_type)
        self.device = torch.device(config.device)
        self.normalizer = normalizer

        self.run_dir = config.get_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.run_dir)
        config.save(str(self.run_dir / "config.json"))

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if model_class is None:
            self.model = _build_model(
                config.model_type, self.device,
                hidden_dim=config.hidden_dim,
                M_init=M_init,
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
        else:
            self.model = model_class(
                device=self.device,
                hidden_dim=config.hidden_dim,
                coupled_damping=config.coupled_damping,
                include_depth_in_potential=config.include_depth_in_potential,
                M_init=M_init,
                actuation_current_feature=config.actuation_current_feature,
                dj_current_feature=config.dj_current_feature,
                T_actuator_init=config.t_actuator_init,
                u_act_scale=config.u_act_scale,
                u_dim=config.u_dim,
                absolute_depth_context=config.absolute_depth_context,
            ).to(self.device)

        spec = get_model_spec(self.config.model_type)
        self.logger.info(
            "Model: %s [%s/%s] | canonical type: %s",
            spec.display_name,
            spec.family,
            spec.group,
            spec.name,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {n_params:,}")
        if config.dataset_id is not None:
            self.logger.info(
                "Training dataset: id=%s | path=%s",
                config.dataset_id,
                config.dataset_path or "n/a",
            )
        if M_init is not None:
            self.logger.info("Mass matrix initialized from physics")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.metrics = TrainingLogger(self.run_dir)
        self.best_loss = float("inf")
        self.best_failure_rate = float("inf")
        self.best_selection_key = (float("inf"), float("inf"))
        self.best_epoch = None
        self.best_state = None

    def _build_scheduler(self) -> LambdaLR:
        """Step-based linear warmup followed by cosine decay."""
        cfg = self.config
        total_steps = max(1, int(cfg.total_steps))
        warmup_steps = max(0, min(int(cfg.warmup_steps), total_steps - 1))
        min_lr = min(float(cfg.min_learning_rate), float(cfg.learning_rate))
        min_lr_ratio = min_lr / float(cfg.learning_rate) if cfg.learning_rate > 0 else 0.0

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                progress = float(step + 1) / float(warmup_steps)
                return min_lr_ratio + (1.0 - min_lr_ratio) * progress

            if total_steps <= warmup_steps + 1:
                return min_lr_ratio

            progress = float(step - warmup_steps) / float(total_steps - warmup_steps - 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _set_initial_learning_rate(self):
        """Set the optimizer LR to the warmup start value before the first step."""
        cfg = self.config
        initial_lr = (
            float(cfg.min_learning_rate)
            if int(cfg.warmup_steps) > 0 else float(cfg.learning_rate)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = initial_lr

    def _run_epoch(self, loader: DataLoader,
                   t_eval: torch.Tensor, train: bool = True,
                   epoch: int = 1,
                   scheduler: Optional[LambdaLR] = None,
                   global_step: int = 0) -> Tuple[Dict, int]:
        """Run one epoch and return averaged metrics with failure statistics."""
        self.model.train() if train else self.model.eval()
        t_eval_dev = t_eval.to(self.device)

        totals: Dict[str, float] = defaultdict(float)
        attempted_batches = 0
        successful_batches = 0
        solver_failed_batches = 0
        invalid_prediction_batches = 0
        invalid_gradient_batches = 0

        for batch in loader:
            if train and global_step >= self.config.total_steps:
                break

            batch = batch.to(self.device)
            attempted_batches += 1
            self.model.reset_nfe()

            # Build (possibly noisy) initial condition; supervision is always clean.
            noise_cfg = self.config.get_noise_config()
            if train and noise_cfg.is_active:
                noisy_block = build_noisy_training_pair(
                    batch, noise_cfg, epoch, t_eval_dev,
                    u_dim=self.config.u_dim,
                    ocean_current=self.config.ocean_current,
                )
                y0 = self.model.to_ode_state(noisy_block[:, 0])
            else:
                y0 = self.model.to_ode_state(batch[:, 0])

            try:
                pred = odeint(
                    self.model,
                    y0,
                    t_eval_dev,
                    method=self.config.ode_solver,
                )
            except (ValueError, RuntimeError):
                solver_failed_batches += 1
                continue

            pred_ode = pred.permute(1, 0, 2)
            if torch.isnan(pred_ode).any() or torch.isinf(pred_ode).any():
                invalid_prediction_batches += 1
                continue

            # Target is always the clean ground-truth trajectory.
            # Convert to ODE convention; position/rotation/actuator are identical.
            target_ode = self.model.to_ode_state(batch)
            loss_target = target_ode
            loss_pred = pred_ode
            if train and noise_cfg.is_active and target_ode.shape[1] > 1:
                # Skip t=0: y0 is noisy but target[:, 0] is clean, so the
                # t=0 residual reflects the injected IC noise, not dynamics error.
                loss_target = target_ode[:, 1:]
                loss_pred = pred_ode[:, 1:]

            loss, comp = se3_trajectory_loss(
                loss_target,
                loss_pred,
                self.normalizer,
                weights={"actuator": self.config.actuator_loss_weight},
                so3_regularization_weight=self.config.so3_regularization_weight,
            )

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                bad_grad = any(
                    p.grad is not None and
                    (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                    for p in self.model.parameters()
                )
                if bad_grad:
                    invalid_gradient_batches += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            for k, v in comp.items():
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    totals[k] += v.item()

            successful_batches += 1

        skipped_batches = (
            solver_failed_batches
            + invalid_prediction_batches
            + invalid_gradient_batches
        )

        metrics = {
            "attempted_batches": float(attempted_batches),
            "successful_batches": float(successful_batches),
            "solver_failed_batches": float(solver_failed_batches),
            "invalid_prediction_batches": float(invalid_prediction_batches),
            "invalid_gradient_batches": float(invalid_gradient_batches),
            "skipped_batches": float(skipped_batches),
            "success_rate": (successful_batches / attempted_batches) if attempted_batches else 0.0,
            "failure_rate": (skipped_batches / attempted_batches) if attempted_batches else 0.0,
        }

        if successful_batches == 0:
            metrics["total"] = float("inf")
            return metrics, global_step

        for k, v in totals.items():
            metrics[k] = v / successful_batches
        return metrics, global_step

    def train(self, train_loader: DataLoader,
              test_loader: DataLoader,
              t_eval: torch.Tensor) -> Dict:
        """Full training loop with validation, scheduling, and checkpointing."""
        cfg = self.config
        scheduler = self._build_scheduler()
        self._set_initial_learning_rate()
        global_step = 0

        self.logger.info(
            "Training: epochs<=%d, target_steps=%d, warmup_steps=%d, lr=%.2e -> %.2e, "
            "solver=%s, so3_reg=%.2e",
            cfg.num_epochs,
            cfg.total_steps,
            cfg.warmup_steps,
            cfg.learning_rate,
            cfg.min_learning_rate,
            cfg.ode_solver,
            cfg.so3_regularization_weight,
        )
        self.logger.info(f"Train samples: {len(train_loader.dataset)}, "
                         f"Test samples: {len(test_loader.dataset)}")

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()

            train_m, global_step = self._run_epoch(
                train_loader,
                t_eval,
                train=True,
                epoch=epoch,
                scheduler=scheduler,
                global_step=global_step,
            )
            if train_m["total"] == float("inf"):
                self.logger.warning(
                    f"Epoch {epoch}: no successful training batches "
                    f"(solver={int(train_m['solver_failed_batches'])}, "
                    f"pred={int(train_m['invalid_prediction_batches'])}, "
                    f"grad={int(train_m['invalid_gradient_batches'])})"
                )
                continue

            test_m, _ = self._run_epoch(test_loader, t_eval, train=False)

            lr = self.optimizer.param_groups[0]["lr"]
            dt = time.time() - t0

            self.metrics.log(epoch, train_m, test_m, lr, dt)
            self.metrics.history["global_step"].append(global_step)

            if epoch % cfg.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch:4d} | "
                    f"Step {global_step:5d}/{cfg.total_steps:5d} | "
                    f"Train {train_m['total']:.4e} | "
                    f"Test {test_m['total']:.4e} | "
                    f"SO3(train/test) {train_m.get('so3_orth', float('nan')):.2e}/"
                    f"{test_m.get('so3_orth', float('nan')):.2e} | "
                    f"Fail(train/test) {int(train_m['skipped_batches'])}/"
                    f"{int(test_m['skipped_batches'])} | "
                    f"LR {lr:.2e} | {dt:.1f}s"
                )

            test_loss = test_m["total"]
            test_failure_rate = float(test_m.get("failure_rate", 1.0))
            selection_key = (test_failure_rate, test_loss)
            if selection_key < self.best_selection_key:
                self.best_selection_key = selection_key
                self.best_failure_rate = test_failure_rate
                self.best_loss = test_loss
                self.best_epoch = epoch
                self.best_state = deepcopy(self.model.state_dict())
                save_checkpoint(
                    self.model, self.optimizer, epoch, self.best_loss,
                    cfg, str(self.run_dir / "best_model.pt"),
                    normalizer=self.normalizer,
                )

            if epoch % cfg.save_interval == 0:
                save_checkpoint(
                    self.model, self.optimizer, epoch, test_loss,
                    cfg, str(self.run_dir / f"checkpoint_{epoch}.pt"),
                    normalizer=self.normalizer,
                )

            if global_step >= cfg.total_steps:
                self.logger.info(
                    "Reached target optimizer steps: %d/%d",
                    global_step,
                    cfg.total_steps,
                )
                break

        if global_step < cfg.total_steps:
            self.logger.warning(
                "Stopped at epoch limit before target steps were reached: %d/%d",
                global_step,
                cfg.total_steps,
            )

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.metrics.save()

        summary = self.metrics.get_summary()
        if self.best_epoch is not None:
            summary["best_test_loss"] = self.best_loss
            summary["best_failure_rate"] = self.best_failure_rate
            summary["best_epoch"] = self.best_epoch
        if summary["best_test_loss"] is None:
            self.logger.warning("Done. No valid test epoch was recorded.")
        else:
            self.logger.info(
                f"Done. Best validation score: failure_rate={self.best_failure_rate:.4f}, "
                f"test_loss={summary['best_test_loss']:.4e} "
                f"(epoch {summary['best_epoch']})"
            )
        return summary

    def get_model(self) -> nn.Module:
        return self.model


def train_auv_hamnode(
    dataset_path: str,
    config: Optional[TrainConfig] = None,
    model_class=None,
    M_init=None,
) -> Tuple[nn.Module, Dict]:
    """
    End-to-end training: load data -> train -> evaluate -> save.

    Args:
        dataset_path: path to pickled dataset
        config:       training configuration
        model_class:  custom model class with the AUVHamNODE interface
        M_init:       6x6 mass matrix for physics initialization (optional)

    Returns:
        model:   trained model (best checkpoint)
        summary: training summary dict
    """
    if config is None:
        config = TrainConfig()

    logger = setup_logging(config.get_run_dir())
    logger.info("Loading dataset...")

    dataset = load_dataset(dataset_path)
    train_loader, test_loader, t_eval, data_cfg = create_dataloaders_from_dataset(
        dataset,
        batch_size=config.batch_size,
    )
    logger.info(f"Loaded: {len(train_loader.dataset)} train, "
                f"{len(test_loader.dataset)} test, t_eval={t_eval.numpy()}")
    logger.info(
        "Dataset split: %s | train trajectories: %s | test trajectories: %s",
        data_cfg.get("split_level", "block"),
        data_cfg.get("num_train_trajectories", "n/a"),
        data_cfg.get("num_test_trajectories", "n/a"),
    )
    config.dataset_path = str(Path(dataset_path).resolve())
    config.dataset_id = data_cfg.get("dataset_id")
    config.dataset_velocity_convention = data_cfg.get("velocity_convention")
    config.dataset_description = data_cfg.get("description")
    config.dataset_generation_config = data_cfg.get("generation_config")
    if "state_dim" in data_cfg:
        config.dataset_state_dim = int(data_cfg["state_dim"])
    config.ocean_current = bool(data_cfg.get("ocean_current", config.ocean_current))
    config.absolute_depth_context = bool(
        data_cfg.get("absolute_depth_available", config.absolute_depth_context)
    )
    config.u_dim = int(data_cfg.get("u_dim", config.u_dim))
    if config.dataset_id is not None:
        logger.info("Dataset ID: %s", config.dataset_id)
    logger.info(
        "Dataset path: %s | velocity convention: %s | state dim: %s",
        config.dataset_path,
        config.dataset_velocity_convention or "n/a",
        config.dataset_state_dim if config.dataset_state_dim is not None else "n/a",
    )
    validate_depth_conditioning_support(
        data_cfg,
        config.include_depth_in_potential,
        context="Training pipeline",
    )
    train_blocks = get_train_blocks(dataset)
    train_blocks = adapt_state_array_for_model(
        train_blocks,
        dataset_cfg=data_cfg,
        ocean_current=config.ocean_current,
    )
    normalizer = StateNormalizer.from_dataset(
        train_blocks,
        device=config.device,
        u_dim=config.u_dim,
    )
    logger.info(normalizer.summary())

    trainer = AUVHamNODETrainer(config, model_class,
                                normalizer=normalizer, M_init=M_init)
    summary = trainer.train(train_loader, test_loader, t_eval)

    logger.info("Running detailed evaluation...")
    model = trainer.get_model()
    results = evaluate_trajectory_prediction(
        model, test_loader, t_eval,
        torch.device(config.device),
        ode_solver=config.ode_solver,
        n_samples=min(500, len(test_loader.dataset)),
    )
    print_evaluation_results(results, logger)
    save_block_evaluation_results(results, trainer.run_dir)

    with open(trainer.run_dir / "evaluation_results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info("Running held-out trajectory evaluation...")
    heldout_results = evaluate_heldout_trajectories(
        model,
        dataset,
        t_eval,
        torch.device(config.device),
        ode_solver=config.ode_solver,
    )
    print_heldout_evaluation_results(heldout_results, logger)
    save_heldout_evaluation_results(heldout_results, trainer.run_dir)

    with open(trainer.run_dir / "heldout_evaluation.pkl", "wb") as f:
        pickle.dump(heldout_results, f)

    return model, summary


def main():
    parser = argparse.ArgumentParser(
        description="Train AUV dynamics models (AUVHamNODE + baselines)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="phnode_full",
        help=(
            "Model architecture to train. Canonical names:\n"
            f"{format_model_type_help()}\n"
            "Legacy aliases from older checkpoints/configs are also accepted."
        ),
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Dataset-aware default when omitted: noc=2048, oc=4096")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Maximum epoch budget; dataset-aware default when omitted")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Target number of optimizer steps for training; dataset-aware default when omitted")
    parser.add_argument("--lr", type=float, default=None,
                        help="Peak learning rate used after warmup; dataset-aware default when omitted")
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Final learning rate floor for cosine decay; dataset-aware default when omitted")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Number of optimizer steps for linear warmup; dataset-aware default when omitted")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--so3_reg", type=float, default=1e-3)
    parser.add_argument("--actuator_loss_weight", type=float, default=0.2,
                        help="Weight for supervised actuator-state loss")
    parser.add_argument("--include_depth_in_potential", action="store_true",
                        help="Condition potential / force baselines on depth")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise_level", type=int, default=0, choices=[0, 1, 2, 3],
        help=(
            "Training-time sensor noise level: "
            "0=clean (no noise), "
            "1=ic (white Gaussian on initial condition only), "
            "2=nav (full-trajectory AR(1) DVL/IMU noise + bias), "
            "3=nav_deg (nav + DVL dropout + random-walk bias + multiplicative noise)"
        ),
    )
    parser.add_argument(
        "--noise_scale", type=float, default=1.0,
        help="Global noise magnitude multiplier (default 1.0); use <1 for ablation",
    )
    parser.add_argument(
        "--noise_ramp", type=int, default=100,
        help="Curriculum ramp: noise increases linearly from 0 to full over N epochs",
    )
    parser.add_argument("--ocean_current", action="store_true",
                        help="Enable ocean current awareness (requires 27D dataset)")
    parser.add_argument("--dj_current_feature", type=str, default=None,
                        choices=["none", "current_body", "total_velocity"],
                        help="Extra 3D context appended to D_net/J_net in ocean-current runs")
    parser.add_argument("--actuation_current_feature", type=str, default=None,
                        choices=["none", "current_body", "total_velocity"],
                        help="Extra 3D context appended to B_net in ocean-current runs")
    parser.add_argument("--mass_init", type=str, default=None,
                        choices=["none", "remus", "file"],
                        help="Mass-matrix prior source; dataset-aware default when omitted")
    parser.add_argument("--mass_init_path", type=str, default=None,
                        help="Path to a .npy/.npz mass matrix file when --mass_init file")
    parser.add_argument("--t_actuator_init", type=float, nargs="+", default=None,
                        help="Actuator time-constant prior(s); length 1 or u_dim")
    parser.add_argument("--u_act_scale", type=float, nargs="+", default=None,
                        help="Actuator scaling used by B_net; length 1 or u_dim")
    args = parser.parse_args()

    dataset_kind = infer_dataset_kind_from_path(args.dataset)
    dataset_defaults = get_dataset_training_defaults(dataset_kind=dataset_kind)

    dj_current_feature = args.dj_current_feature
    if dj_current_feature is None:
        dj_current_feature = dataset_defaults["dj_current_feature"]

    actuation_current_feature = args.actuation_current_feature
    if actuation_current_feature is None:
        actuation_current_feature = dataset_defaults["actuation_current_feature"]

    mass_init = args.mass_init if args.mass_init is not None else dataset_defaults["mass_init"]
    t_actuator_init = (
        deepcopy(args.t_actuator_init)
        if args.t_actuator_init is not None else deepcopy(dataset_defaults["t_actuator_init"])
    )
    u_act_scale = (
        deepcopy(args.u_act_scale)
        if args.u_act_scale is not None else deepcopy(dataset_defaults["u_act_scale"])
    )

    config = TrainConfig(
        model_type=canonicalize_model_type(args.model_type),
        batch_size=args.batch_size if args.batch_size is not None else dataset_defaults["batch_size"],
        num_epochs=args.epochs if args.epochs is not None else dataset_defaults["num_epochs"],
        total_steps=args.total_steps if args.total_steps is not None else dataset_defaults["total_steps"],
        learning_rate=args.lr if args.lr is not None else dataset_defaults["learning_rate"],
        min_learning_rate=args.min_lr if args.min_lr is not None else dataset_defaults["min_learning_rate"],
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else dataset_defaults["warmup_steps"],
        hidden_dim=args.hidden_dim,
        so3_regularization_weight=args.so3_reg,
        actuator_loss_weight=args.actuator_loss_weight,
        include_depth_in_potential=args.include_depth_in_potential,
        save_dir=args.save_dir,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        noise_level=args.noise_level,
        noise_scale=args.noise_scale,
        noise_ramp_epochs=args.noise_ramp,
        ocean_current=args.ocean_current,
        dj_current_feature=dj_current_feature,
        actuation_current_feature=actuation_current_feature,
        mass_init=mass_init,
        mass_init_path=args.mass_init_path,
        t_actuator_init=t_actuator_init,
        u_act_scale=u_act_scale,
    )

    M_init = _load_mass_init(mass_init, args.mass_init_path)
    model, summary = train_auv_hamnode(args.dataset, config, M_init=M_init)
    print(f"\nTraining finished. Results: {config.get_run_dir()}")


if __name__ == "__main__":
    main()
