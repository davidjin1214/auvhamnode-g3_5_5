"""
AUV dynamics model training pipeline.

Supports all models in the ablation chain:
  hamnode           -- AUVHamNODE (structured open pH core, decomposed D/J/V/B)
  hamnode_noinit    -- AUVHamNODE without physics-based mass initialization
  hamnode_diag_d    -- AUVHamNODE with diagonal damping only
  hamnode_no_j      -- AUVHamNODE without learned skew-symmetric lift
  hamnode_bu_only   -- AUVHamNODE with B(u) only
  hamnode_no_potential -- mass + co-adjoint + SE(3), but no explicit V(q)
  momentum_se3      -- exact SE(3) + constant mass matrix in momentum coordinates
  unstructured_ham  -- SE(3) kinematics + single H_net + F_net
  unstructured_se3  -- SE(3) kinematics + black-box f_net
  blackbox          -- fully unstructured (even kinematics learned)

Usage:
    python train_auv_hamnode.py --dataset ./data/auv_dataset.pkl
    python train_auv_hamnode.py --dataset ./data/dataset.pkl --model_type unstructured_se3
    python train_auv_hamnode.py --dataset ./data/dataset.pkl --model_type blackbox --epochs 500
    python train_auv_hamnode.py --dataset ./data/dataset.pkl --init_state_noise --observation_noise
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
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
    create_dataloaders, se3_trajectory_loss,
    save_checkpoint, evaluate_trajectory_prediction,
    print_evaluation_results, setup_logging,
    load_dataset, get_train_blocks, evaluate_heldout_trajectories,
    print_heldout_evaluation_results, save_heldout_evaluation_results,
    save_block_evaluation_results, build_noisy_training_pair,
    adapt_state_array_for_model,
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
    """Instantiate the requested model type with appropriate hidden dimensions."""
    if model_type in {'hamnode', 'hamnode_full', 'hamnode_noinit', 'hamnode_diag_d', 'hamnode_no_j', 'hamnode_bu_only'}:
        from AUVHamNODE import AUVHamNODE
        use_mass_init = model_type != 'hamnode_noinit'
        return AUVHamNODE(
            device=device,
            hidden_dim=hidden_dim,
            coupled_damping=(
                kwargs.get('coupled_damping', True)
                if model_type != 'hamnode_diag_d' else False
            ),
            include_depth_in_potential=kwargs.get('include_depth_in_potential', False),
            M_init=M_init if use_mass_init else None,
            ocean_current=kwargs.get('ocean_current', False),
            learn_lift=(model_type != 'hamnode_no_j'),
            actuation_condition_on_velocity=(model_type != 'hamnode_bu_only'),
            actuation_condition_on_current=(model_type != 'hamnode_bu_only'),
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
            u_dim=kwargs.get('u_dim', 3),
        ).to(device)

    from auv_baselines import BASELINE_MODELS

    oc = kwargs.get('ocean_current', False)

    if model_type == 'merged_nc':
        return BASELINE_MODELS['merged_nc'](
            device=device, hidden_dim=hidden_dim, M_init=M_init,
            include_depth_in_potential=kwargs.get('include_depth_in_potential', False),
            ocean_current=oc,
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
        ).to(device)

    if model_type in {'hamnode_no_potential', 'no_potential'}:
        return BASELINE_MODELS['hamnode_no_potential'](
            device=device, hidden_dim=hidden_dim, M_init=M_init,
            include_depth=kwargs.get('include_depth_in_potential', False),
            ocean_current=oc,
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
        ).to(device)

    if model_type == 'momentum_se3':
        return BASELINE_MODELS['momentum_se3'](
            device=device, hidden_dim=hidden_dim, M_init=M_init,
            include_depth=kwargs.get('include_depth_in_potential', False),
            ocean_current=oc,
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
        ).to(device)

    if model_type == 'unstructured_ham':
        return BASELINE_MODELS['unstructured_ham'](
            device=device, hidden_dim=hidden_dim, M_init=M_init,
            ocean_current=oc,
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
        ).to(device)

    if model_type == 'unstructured_se3':
        h = int(hidden_dim * 1.88)
        return BASELINE_MODELS['unstructured_se3'](
            device=device, hidden_dim=h,
            include_depth=kwargs.get('include_depth_in_potential', False),
            ocean_current=oc,
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
        ).to(device)

    if model_type == 'blackbox':
        h = int(hidden_dim * 1.78)
        return BASELINE_MODELS['blackbox'](
            device=device, hidden_dim=h,
            ocean_current=oc,
            T_actuator_init=kwargs.get('t_actuator_init'),
            u_act_scale=kwargs.get('u_act_scale'),
        ).to(device)

    raise ValueError(f"Unknown model_type: {model_type}")


class AUVHamNODETrainer:
    """Handles model creation, training loop, validation, and checkpointing."""

    def __init__(self, config: TrainConfig,
                 model_class=None,
                 normalizer: Optional[StateNormalizer] = None,
                 M_init=None):
        self.config = config
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
                t_actuator_init=config.t_actuator_init,
                u_act_scale=config.u_act_scale,
                u_dim=config.u_dim,
            )
        else:
            self.model = model_class(
                device=self.device,
                hidden_dim=config.hidden_dim,
                coupled_damping=config.coupled_damping,
                include_depth_in_potential=config.include_depth_in_potential,
                M_init=M_init,
                T_actuator_init=config.t_actuator_init,
                u_act_scale=config.u_act_scale,
                u_dim=config.u_dim,
            ).to(self.device)

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
        self.best_state = None

    def _run_epoch(self, loader: DataLoader,
                   t_eval: torch.Tensor, train: bool = True,
                   epoch: int = 1) -> Dict:
        """Run one epoch and return averaged metrics with failure statistics."""
        self.model.train() if train else self.model.eval()
        t_eval_dev = t_eval.to(self.device)

        totals: Dict[str, float] = defaultdict(float)
        attempted_batches = 0
        successful_batches = 0
        solver_failed_batches = 0
        invalid_prediction_batches = 0
        invalid_gradient_batches = 0

        for batch, _ in loader:
            batch = batch.to(self.device)
            attempted_batches += 1
            self.model.reset_nfe()

            batch_target = batch
            loss_start_step = 0
            if train:
                batch_input, batch_target, noise_applied = build_noisy_training_pair(
                    batch, self.config, epoch, t_eval=t_eval_dev)
                y0 = batch_input[:, 0]
                if noise_applied and self.config.drop_t0_loss and batch_target.shape[1] > 1:
                    loss_start_step = 1
            else:
                y0 = batch[:, 0]

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

            loss, comp = se3_trajectory_loss(
                batch_target,
                pred_ode,
                self.normalizer,
                weights={"actuator": self.config.actuator_loss_weight},
                so3_regularization_weight=self.config.so3_regularization_weight,
                start_step=loss_start_step,
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
            return metrics

        for k, v in totals.items():
            metrics[k] = v / successful_batches
        return metrics

    def train(self, train_loader: DataLoader,
              test_loader: DataLoader,
              t_eval: torch.Tensor) -> Dict:
        """Full training loop with validation, scheduling, and checkpointing."""
        cfg = self.config
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=[int(cfg.num_epochs * 0.8), int(cfg.num_epochs * 0.95)],
            gamma=0.1,
        )

        self.logger.info(f"Training: {cfg.num_epochs} epochs, lr={cfg.learning_rate}, "
                         f"solver={cfg.ode_solver}, so3_reg={cfg.so3_regularization_weight}")
        self.logger.info(f"Train samples: {len(train_loader.dataset)}, "
                         f"Test samples: {len(test_loader.dataset)}")

        for epoch in range(1, cfg.num_epochs + 1):
            t0 = time.time()

            train_m = self._run_epoch(train_loader, t_eval, train=True,
                                      epoch=epoch)
            if train_m["total"] == float("inf"):
                self.logger.warning(
                    f"Epoch {epoch}: no successful training batches "
                    f"(solver={int(train_m['solver_failed_batches'])}, "
                    f"pred={int(train_m['invalid_prediction_batches'])}, "
                    f"grad={int(train_m['invalid_gradient_batches'])})"
                )
                continue

            test_m = self._run_epoch(test_loader, t_eval, train=False)

            lr = self.optimizer.param_groups[0]["lr"]
            scheduler.step()
            dt = time.time() - t0

            self.metrics.log(epoch, train_m, test_m, lr, dt)

            if epoch % cfg.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch:4d} | "
                    f"Train {train_m['total']:.4e} | "
                    f"Test {test_m['total']:.4e} | "
                    f"SO3(train/test) {train_m.get('so3_orth', float('nan')):.2e}/"
                    f"{test_m.get('so3_orth', float('nan')):.2e} | "
                    f"Fail(train/test) {int(train_m['skipped_batches'])}/"
                    f"{int(test_m['skipped_batches'])} | "
                    f"LR {lr:.2e} | {dt:.1f}s"
                )

            test_loss = test_m["total"]
            if test_loss < self.best_loss:
                self.best_loss = test_loss
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

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self.metrics.save()

        summary = self.metrics.get_summary()
        if summary["best_test_loss"] is None:
            self.logger.warning("Done. No valid test epoch was recorded.")
        else:
            self.logger.info(
                f"Done. Best test loss: {summary['best_test_loss']:.4e} "
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
        model_class:  AUVHamNODE or compatible class
        M_init:       6x6 mass matrix for physics initialization (optional)

    Returns:
        model:   trained model (best checkpoint)
        summary: training summary dict
    """
    if config is None:
        config = TrainConfig()

    logger = setup_logging(config.get_run_dir())
    logger.info("Loading dataset...")

    train_loader, test_loader, t_eval, data_cfg = create_dataloaders(
        dataset_path, batch_size=config.batch_size)
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
    config.u_dim = int(data_cfg.get("u_dim", config.u_dim))
    if config.dataset_id is not None:
        logger.info("Dataset ID: %s", config.dataset_id)
    logger.info(
        "Dataset path: %s | velocity convention: %s | state dim: %s",
        config.dataset_path,
        config.dataset_velocity_convention or "n/a",
        config.dataset_state_dim if config.dataset_state_dim is not None else "n/a",
    )
    dataset = load_dataset(dataset_path)
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
    parser.add_argument("--model_type", type=str, default="hamnode",
                        choices=["hamnode", "hamnode_noinit", "hamnode_diag_d",
                                 "hamnode_no_j", "hamnode_bu_only",
                                 "hamnode_no_potential", "momentum_se3",
                                 "merged_nc", "no_potential",
                                 "unstructured_ham", "unstructured_se3",
                                 "blackbox"],
                        help="Model architecture to train (default: hamnode)")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--so3_reg", type=float, default=1e-3)
    parser.add_argument("--actuator_loss_weight", type=float, default=0.2,
                        help="Weight for supervised actuator-state loss")
    parser.add_argument("--include_depth_in_potential", action="store_true",
                        help="Condition potential / force baselines on depth")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_state_noise", action="store_true",
                        help="Enable initial navigation/actuation state perturbations")
    parser.add_argument("--observation_noise", action="store_true",
                        help="Enable correlated navigation-style observation noise")
    parser.add_argument("--observation_bias", action="store_true",
                        help="Enable persistent block-wise velocity bias")
    parser.add_argument("--noise_ramp_epochs", type=int, default=100,
                        help="Ramp training noise from 0 to full over N epochs")
    parser.add_argument("--ocean_current", action="store_true",
                        help="Enable ocean current awareness (requires 27D dataset)")
    parser.add_argument("--mass_init", type=str, default="none",
                        choices=["none", "remus", "file"],
                        help="Mass-matrix prior source")
    parser.add_argument("--mass_init_path", type=str, default=None,
                        help="Path to a .npy/.npz mass matrix file when --mass_init file")
    parser.add_argument("--t_actuator_init", type=float, nargs="+", default=None,
                        help="Actuator time-constant prior(s); length 1 or u_dim")
    parser.add_argument("--u_act_scale", type=float, nargs="+", default=None,
                        help="Actuator scaling used by B_net; length 1 or u_dim")
    args = parser.parse_args()

    config = TrainConfig(
        model_type=args.model_type,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        so3_regularization_weight=args.so3_reg,
        actuator_loss_weight=args.actuator_loss_weight,
        include_depth_in_potential=args.include_depth_in_potential,
        save_dir=args.save_dir,
        run_name=args.run_name,
        device=args.device,
        seed=args.seed,
        init_state_noise=args.init_state_noise,
        observation_noise=args.observation_noise,
        observation_bias=args.observation_bias,
        noise_ramp_epochs=args.noise_ramp_epochs,
        ocean_current=args.ocean_current,
        mass_init=args.mass_init,
        mass_init_path=args.mass_init_path,
        t_actuator_init=args.t_actuator_init,
        u_act_scale=args.u_act_scale,
    )

    M_init = _load_mass_init(args.mass_init, args.mass_init_path)
    model, summary = train_auv_hamnode(args.dataset, config, M_init=M_init)
    print(f"\nTraining finished. Results: {config.get_run_dir()}")


if __name__ == "__main__":
    main()
