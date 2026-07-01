# Repository Guidelines

## Role And Scope

This is an active research codebase for **AUV dynamics modeling on `SE(3)`**, using structured port-Hamiltonian Neural ODEs, ablations, black-box baselines, clean/noisy initial-condition training, and long-horizon rollout benchmarks. It also contains the `paper/` workspace for the AUVHamNODE thesis chapter.

Treat this file as an agent operating guide, not as a replacement for `README.md`, `docs/`, experiment reports, or paper notes. Keep future updates concise and move detailed evidence narratives into the appropriate documentation.

## Operating Rules

- Work from an expert, evidence-aware perspective. Do not turn a number into a claim until its source, protocol, and evidence status are clear.
- Run local Python/experiment commands in the Conda environment `mytorch1`; `deepxiv` is the routine exception.
- Prefer documented workflows, catalog/current-evidence tables, and provenance notes over ad hoc checkpoint browsing.
- Keep code clean, concise, modular, and consistent with local style. Comments must be in English and only explain non-obvious logic.
- Do not rewrite or delete user/generated research artifacts unless explicitly asked.

## Orientation

Start here for current repo state:

- `README.md`
- `EXPERIMENT_PROGRESS_TRACKER.md`
- `docs/repo_structure_audit.md`
- `docs/experiment_stages_overview.md`

Use these for evidence and catalog interpretation:

- `docs/provenance_audit_phnode_full_clean.md`
- `docs/phase1a_oc_v4lite_cleanrun_v1_report.md`
- `docs/oc_result_selection_policy.md`
- `docs/oc_data_catalog_dictionary.md`
- `analysis/oc_data_catalog/`

Use these for protocol and command details:

- `docs/noise_experiment_runbook.md`
- `docs/noise_cli_parameter_reference.md`
- `docs/noise_cli_command_templates.md`

For thesis-chapter work, start from:

- `paper/README.md`
- `paper/drafts/auvhamnode_thesis_chapter_zh.tex`

## Active Code Map

- `remus100_core.py`: REMUS100 simulator core
- `data_collection.py`: dataset generation
- `AUVHamNODE.py`: main structured model
- `auv_baselines.py`: baselines and ablations
- `auv_model_registry.py`: model registry
- `train_auv_hamnode.py`, `train_utils.py`: training entrypoint and utilities
- `evaluate_rollout_benchmark.py`, `rollout_benchmark_engine.py`, `rollout_benchmark_reporting.py`: rollout benchmark stack
- `scripts/train_all_models_noise_profile.sh`, `scripts/eval_all_models_noise_profile.sh`: current sweep wrappers
- `scripts/build_oc_data_catalog.py`: OC catalog generation
- `scripts/export_section8_t2_evidence.py`: thesis §8 current-evidence export

Legacy/reference boundaries:

- `original/bf3n/`: legacy reference, not active implementation
- `checkpoints/unused/`: old incorrect noise design, invalid for current evidence
- smoke/probe checkpoint directories: flow validation only, not paper evidence

## Commands

```bash
conda activate mytorch1
```

Single-run workflow:

```bash
python data_collection.py --num_traj 500 --blocks 150 --seed 42 --save_dir ./data/oc --workers 4 --ocean_current --current_speed_max 0.5
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints
python train_auv_hamnode.py --dataset ./data/oc/<dataset>.pkl --model_type phnode_full --save_dir ./checkpoints --noise_profile nominal_train --noise_warmup_epochs 20 --noise_ramp 80 --noise_mix_ratio 0.5
python evaluate_rollout_benchmark.py --checkpoint ./checkpoints/<run>/best_model.pt --mode resampled --noise_profiles clean nominal_eval degraded_eval heading_biased_eval --output_dir ./checkpoints/<run>/rollout_benchmark
```

Recommended sweep and reporting workflow:

```bash
bash scripts/train_all_models_noise_profile.sh --profile oc --group core --noise-profile nominal_train
bash scripts/eval_all_models_noise_profile.sh --suite-dir ./checkpoints/<suite>
python scripts/summarize_sweep.py --suite-dir ./checkpoints/<suite>
python scripts/build_experiment_report.py --suite-dir ./checkpoints/<suite>
python scripts/build_oc_data_catalog.py
```

Deprecated paths:

- `scripts/train_all_models_noise.sh`
- `scripts/eval_all_models_noise.sh`
- training CLI `--noise_level`

## Evidence Rules

- Before citing catalog rows, check `evidence_status` and `analysis/oc_data_catalog/evidence_status_overrides.csv`.
- `is_canonical = 1` means “selected default rollout,” not “safe current paper evidence.”
- Catalog-era `phnode_full clean seed42/46` rows are `stale_environment_drift`; do not use the old ~11 m 5-seed mean as current model-fragility evidence.
- Use `analysis/section8_current_evidence/aggregate.csv` and `per_seed_long.csv` for thesis §8 current-evidence claims; regenerate them via `scripts/export_section8_t2_evidence.py`.
- Phase-1A/T2 results are separate from the canonical OC catalog. `scripts/build_oc_data_catalog.py` intentionally excludes `sweep_oc_phase1a_*` suites to prevent smoke/decision suites from leaking into canonical views.
- For new evidence-bearing runs, record `_audit_meta/code_revision.txt` and `_audit_meta/environment.txt`.

## Generated Files

Do not hand-edit generated outputs:

- `data/*`
- `checkpoints/*`
- generated `analysis/oc_data_catalog/*.csv`
- generated `analysis/section8_current_evidence/*.csv`

The hand-editable catalog sidecar is:

- `analysis/oc_data_catalog/evidence_status_overrides.csv`

If catalog behavior must change, edit the relevant script, then regenerate:

- `scripts/build_oc_data_catalog.py`
- `scripts/query_oc_catalog_examples.py`
- `scripts/oc_catalog_templates.py`

If thesis §8 evidence behavior must change, edit `scripts/export_section8_t2_evidence.py`, then regenerate `analysis/section8_current_evidence/`.

## Paper Boundaries

For `paper/` work:

- The method contribution is AUV structured continuous-time dynamics modeling; long-horizon state prediction is the validation task.
- Do not claim a fully closed, strict port-Hamiltonian AUV system.
- Do not claim ordinary ODE integration strictly preserves `SO(3)`.
- Do not present `B_net` as a standard port-Hamiltonian input matrix `G(q)u`.
- Do not model ocean current as a closed Hamiltonian environment subsystem.
- Do not present `v4_lite` as a confirmed superior final training protocol.
- Formal Chinese thesis prose must not contain internal planning notes, implementation labels, or memo-style guidance.

## Style And Validation

- Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, concise module docstrings where useful.
- No dedicated `tests/` directory exists. Validate with the smallest affected workflow: one data generation, one training job, one rollout, one sweep summary, one catalog rebuild, or one §8 export as appropriate.
- For docs-only edits, inspect the diff and verify referenced paths/commands exist.
- When changing experiment logic, record the command, output path, and key structured outputs such as `config.json`, `heldout_evaluation.json`, rollout `summary.json`, or exported CSVs.

## Commit And PR

Keep commits short and task-focused. Concise Chinese or English subjects are both fine.

Examples:

- `Refine OC catalog canonical export`
- `更新 noisy sweep 汇总逻辑`

PRs should explain motivation, modified entrypoints, validation commands actually run, and any changed reports, plots, or catalog outputs.
