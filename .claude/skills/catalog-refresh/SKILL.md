---
name: catalog-refresh
description: Rebuild `analysis/oc_data_catalog/` from current checkpoint state, then regenerate sweep summary and experiment report for a target suite. Use after new training/evaluation runs land, before citing any catalog table in docs or paper drafts.
disable-model-invocation: true
---

# Catalog Refresh

Single entry point for the three-step regeneration pipeline. Avoids the common bug where one of the steps is skipped and downstream tables drift from the source checkpoints.

## Usage

```
/catalog-refresh                          # full rebuild, no per-suite report
/catalog-refresh <suite_dir>              # full rebuild + summary + report for <suite_dir>
/catalog-refresh --suite <suite_dir>      # same, explicit
```

`<suite_dir>` is a path under `checkpoints/`, typically of the form
`checkpoints/sweep_oc_all/sweep_oc_<group>_<dataset_id>_<seeds>_<timestamp>`.

## Pipeline (run in this order, do not reorder)

1. **Pre-flight**
   - `git status --short analysis/oc_data_catalog/` — abort if dirty; catalog must be clean before regenerate
   - Record current row counts of `run_inventory.csv`, `canonical_rollout_summary_long.csv`, `rollout_run_registry.csv`

2. **Catalog rebuild**
   ```bash
   conda run -n mytorch1 python scripts/build_oc_data_catalog.py
   ```
   Must complete with exit 0. Re-record row counts; diff vs pre-flight.

3. **Sweep summary** (only if `<suite_dir>` given)
   ```bash
   conda run -n mytorch1 python scripts/summarize_sweep.py --suite-dir <suite_dir>
   ```

4. **Experiment report** (only if `<suite_dir>` given)
   ```bash
   conda run -n mytorch1 python scripts/build_experiment_report.py --suite-dir <suite_dir>
   ```

5. **Post-checks**
   - New runs that should appear: cross-check `run_inventory.csv` row delta against the directories Claude was told (or expected) to ingest
   - `canonical_run_inventory.csv` should reflect the selection policy in [docs/oc_result_selection_policy.md](docs/oc_result_selection_policy.md)
   - If `evidence_status_overrides.csv` exists, confirm overrides still match real run statuses

## Output to user

Report a short table:

| Table | Rows before | Rows after | Δ |
| --- | --- | --- | --- |
| run_inventory.csv | … | … | … |
| canonical_run_inventory.csv | … | … | … |
| canonical_rollout_summary_long.csv | … | … | … |
| rollout_run_registry.csv | … | … | … |

Plus any new run_uids added, and any unexpectedly missing ones.

## Do NOT

- Do not hand-edit any `analysis/oc_data_catalog/*.csv` — the PreToolUse hook will block this. Only `evidence_status_overrides.csv` and `run_annotations.csv` are human-maintained.
- Do not run rollout evaluation as part of this skill. If new rollouts are needed first, that's a separate `evaluate_rollout_benchmark.py` step the user does before invoking `/catalog-refresh`.
- Do not delete or move existing `analysis/oc_data_catalog/examples/` content.
