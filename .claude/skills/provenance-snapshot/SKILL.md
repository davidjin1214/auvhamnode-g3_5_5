---
name: provenance-snapshot
description: Capture a reproducibility snapshot (git SHA, dataset checksum, conda env, CLI invocation, config hash) for a training or evaluation run directory under `checkpoints/<run>/provenance/`. Use after a new training run finishes, or to retro-snapshot a historical run before treating it as paper evidence.
disable-model-invocation: true
---

# Provenance Snapshot

Produce the same evidence pack that `analysis/provenance_audit/phase1_static/run_lock.md` collected manually,
but for any run directory the user names — written into `<run_dir>/provenance/`.

## Usage

```
/provenance-snapshot <run_dir>
```

`<run_dir>` must contain `config.json`. Both forms work:
- absolute path
- path relative to repo root (e.g. `checkpoints/sweep_oc_all/.../main_phnode_full_seed46`)

## Snapshot Contents

Create `<run_dir>/provenance/` containing:

| File | Source |
| --- | --- |
| `run_lock.md` | Markdown table mirroring [phase1_static/run_lock.md](analysis/provenance_audit/phase1_static/run_lock.md) — suite, run_uid, dataset path/id, train_type, noise profile, num_epochs, best_epoch, best_loss, status |
| `cli_invocation.txt` | Full CLI from `config.json` (reconstructed via `argparse._actions` defaults vs config) — same shape as [phase1_static/cleanrun_train_invocation.txt](analysis/provenance_audit/phase1_static/cleanrun_train_invocation.txt) |
| `dataset_sha256.txt` | `sha256` of the pickle in `config.json::dataset` |
| `config_sha256.txt` | `sha256` of `config.json` itself |
| `git_sha.txt` | `git rev-parse HEAD` + `git status --short` at snapshot time |
| `git_diff.patch` | `git diff HEAD` so any uncommitted drift is captured |
| `pip_freeze.txt` | `conda run -n mytorch1 pip freeze` |
| `conda_env.txt` | `conda run -n mytorch1 conda list --explicit` |
| `snapshot_metadata.json` | `{snapshot_time, run_dir, host, user}` |

## Steps for Claude

1. Verify `<run_dir>/config.json` exists. If not, abort with a clear error.
2. `mkdir -p <run_dir>/provenance`
3. Compute and write each artifact above. Use:
   - `sha256sum` or `shasum -a 256` for checksums (mac-friendly)
   - `git -C <repo_root> rev-parse HEAD` (run from repo root, not run_dir)
   - `conda run -n mytorch1 pip freeze`
4. Read `config.json` to derive the CLI invocation. For each key that maps to a `--flag` in [train_utils.py](train_utils.py), emit `--flag value`. Skip keys whose value equals the argparse default.
5. Pull `best_epoch` and `best_loss` from `training.log` if present (look for `best test loss` lines), else from `training_history.pkl`.
6. Write `run_lock.md` using the same table format as the reference file. One section per run.
7. Append a one-line entry to `analysis/provenance_audit/snapshot_log.csv` with: `timestamp,run_dir,git_sha,dataset_sha256,config_sha256,status`. Create the CSV with header if missing.
8. Report a summary table showing key fields and the snapshot path.

## Idempotence

If `<run_dir>/provenance/` already exists, prompt before overwriting. Older snapshots can be moved to `provenance/.archive/<timestamp>/` rather than deleted.

## Do NOT

- Do not edit any file under `analysis/oc_data_catalog/*.csv` (the PreToolUse hook will block you anyway — that's a guardrail, not a target).
- Do not write to the run's `config.json`, `best_model.pt`, or `training_history.pkl`. Snapshot is read-only on those.
- Do not run training or evaluation — this skill is pure capture, not reproduction. For reproduction, see the audit notebooks under `notebook/phase3_provenance_audit_*`.
