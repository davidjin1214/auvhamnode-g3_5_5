"""Build a machine-readable retention manifest for checkpoints/.

Usage:
    python scripts/build_checkpoints_retention_manifest.py [repo_root]

Classifies every run directory (a directory holding config.json) into a
retention class using the vocabulary already established in
docs/repo_structure_audit.md and the R-A evidence rule in
docs/section8_evidence_merge_plan.md, and records whether any tracked file in
the repository refers to it.

Sizes are logical file sizes, not on-disk allocation: most of checkpoints/ is
OneDrive cloud placeholders that occupy no local blocks.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]

CHECKPOINTS = REPO / "checkpoints"
OUT = REPO / "docs" / "checkpoints_retention_manifest.csv"

CORE_ARTIFACTS = {
    "config.json",
    "training_history.pkl",
    "best_model.pt",
    "block_evaluation.json",
    "heldout_evaluation.json",
}


def classify(suite_rel: str) -> tuple[str, str]:
    """Return (retention_class, note) for a suite path relative to checkpoints/.

    Order matters. "smoke" in a suite name does not imply flow-validation: the
    smoke3 suites physically hold 27 of the 45 cleanrun-v1 decision runs
    (docs/experiment_full_inventory_zh.md section B.2/B.6), so they are matched
    before any generic smoke rule.
    """
    low = suite_rel.lower()
    if low.startswith("unused/"):
        return (
            "delete-candidate",
            "old noise design, invalid for current evidence (repo_structure_audit 2.8)",
        )
    if "smoke1" in low:
        return (
            "flow-validation-only",
            "single-seed/single-model smoke; superseded by the smoke3 and t2_wpfrag suites",
        )
    if low.split("/")[0] == "smoke_v4lite":
        return (
            "flow-validation-only",
            "pure code/protocol smoke at ep=3, no rollout (inventory B.3)",
        )
    if re.search(r"probe|phase1_smoke|seed42_smoke", low) or low.split("/")[0] == "sweep_oc_smoke":
        return (
            "flow-validation-only",
            "flow/protocol validation only, never paper evidence (repo_structure_audit 2.6)",
        )
    if "smoke3" in low and "cleanrun_v1" in low:
        return (
            "evidence-bearing-B",
            "holds 27 of the 45 cleanrun-v1 decision runs (seeds 42/44/46); named smoke but is evidence, see inventory B.2/B.6",
        )
    if "extra43-45" in low and "cleanrun_v1" in low:
        return (
            "evidence-bearing-B",
            "holds 18 of the 45 cleanrun-v1 decision runs (seeds 43/45)",
        )
    if "t2_wpfrag" in low:
        return (
            "evidence-bearing-B",
            "B-zone t2_wpfrag decision suite; primary section 8 evidence under the R-A rule",
        )
    if "cleanrun_v1" in low:
        return (
            "evidence-bearing-B",
            "cleanrun-v1 aggregation view; read with docs/phase1a_oc_v4lite_cleanrun_v1_report.md",
        )
    if low.startswith(("sweep_oc_all/", "sweep_oc_all_noise/")) or low.startswith(
        ("sweep_oc_main_noise_", "sweep_oc_key_ablation_noise_")
    ):
        return (
            "evidence-bearing-A",
            "A-zone catalog-era suite; still cited for diag_damping / bu_only / black-box noise line (R-A rule 2)",
        )
    return ("other", "unclassified, review before any action")


def tracked_text_blobs() -> list[tuple[str, str]]:
    names = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    blobs = []
    for name in names:
        if not name.endswith((".md", ".csv", ".py", ".sh", ".tex", ".json", ".txt")):
            continue
        path = REPO / name
        try:
            blobs.append((name, path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    return blobs


def dir_stats(path: Path) -> tuple[int, int]:
    total = 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for fname in files:
            try:
                total += (Path(root) / fname).stat().st_size
            except OSError:
                continue
            count += 1
    return total, count


def main() -> None:
    blobs = tracked_text_blobs()
    rows = []
    for config in sorted(CHECKPOINTS.rglob("config.json")):
        run_dir = config.parent
        suite_dir = run_dir.parent
        suite_rel = suite_dir.relative_to(CHECKPOINTS).as_posix()
        run_rel = run_dir.relative_to(REPO).as_posix()
        retention_class, note = classify(suite_rel)
        size, count = dir_stats(run_dir)
        present = {f.name for f in run_dir.iterdir() if f.is_file()}
        has_core = sorted(CORE_ARTIFACTS & present)
        referenced = sorted(
            {name for name, text in blobs if run_dir.name in text and suite_dir.name in text}
        )
        rows.append(
            {
                "suite_dir": suite_dir.relative_to(REPO).as_posix(),
                "run_dir": run_rel,
                "retention_class": retention_class,
                "size_bytes": size,
                "file_count": count,
                "has_core_artifacts": ";".join(has_core),
                "referenced_by": ";".join(referenced),
                "note": note,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "suite_dir",
                "run_dir",
                "retention_class",
                "size_bytes",
                "file_count",
                "has_core_artifacts",
                "referenced_by",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT} with {len(rows)} run rows")
    summary: dict[str, list[int]] = {}
    for row in rows:
        acc = summary.setdefault(row["retention_class"], [0, 0, 0])
        acc[0] += 1
        acc[1] += row["size_bytes"]
        acc[2] += row["file_count"]
    print(f"{'retention_class':<24}{'runs':>6}{'GB':>9}{'files':>10}")
    for name, (n, size, files) in sorted(summary.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:<24}{n:>6}{size / 1024**3:>9.2f}{files:>10}")


if __name__ == "__main__":
    main()
