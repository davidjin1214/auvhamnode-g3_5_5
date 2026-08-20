"""List regenerable rollout plots under checkpoints/ as a reviewable purge manifest.

Only PNGs inside a rollout_benchmark/ directory are candidates: they are plotted
from the CSV/JSON in the same directory, which this manifest never touches. Any
PNG referenced by a tracked file is marked keep, so a document link never dies
because of a purge.

Writes docs/checkpoints_png_purge_manifest.csv. Deleting nothing is deliberate --
review the manifest first, then delete with the paths it marks purge.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
CHECKPOINTS = REPO / "checkpoints"
OUT = REPO / "docs" / "checkpoints_png_purge_manifest.csv"


def tracked_references() -> set[str]:
    """Every checkpoints/**.png path mentioned by a tracked file."""
    names = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    refs: set[str] = set()
    for name in names:
        if not name.endswith((".md", ".csv", ".py", ".sh", ".tex", ".json", ".txt")):
            continue
        try:
            text = (REPO / name).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "checkpoints/" not in text:
            continue
        for token in text.replace("(", " ").replace(")", " ").replace("`", " ").split():
            if token.startswith("checkpoints/") and token.endswith(".png"):
                refs.add(token)
    return refs


def has_data_source(png: Path) -> bool:
    """True when the plot's eval-profile directory still holds the metrics it was drawn from.

    Layout is <rollout_run>/<profile>/{plots, diagnostic_plots/plots, *.csv, summary.json},
    so a plot in diagnostic_plots/ is checked one level up.
    """
    profile_dir = png.parent.parent if png.parent.name == "diagnostic_plots" else png.parent
    return any(f.suffix.lower() in {".csv", ".json"} for f in profile_dir.iterdir() if f.is_file())


def main() -> None:
    refs = tracked_references()
    rows = []
    for png in sorted(CHECKPOINTS.rglob("*.png")):
        rel = png.relative_to(REPO).as_posix()
        in_rollout = "/rollout_benchmark/" in f"/{rel}/" or "/rollout_benchmark/" in rel
        referenced = rel in refs
        if not in_rollout:
            action, reason = "keep", "not under rollout_benchmark/"
        elif referenced:
            action, reason = "keep", "linked from a tracked document"
        elif not has_data_source(png):
            action, reason = "keep", "no metrics file in its eval-profile dir; the plot is the only artifact"
        else:
            action, reason = "purge", "regenerable from the CSV/JSON beside it"
        rows.append(
            {
                "path": rel,
                "size_bytes": png.stat().st_size,
                "under_rollout_benchmark": int(in_rollout),
                "referenced_by_tracked_doc": int(referenced),
                "action": action,
                "reason": reason,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    purge = [r for r in rows if r["action"] == "purge"]
    keep = [r for r in rows if r["action"] == "keep"]
    total = sum(r["size_bytes"] for r in rows)
    print(f"wrote {OUT}")
    print(f"  total PNG under checkpoints/: {len(rows)}  ({total / 1024**3:.2f} GB)")
    print(f"  purge: {len(purge)}  ({sum(r['size_bytes'] for r in purge) / 1024**3:.2f} GB)")
    print(f"  keep : {len(keep)}   ({sum(r['size_bytes'] for r in keep) / 1024**2:.1f} MB)")
    print("\n  keep 明细:")
    for r in keep:
        print(f"    [{r['reason']}] {r['path']}")


if __name__ == "__main__":
    main()
