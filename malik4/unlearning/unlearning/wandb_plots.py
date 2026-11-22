"""Generate comparison plots for all Task-3 runs in W&B.

Usage:
    python -m unlearning.wandb_plots \
        --project task3-unlearning \
        --diag-root /p/scratch/training2666/malik4/unlearning/diag_stage6_v3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "task3-unlearning"
DEFAULT_DIAG_ROOT = "/p/scratch/training2666/malik4/unlearning/diag_stage6_v3"
DEFAULT_OUT = "/p/scratch/training2666/malik4/unlearning/wandb_plots"
DEFAULT_METRICS = (
    "proxy_score",
    "retain_score",
    "forget_clean",
    "forget_altered",
    "forget_loss_mean",
    "retain_loss_mean",
    "lr",
)


def _diag_rows(root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        models = data.get("models")
        if not isinstance(models, dict):
            continue
        for name, model in models.items():
            if name == "orig" or not isinstance(model, dict):
                continue
            exact = model.get("exact") or {}
            mosaic = model.get("proxy_mosaic") or {}
            rows.append(
                {
                    "checkpoint": name,
                    "exact_loss": float(exact.get("loss_mean", -1.0)),
                    "retain": float(mosaic.get("retain_proxy_score", -1.0)),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--entity", default="")
    parser.add_argument("--diag-root", default=DEFAULT_DIAG_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--x", default="epoch")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--max-runs", type=int, default=20)
    parser.add_argument("--diag-only", action="store_true")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    import wandb
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]

    api = wandb.Api()
    entity = args.entity or api.default_entity
    if not entity and not args.diag_only:
        print("No W&B entity: pass --entity or run `python -m wandb login`.")
        return 1

    runs = []
    if not args.diag_only:
        for run in api.runs(f"{entity}/{args.project}", order="-created_at"):
            if run.job_type == "comparison-plots":
                continue
            runs.append(run)
            if len(runs) >= args.max_runs:
                break
        print(f"found {len(runs)} runs in {entity}/{args.project}")

    pngs: list[Path] = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for run in runs:
            history = run.history(keys=[args.x, metric], pandas=True)
            if history.empty:
                continue
            xcol = args.x
            if xcol not in history.columns:
                history = run.history(keys=["_step", metric], pandas=True)
                xcol = "_step"
            if history.empty or xcol not in history.columns or metric not in history.columns:
                continue
            history = history.dropna(subset=[xcol, metric])
            if history.empty:
                continue
            ax.plot(
                history[xcol],
                history[metric],
                marker="o",
                linewidth=1.4,
                markersize=3,
                label=run.name,
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel(args.x)
        ax.set_ylabel(metric)
        ax.set_title(f"Task 3 — {metric} vs {args.x}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / f"{metric}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        pngs.append(path)
        print(f"wrote {path}")

    diag = _diag_rows(args.diag_root)
    if diag:
        labels = [row["checkpoint"] for row in diag]
        exact = [row["exact_loss"] for row in diag]
        retain = [row["retain"] for row in diag]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(labels, exact, color="#4c72b0")
        axes[0].axhline(1.4, color="#dd8452", linestyle="--", label="target 1.4")
        axes[0].axhline(2.2, color="#dd8452", linestyle="--", label="target 2.2")
        axes[0].set_title("Exact loss")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].legend(fontsize=7)
        axes[1].bar(labels, retain, color="#55a868")
        axes[1].axhline(0.90, color="#c44e52", linestyle="--", label="target 0.90")
        axes[1].set_title("Retain proxy")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        path = out_dir / "diagnostics.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        pngs.append(path)
        print(f"wrote {path}")

    if not pngs:
        print("No plots generated; check run history keys and --diag-root.")
        return 1

    if not args.no_log:
        with wandb.init(
            project=args.project,
            entity=args.entity or None,
            name="comparison-plots",
            job_type="comparison-plots",
        ) as run:
            for path in pngs:
                run.log({f"plots/{path.stem}": wandb.Image(str(path))})
            print(f"W&B comparison run: {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
