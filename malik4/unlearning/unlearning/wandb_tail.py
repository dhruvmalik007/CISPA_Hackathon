"""Tail a Task-3 training metrics.jsonl and stream each epoch to Weights & Biases.

The trainers (`unlearning.scrub` and `unlearning.salun`) write one JSON object per
epoch to `<output-dir>/metrics.jsonl`.  This process tails that file and logs each
row to W&B, so the W&B UI updates while the JURECA job is still running.  It does
not modify the trainer or the checkpoint.

Usage:
  export WANDB_API_KEY=...        # or run `python -m wandb login` once
  $VENV/bin/python -m unlearning.wandb_tail \
      --metrics /p/scratch/training2666/malik4/unlearning/output/run1-kl/metrics.jsonl \
      --project task3-unlearning \
      --run-name run1-kl \
      --stop-on-summary

For a run that already finished, omit `--stop-on-summary` only if you want the
tailer to keep waiting; otherwise it logs the existing rows and exits when
`summary.json` is present.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


def _flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(f"{prefix}.{key}" if prefix else str(key), item)
    elif isinstance(value, (int, float, bool)) and value is not None:
        yield prefix, value


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "scores":
            continue
        if isinstance(value, dict):
            out.update(_flatten(key, value))
        elif isinstance(value, (int, float, bool)) and value is not None:
            out[key] = value
    scores = row.get("scores")
    if isinstance(scores, dict):
        out.update(_flatten("", scores))
    return out


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="path to the run's metrics.jsonl")
    parser.add_argument("--project", default="task3-unlearning")
    parser.add_argument("--entity", default="", help="W&B entity; empty = default")
    parser.add_argument("--run-name", default="", help="default = metrics directory name")
    parser.add_argument("--group", default="task3-unlearning")
    parser.add_argument(
        "--mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between file checks")
    parser.add_argument("--idle-timeout", type=float, default=0.0, help="0 = wait forever")
    parser.add_argument("--skip-existing", action="store_true", help="only log rows written after start")
    parser.add_argument("--stop-on-summary", action="store_true", help="exit after summary.json is logged")
    parser.add_argument("--log-artifact", action="store_true", help="upload submission.pt as a W&B artifact")
    parser.add_argument("--config", default="", help="optional JSON config to attach at init")
    args = parser.parse_args()

    metrics_path = Path(args.metrics).expanduser().resolve()
    run_name = args.run_name or metrics_path.parent.name
    config: dict[str, Any] = {}
    if args.config:
        loaded = _read_json(Path(args.config).expanduser())
        if loaded:
            config = loaded

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        name=run_name,
        group=args.group,
        job_type="epoch-tail",
        mode=args.mode,
        config={"metrics_path": str(metrics_path), **config},
    )
    print(f"W&B run: {run.url}", flush=True)
    print(f"tailing: {metrics_path}", flush=True)

    seen = 0
    if args.skip_existing and metrics_path.exists():
        try:
            seen = len(metrics_path.read_text().splitlines())
        except OSError:
            seen = 0

    last_change = time.time()
    try:
        while True:
            lines: list[str] = []
            if metrics_path.exists():
                try:
                    lines = metrics_path.read_text().splitlines()
                except OSError:
                    lines = []

            # The trainers truncate metrics.jsonl when a run starts.
            if len(lines) < seen:
                seen = 0

            for line in lines[seen:]:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                flat = _metrics(row)
                epoch = row.get("epoch")
                step = int(epoch) if isinstance(epoch, (int, float)) else seen
                wandb.log(flat, step=step)
                print(
                    f"logged epoch={epoch} "
                    f"proxy={flat.get('proxy_score')} "
                    f"retain={flat.get('retain_score')}",
                    flush=True,
                )
                last_change = time.time()
            seen = len(lines)

            summary_path = metrics_path.parent / "summary.json"
            if args.stop_on_summary and summary_path.exists():
                summary = _read_json(summary_path)
                if summary:
                    cfg = summary.get("config")
                    if isinstance(cfg, dict):
                        run.config.update(cfg, allow_val_change=True)
                    best = summary.get("best")
                    if isinstance(best, dict):
                        run.summary["best_epoch"] = best.get("epoch")
                        run.summary["best_step"] = best.get("step")
                        run.summary["best_proxy_score"] = best.get("proxy_score")
                if args.log_artifact:
                    submission = metrics_path.parent / "submission.pt"
                    if submission.exists():
                        artifact = wandb.Artifact(name=f"{run_name}-submission", type="model")
                        artifact.add_file(str(submission))
                        run.log_artifact(artifact)
                break

            if args.idle_timeout and time.time() - last_change > args.idle_timeout:
                print("idle timeout; stopping", flush=True)
                break

            time.sleep(max(args.poll, 0.5))
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        run.finish()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
