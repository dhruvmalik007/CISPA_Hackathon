"""Run the shared diagnose.py over every saved step checkpoint.

Usage:
    python -m unlearning.run_diag_checkpoints
    python -m unlearning.run_diag_checkpoints --skip-existing
    python -m unlearning.run_diag_checkpoints --dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CHECKPOINTS = "/p/scratch/training2666/malik4/unlearning/output/stage6-match-clean"
DEFAULT_DIAGNOSE = "/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/diagnose.py"
DEFAULT_MODEL = "/p/scratch/training2666/malik4/unlearning/model.pt"
DEFAULT_FORGET = "/p/scratch/training2666/malik4/unlearning/forget_images.npz"
DEFAULT_OUT = "/p/scratch/training2666/malik4/unlearning/diag_stage6_clean"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--diagnose-py", default=DEFAULT_DIAGNOSE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--forget", default=DEFAULT_FORGET)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=3500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(os.path.join(args.checkpoints_dir, "step-*.pt")))
    submission = Path(args.checkpoints_dir) / "submission.pt"
    if submission.is_file():
        checkpoints.append(str(submission))
    if not checkpoints:
        print(f"no step-*.pt or submission.pt under {args.checkpoints_dir}")
        return 1

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    for checkpoint in checkpoints:
        name = Path(checkpoint).stem
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and (out_dir / "summary.json").is_file():
            print(f"skip existing {name}")
            continue
        command = [
            sys.executable,
            args.diagnose_py,
            "--model",
            args.model,
            "--forget",
            args.forget,
            "--limit",
            str(args.limit),
            "--batch",
            str(args.batch),
            "--subs",
            checkpoint,
            "--out",
            str(out_dir),
        ]
        print("+ " + " ".join(command), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        log_path = out_dir / "diagnose.log"
        log_path.write_text(result.stdout)
        print(result.stdout, end="")
        if result.returncode:
            print(f"diagnose failed rc={result.returncode} for {checkpoint}", file=sys.stderr)

    print("\ndone. Print the gate table with:")
    print(f"  {sys.executable} -m unlearning.diag_gate_table --root {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
