#!/usr/bin/env python3
"""Run the method on the practice truth and score it with the organisers' own scorer.

Two things are verified here that the in-process score cannot tell you:

1. The output file is in the exact format `score_practice.py` expects (header, one row per
   endpoint, family/model labels without commas).
2. The MCC matches what the organisers compute, using their code rather than mine.

The practice set is a single family, so `MCC_family` is mathematically degenerate there (every pair
is truly "same family", leaving no true negatives). `score_practice.py` therefore scores only the
model question, which is the number to optimise on practice.

Usage:
    python scripts/validate_offline.py --n-prompts 40 --seed 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from met.client import write_submission  # noqa: E402
from met.simulate import decide, read_practice_truth, synthetic_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=40)
    parser.add_argument("--prompts-per-endpoint", type=int, default=None,
                        help="simulate a budget cap; default: one per prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(ROOT / "out" / "practice_answer.csv"))
    args = parser.parse_args()

    truth_path = ROOT / "official" / "practice_answers.csv"
    truth = read_practice_truth(truth_path)
    endpoints = sorted(truth)

    data = synthetic_dataset(truth, n_prompts=args.n_prompts, seed=args.seed)
    if args.prompts_per_endpoint is not None:
        # Drop prompts per endpoint to mimic a budget-truncated collection.
        trimmed: dict[str, dict[str, str]] = {}
        for index, (pid, responses) in enumerate(sorted(data.items())):
            if index >= args.prompts_per_endpoint:
                break
            trimmed[pid] = responses
        data = trimmed

    labels, tests = decide(data, endpoints, seed=args.seed)
    out = Path(args.out)
    write_submission([(ep, labels[ep][0], labels[ep][1]) for ep in endpoints], out)

    print(f"wrote {out}")
    print(f"pairs tested: {len(tests)}  significant: {sum(1 for t in tests if t.adjusted_p <= 0.05)}")

    result = subprocess.run(
        [sys.executable, str(ROOT / "official" / "score_practice.py"), str(out), str(truth_path)],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
