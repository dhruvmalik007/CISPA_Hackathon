#!/usr/bin/env python3
"""End-to-end check: synthetic answers -> solve.py -> organisers' scorer.

This is the test that matters before spending any budget. It writes answers in exactly the format
`query_reference.py` produces, runs the real `solve.py` as a subprocess (so its CLI path is
exercised, not just its functions), and scores the result with `score_practice.py`.

Usage:
    python scripts/e2e_check.py --n-prompts 30 --seed 0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from met.simulate import read_practice_truth, synthetic_dataset  # noqa: E402


def write_answers(data: dict[str, dict[str, str]], path: Path) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for prompt_id, responses in sorted(data.items()):
            for endpoint, text in sorted(responses.items()):
                handle.write(
                    json.dumps(
                        {
                            "endpoint": endpoint,
                            "id": prompt_id,
                            "repeat": 0,
                            "prompt": f"prompt {prompt_id}",
                            "response": text,
                        }
                    )
                    + "\n"
                )
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", action="store_true", help="keep the temp dir for inspection")
    args = parser.parse_args()

    truth_path = ROOT / "official" / "practice_answers.csv"
    truth = read_practice_truth(truth_path)
    data = synthetic_dataset(truth, n_prompts=args.n_prompts, seed=args.seed)

    tmp = Path(tempfile.mkdtemp(prefix="e2e_"))
    answers = tmp / "reference_answers.jsonl"
    count = write_answers(data, answers)
    print(f"{count} synthetic answers -> {answers}")

    solve = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "solve.py"),
            "--answers", str(answers),
            "--out", str(tmp / "answer.csv"),
            "--seed", str(args.seed),
        ],
        capture_output=True,
        text=True,
    )
    print(solve.stdout.strip()[-1500:])
    if solve.returncode != 0:
        print(solve.stderr.strip()[-1500:], file=sys.stderr)
        return solve.returncode

    score = subprocess.run(
        [sys.executable, str(ROOT / "official" / "score_practice.py"), str(tmp / "answer.csv"), str(truth_path)],
        capture_output=True,
        text=True,
    )
    print()
    print(score.stdout.strip())

    if args.keep:
        print(f"\nkept: {tmp}")
    return 0 if score.returncode == 0 else score.returncode


if __name__ == "__main__":
    raise SystemExit(main())
