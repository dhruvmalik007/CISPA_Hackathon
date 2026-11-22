#!/usr/bin/env python3
"""Validate a submission CSV against the organisers' stated contract.

A rejected submission still burns the 30-minute cooldown, and so does a malformed one that scores 0
for the wrong reason. This checks every requirement from the task PDF and `submission.py` before a
slot is spent:

  * one line per endpoint, `endpoint,family,model`, header optional
  * labels are free text with **no commas**
  * same `model` label implies the same `family` label
  * no duplicate endpoints, no blank labels, no unknown endpoint names
  * the answer is not one of the two known-degenerate forms (both score exactly 0)

It also prints the pair counts the score will be computed from, so you can see what you are claiming
before you claim it.

Usage:
    python scripts/validate_submission.py out/submission_v1.csv
    python scripts/validate_submission.py out/submission_v1.csv --expect 33
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_PREFIX = "ep-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--expect", type=int, default=33, help="expected number of endpoints")
    parser.add_argument("--prefix", default=EXPECTED_PREFIX)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        print(f"file not found: {path}", file=sys.stderr)
        return 2
    if path.suffix.lower() != ".csv":
        print(f"must be a .csv, got {path.name}", file=sys.stderr)
        return 2

    problems: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [[cell.strip() for cell in row] for row in csv.reader(handle)]
    rows = [r for r in rows if any(c for c in r)]

    if not rows:
        print("file is empty", file=sys.stderr)
        return 2
    if rows[0][:3] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    elif len(rows[0]) >= 3 and rows[0][0].startswith(args.prefix):
        print("note: no header row (allowed)")

    body = [r for r in rows if len(r) >= 3]
    if len(body) != len(rows):
        problems.append(f"{len(rows) - len(body)} row(s) have fewer than 3 columns")

    endpoints = [r[0] for r in body]
    duplicates = [e for e, n in Counter(endpoints).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate endpoints: {duplicates}")
    if len(set(endpoints)) != args.expect:
        problems.append(f"expected {args.expect} endpoints, found {len(set(endpoints))}")
    unexpected = [e for e in endpoints if not e.startswith(args.prefix)]
    if unexpected:
        problems.append(f"endpoints not starting with {args.prefix!r}: {unexpected[:5]}")
    blanks = [r[0] for r in body if not r[1] or not r[2]]
    if blanks:
        problems.append(f"blank family/model label for: {blanks[:5]}")

    # A model label must never span two families: the PDF requires model => same family.
    families_for_model: dict[str, set[str]] = defaultdict(set)
    for endpoint, family, model, *_ in body:
        families_for_model[model].add(family)
    conflicts = {m: sorted(f) for m, f in families_for_model.items() if len(f) > 1}
    if conflicts:
        problems.append(f"model label used in several families: {dict(list(conflicts.items())[:3])}")

    families = {r[1] for r in body}
    models = {r[2] for r in body}
    n = len(set(endpoints))
    total_pairs = n * (n - 1) // 2
    same_family_pairs = sum(count * (count - 1) // 2 for count in Counter(r[1] for r in body).values())
    same_model_pairs = sum(count * (count - 1) // 2 for count in Counter(r[2] for r in body).values())

    print(f"file: {path}")
    print(f"endpoints: {n}   families: {len(families)}   models: {len(models)}")
    print(f"pairs: {total_pairs} total")
    print(f"  claimed same-family: {same_family_pairs} ({100 * same_family_pairs / total_pairs:.1f}%)")
    print(f"  claimed same-model : {same_model_pairs} ({100 * same_model_pairs / total_pairs:.1f}%)")

    # The two degenerate answers score exactly 0, because each zeroes a factor of tp*tn - fp*fn.
    if len(families) == n and len(models) == n:
        problems.append(
            "all-distinct: every endpoint has its own family and model. This scores EXACTLY 0 "
            "(tp=0 and fn=0). Not worth a submission slot."
        )
    if len(families) == 1 and len(models) == 1:
        problems.append(
            "all-same: one family and one model for everything. This also scores EXACTLY 0 "
            "(tn=0 and fp=0)."
        )

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nvalid: format matches the organisers' contract")
    print("reminder: the public score is only a rough check; the final ranking uses unseen pairs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
