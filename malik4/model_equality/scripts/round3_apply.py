#!/usr/bin/env python3
"""Calibrate the ft01 edit-distance threshold on refs, then label the scored endpoints.

The reference gate is enforced: the scored file is only written when the reference MCC is >= 0.90,
unless --force is passed.

Usage:
    python scripts/round3_apply.py \
        --ref data/ref_round3_ft01.jsonl \
        --apply data/ep_round3_ft01.jsonl \
        --out out/submission_round3.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from round3_eval import (  # noqa: E402
    average_grid,
    best_threshold,
    cluster,
    labels_from,
    load_pooled,
    load_truth,
    pair_distances,
    practice_mcc,
    score_with_official,
)


def build_grid(pooled: dict[str, dict[str, list[str]]], endpoints: list[str], metric: str, min_repeats: int):
    usable = {
        pid: per_ep
        for pid, per_ep in pooled.items()
        if all(ep in per_ep and len(per_ep[ep]) >= min_repeats for ep in endpoints)
    }
    if not usable:
        raise SystemExit(f"no prompt has >= {min_repeats} repeats for all {len(endpoints)} endpoints")
    selected = sorted(usable)
    distances = {pid: pair_distances(usable[pid], endpoints)[metric] for pid in selected}
    grid, counts = average_grid(distances, selected, endpoints)
    return grid, selected, counts


def write_submission(labels: dict[str, tuple[str, str]], endpoints: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        for endpoint in endpoints:
            writer.writerow([endpoint, *labels[endpoint]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="data/ref_round3_ft01.jsonl")
    parser.add_argument("--apply", default="data/ep_round3_ft01.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--out", default="out/submission_round3.csv")
    parser.add_argument("--metric", choices=["exact", "edit", "char3", "composite"], default="edit")
    parser.add_argument("--min-repeats", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    truth = load_truth(Path(args.truth))
    ref_endpoints = sorted(truth)
    ref_pooled = load_pooled(ROOT / args.ref)
    ref_grid, ref_prompts, _ = build_grid(ref_pooled, ref_endpoints, args.metric, args.min_repeats)
    tau, practice, groups = best_threshold(ref_grid, ref_endpoints, truth)
    labels = labels_from(cluster(ref_grid, tau), ref_endpoints)
    official = score_with_official(labels, ref_endpoints)
    print(f"reference metric={args.metric} prompts={len(ref_prompts)}")
    print(f"threshold={tau:.6f} practice_mcc={practice:.3f} groups={groups} official={official:.3f}")
    if practice < 0.90:
        print("WARNING: reference practice MCC is below 0.90")
        if not args.force:
            print("refusing to apply; pass --force to override")
            return 2

    apply_path = ROOT / args.apply
    if not apply_path.is_file():
        print(f"scored answers not found: {apply_path}")
        return 2
    scored = load_pooled(apply_path, prefix="ep-")
    endpoints = sorted({ep for rows in scored.values() for ep in rows})
    grid, prompts, _ = build_grid(scored, endpoints, args.metric, args.min_repeats)
    roots = cluster(grid, tau)
    scored_labels = labels_from(roots, endpoints)
    out = ROOT / args.out
    write_submission(scored_labels, endpoints, out)
    print(f"wrote {out}: {len(endpoints)} endpoints, {len(set(roots))} groups, prompts={prompts}")
    print("labels:")
    for endpoint in endpoints:
        print(f"  {endpoint} {scored_labels[endpoint][0]} {scored_labels[endpoint][1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
