#!/usr/bin/env python3
"""Reproduce the headline numbers in VERDICT.md, and assert them.

An independent check should be able to run one command and confirm the verdict. This prints the two
claims that matter and exits non-zero if the data no longer supports them:

  1. maximum-power text similarity is at chance (AUROC ~0.5), so `1st-try.md`'s core does not work;
  2. a gross-divergence split is nevertheless detectable and worth MCC ~0.33, so a non-zero
     submission exists.

Usage:
    python scripts/verify_verdict.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_pooled(path: Path):
    pooled = defaultdict(lambda: defaultdict(list))
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            endpoint = str(row.get("endpoint") or "")
            if endpoint.startswith("ref-") and row.get("response") is not None:
                pooled[str(row["id"])][endpoint].append(str(row["response"]))
    return pooled


def load_truth(path: Path):
    truth = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def mmd2(a, b) -> float:
    sa, sb = [tokens(x) for x in a], [tokens(y) for y in b]

    def k(u, v):
        if not u and not v:
            return 1.0
        if not u or not v:
            return 0.0
        return len(u & v) / len(u | v)

    m, n = len(sa), len(sb)
    if m < 2 or n < 2:
        return float("nan")
    wa = sum(k(sa[i], sa[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    wb = sum(k(sb[i], sb[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    cr = sum(k(x, y) for x in sa for y in sb) / (m * n)
    return wa + wb - 2 * cr


def auroc_lower_is_same(same, diff) -> float:
    same = [v for v in same if np.isfinite(v)]
    diff = [v for v in diff if np.isfinite(v)]
    if not same or not diff:
        return float("nan")
    wins = sum(sum(1 for d in diff if d > v) + 0.5 * sum(1 for d in diff if d == v) for v in same)
    return wins / (len(same) * len(diff))


def mcc(tp, tn, fp, fn) -> float:
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return max(0.0, (tp * tn - fp * fn) / denominator) if denominator else 0.0


def main() -> int:
    answers = ROOT / "data" / "ref_power.jsonl"
    if not answers.exists():
        print(f"missing {answers}; run scripts/collect_ref.py first", file=sys.stderr)
        return 2
    truth = load_truth(ROOT / "official" / "practice_answers.csv")
    endpoints = sorted(truth)
    pooled = load_pooled(answers)
    usable = {p: e for p, e in pooled.items() if all(ep in e for ep in endpoints)}

    per_endpoint = min(min(len(e[ep]) for ep in endpoints) for e in usable.values())
    total = len(usable) * len(endpoints) * per_endpoint
    print(f"data: {len(usable)} complete prompts x {len(endpoints)} endpoints x {per_endpoint} repeats = {total} answers\n")

    pairs = [(a, b) for i, a in enumerate(endpoints) for b in endpoints[i + 1 :]]
    is_same = {p: truth[p[0]][1] == truth[p[1]][1] for p in pairs}

    mmd = {}
    for left, right in pairs:
        a = [r for pid in usable for r in usable[pid][left]]
        b = [r for pid in usable for r in usable[pid][right]]
        mmd[(left, right)] = mmd2(a, b)

    same = [mmd[p] for p in pairs if is_same[p]]
    diff = [mmd[p] for p in pairs if not is_same[p]]
    auc = auroc_lower_is_same(same, diff)
    print(f"CLAIM 1  maximum-power text similarity is at chance")
    print(f"  AUROC(same-model closer) = {auc:.3f}   (0.5 = chance)")
    print(f"  same-model mean {np.mean(same):.5f}   different-model mean {np.mean(diff):.5f}")

    worst = max(same)
    second = sorted(diff)[1] if len(diff) > 1 else float("nan")
    print(f"\n  highest same-model MMD2 {worst:.5f} vs lowest different-model MMD2 {min(diff):.5f}")
    print(f"  -> classes overlap, which is why AUROC is {auc:.3f}")

    # Gross-divergence split: the endpoint that is furthest from everyone else.
    centrality = {ep: float(np.mean([mmd[p] for p in pairs if ep in p])) for ep in endpoints}
    outlier = max(centrality, key=centrality.get)
    outlier_min = min(mmd[p] for p in pairs if outlier in p)
    rest_max = max(mmd[p] for p in pairs if outlier not in p)
    separated = outlier_min > rest_max

    tp = tn = fp = fn = 0
    for left, right in pairs:
        truly_same = is_same[(left, right)]
        called_same = outlier not in (left, right)
        if truly_same and called_same:
            tp += 1
        elif not truly_same and not called_same:
            tn += 1
        elif not truly_same:
            fp += 1
        else:
            fn += 1
    value = mcc(tp, tn, fp, fn)

    print(f"\nCLAIM 2  gross divergence IS detectable")
    print(f"  most divergent endpoint: {outlier}")
    print(f"  its minimum MMD2 {outlier_min:.5f} vs max among the rest {rest_max:.5f} -> separated: {separated}")
    print(f"  MCC of '{outlier} alone, rest together' = {value:.3f}  (tp/tn/fp/fn = {tp}/{tn}/{fp}/{fn})")

    ok = True
    if not (0.35 <= auc <= 0.65):
        print(f"\nFAIL: claim 1 expected AUROC near 0.5, got {auc:.3f}", file=sys.stderr)
        ok = False
    if value < 0.2:
        print(f"\nFAIL: claim 2 expected MCC >= 0.2, got {value:.3f}", file=sys.stderr)
        ok = False
    print("\n" + ("verdict reproduced" if ok else "verdict NOT reproduced"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
