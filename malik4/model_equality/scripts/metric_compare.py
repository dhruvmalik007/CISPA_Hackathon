#!/usr/bin/env python3
"""Compare candidate similarity metrics on REAL reference answers.

My char-4gram Jaccard scored AUROC 0.508 on real data - useless - despite scoring 1.000 on my own
simulator. The simulator encoded my assumptions, so it could not falsify them. This script tests
several metrics on the real responses and reports AUROC for each, so the choice is made from
evidence rather than intuition.

AUROC = P(random same-model pair scores above a random different-model pair).
1.0 perfect, 0.5 useless, <0.5 inverted.

Usage:
    python scripts/metric_compare.py --answers data/falcon2/ref_calib.jsonl
    python scripts/metric_compare.py --answers data/falcon2/ref_p2.jsonl --per-prompt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_real_answers(path: Path) -> dict[str, dict[str, str]]:
    by_prompt: dict[str, dict[str, str]] = defaultdict(dict)
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
            prompt_id = row.get("id")
            response = row.get("response")
            if not endpoint.startswith("ref-") or prompt_id is None or response is None:
                continue
            by_prompt[str(prompt_id)][endpoint] = str(response)
    return dict(by_prompt)


def load_truth(path: Path) -> dict[str, tuple[str, str]]:
    truth: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def auroc(same: np.ndarray, diff: np.ndarray) -> float:
    if same.size == 0 or diff.size == 0:
        return float("nan")
    wins = sum(float((diff < v).sum()) + 0.5 * float((diff == v).sum()) for v in same)
    return wins / (same.size * diff.size)


# --------------------------------------------------------------------------------------
# candidate metrics
# --------------------------------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def m_jaccard_char4(a: str, b: str) -> float:
    """Current production metric."""
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    ga = {na[i : i + 4] for i in range(len(na) - 3)} if len(na) >= 4 else {na}
    gb = {nb[i : i + 4] for i in range(len(nb) - 3)} if len(nb) >= 4 else {nb}
    return len(ga & gb) / len(ga | gb) if (ga | gb) else 1.0


def m_token_jaccard(a: str, b: str) -> float:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def m_token_multiset(a: str, b: str) -> float:
    ca, cb = Counter(_tokens(a)), Counter(_tokens(b))
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    overlap = sum((ca & cb).values())
    return 2 * overlap / (sum(ca.values()) + sum(cb.values()))


def _ngrams(text: str, n: int) -> Counter:
    toks = _tokens(text)
    if len(toks) < n:
        return Counter(toks)
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def m_token_bigram(a: str, b: str) -> float:
    ca, cb = _ngrams(a, 2), _ngrams(b, 2)
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    return 2 * sum((ca & cb).values()) / (sum(ca.values()) + sum(cb.values()))


def m_length(a: str, b: str) -> float:
    """Response length similarity. Crude, but quantization and LoRA often change verbosity."""
    la, lb = len(_tokens(a)), len(_tokens(b))
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(la, lb) if max(la, lb) else 0.0


def m_prefix(a: str, b: str) -> float:
    """Exact prefix agreement over the first 40 characters.

    Targets the fact that these are continuation prompts: identical weights should produce the
    identical opening tokens even when the tail diverges.
    """
    na, nb = " ".join(a.lower().split()), " ".join(b.lower().split())
    k = 40
    pa, pb = na[:k], nb[:k]
    if not pa and not pb:
        return 1.0
    same = sum(1 for x, y in zip(pa, pb) if x == y)
    return same / k


def m_first_tokens(a: str, b: str, k: int = 8) -> float:
    ta, tb = _tokens(a)[:k], _tokens(b)[:k]
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    same = sum(1 for x, y in zip(ta, tb) if x == y)
    return same / k


METRICS = {
    "jaccard_char4": m_jaccard_char4,
    "token_jaccard": m_token_jaccard,
    "token_multiset": m_token_multiset,
    "token_bigram": m_token_bigram,
    "length": m_length,
    "prefix40": m_prefix,
    "first8_tokens": m_first_tokens,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--per-prompt", action="store_true", help="also report per prompt id")
    parser.add_argument("--min-endpoints", type=int, default=9)
    args = parser.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.is_absolute():
        answers_path = ROOT / args.answers
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    data_all = load_real_answers(answers_path)
    paired = {p: r for p, r in data_all.items() if all(ep in r for ep in endpoints)}
    if not paired:
        print(f"no fully-paired prompts in {answers_path.name}", file=sys.stderr)
        return 2
    prompt_ids = sorted(paired)
    print(f"{answers_path.name}: {len(prompt_ids)} fully-paired prompts x {len(endpoints)} endpoints\n")

    is_same = {}
    for i, left in enumerate(endpoints):
        for right in endpoints[i + 1 :]:
            is_same[(left, right)] = truth[left][1] == truth[right][1]

    print(f"{'metric':<18}{'AUROC':>8}{'same':>9}{'diff':>9}{'gap':>8}")
    best = (None, -1.0)
    for name, fn in METRICS.items():
        same_vals, diff_vals = [], []
        for (left, right), same in is_same.items():
            vals = [fn(paired[p][left], paired[p][right]) for p in prompt_ids]
            mean = float(np.mean(vals))
            (same_vals if same else diff_vals).append(mean)
        auc = auroc(np.asarray(same_vals), np.asarray(diff_vals))
        gap = float(np.mean(same_vals) - np.mean(diff_vals))
        print(f"{name:<18}{auc:>8.3f}{np.mean(same_vals):>9.3f}{np.mean(diff_vals):>9.3f}{gap:>8.3f}")
        if auc > best[1]:
            best = (name, auc)

    print(f"\nbest: {best[0]} (AUROC {best[1]:.3f})")

    if args.per_prompt and best[0]:
        fn = METRICS[best[0]]
        print(f"\nper-prompt AUROC for {best[0]} (which prompts carry the signal):")
        print(f"{'prompt':<22}{'AUROC':>8}{'same':>8}{'diff':>8}")
        for pid in prompt_ids:
            sv, dv = [], []
            for (left, right), same in is_same.items():
                value = fn(paired[pid][left], paired[pid][right])
                (sv if same else dv).append(value)
            auc = auroc(np.asarray(sv), np.asarray(dv))
            print(f"{pid:<22}{auc:>8.3f}{np.mean(sv):>8.3f}{np.mean(dv):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
