#!/usr/bin/env python3
"""THE GATE: does a proxy-trained verifier transfer to reference data it has never seen?

This is the step that protects the remaining query budget. Everything before it is cheap; everything
after it spends responses that cannot be recovered.

What it measures, in order of importance:

  1. **Reference AUROC** - the verifier's embedding distance on `ref-01..ref-09`, scored against the
     published `practice_answers.csv`. These endpoints were never in the proxy zoo, so this is a real
     held-out test of transfer.
  2. **Comparison to the hand-built baseline** - char-4gram similarity scored 0.508 on the same data.
     A learned verifier has to beat that to justify its complexity.
  3. **Proxy AUROC** - on its own training distribution, as a sanity ceiling. A large gap between
     proxy and reference AUROC is the domain-shift warning.

Exit codes:
    0  gate passed (reference AUROC >= threshold)
    1  gate FAILED - do not spend ep-* budget
    2  cannot run (missing inputs)

The gate defaults to 0.65: comfortably above the 0.508 baseline, and below the 0.887 the proxy itself
achieves, so it demands genuine transfer without demanding a miracle.

Usage:
    python training/eval_transfer.py --verifier training/verifier.pt --answers data/ref_power.jsonl
    python training/eval_transfer.py --answers data/ref_power.jsonl --metric-only   # baseline only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GATE_THRESHOLD = 0.65
BASELINE_AUROC = 0.508  # measured: char-4gram on ref-01..ref-09 at maximum power


def load_truth(path: Path) -> dict[str, tuple[str, str]]:
    truth: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            truth[row["endpoint"]] = (row["family"], row["model"])
    return truth


def load_pooled(path: Path, prefix: str) -> dict[str, dict[str, list[str]]]:
    pooled: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
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
            if not endpoint.startswith(prefix) or row.get("response") is None:
                continue
            pooled[str(row["id"])][endpoint].append(str(row["response"]))
    return {pid: dict(eps) for pid, eps in pooled.items()}


def char4(text: str) -> set[str]:
    lowered = " ".join(str(text).lower().split())
    if len(lowered) < 4:
        return {lowered} if lowered else set()
    return {lowered[i : i + 4] for i in range(len(lowered) - 3)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def auroc(same: np.ndarray, diff: np.ndarray) -> float:
    """P(random same pair scores higher than random different pair)."""
    same = same[np.isfinite(same)]
    diff = diff[np.isfinite(diff)]
    if same.size == 0 or diff.size == 0:
        return float("nan")
    wins = sum(float((same > v).sum()) + 0.5 * float((same == v).sum()) for v in diff)
    return wins / (same.size * diff.size)


def pair_scores_from_text(pooled, endpoints, encoder=None) -> dict[tuple[str, str], float]:
    """Mean similarity per endpoint pair. Uses the encoder if given, else char-4gram."""
    scores: dict[tuple[str, str], float] = {}
    for i, left in enumerate(endpoints):
        for right in endpoints[i + 1 :]:
            values = []
            for _pid, rows in pooled.items():
                if left not in rows or right not in rows:
                    continue
                for a in rows[left]:
                    for b in rows[right]:
                        if encoder is None:
                            values.append(jaccard(char4(a), char4(b)))
                        else:
                            values.append(encoder(a, b))
            scores[(left, right)] = float(np.mean(values)) if values else float("nan")
    return scores


def evaluate(scores: dict[tuple[str, str], float], truth: dict[str, tuple[str, str]]) -> dict:
    same_vals, diff_vals = [], []
    for (left, right), value in scores.items():
        if not np.isfinite(value):
            continue
        if truth[left][1] == truth[right][1]:
            same_vals.append(value)
        else:
            diff_vals.append(value)
    same, diff = np.array(same_vals), np.array(diff_vals)
    return {
        "auroc": auroc(same, diff),
        "same_mean": float(same.mean()) if same.size else float("nan"),
        "diff_mean": float(diff.mean()) if diff.size else float("nan"),
        "n_same": int(same.size),
        "n_diff": int(diff.size),
        "margin": float(same.mean() - diff.mean()) if same.size and diff.size else float("nan"),
    }


def make_encoder(verifier_path: Path):
    """Load a trained verifier and return an (a, b) -> similarity function."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    payload = torch.load(verifier_path, map_location="cpu", weights_only=False)
    model_name = payload.get("base_model", "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    state = payload.get("state_dict") or payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"note: {len(missing)} parameter(s) not in checkpoint (expected if the head was dropped)")
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cache: dict[str, np.ndarray] = {}

    def embed(text: str) -> np.ndarray:
        if text in cache:
            return cache[text]
        batch = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**batch).last_hidden_state[:, 0]
        vector = out[0].float().cpu().numpy()
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        cache[text] = vector
        return vector

    def similarity(a: str, b: str) -> float:
        return float(embed(a) @ embed(b))

    return similarity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/ref_power.jsonl")
    parser.add_argument("--truth", default=str(ROOT / "official" / "practice_answers.csv"))
    parser.add_argument("--verifier", default=None)
    parser.add_argument("--proxy-results", default=None, help="also report AUROC on the proxy corpus")
    parser.add_argument("--threshold", type=float, default=GATE_THRESHOLD)
    parser.add_argument("--metric-only", action="store_true", help="baseline char-4gram only; no torch")
    args = parser.parse_args()

    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = ROOT / answers
    if not answers.is_file():
        print(f"missing answers: {answers}", file=sys.stderr)
        return 2
    truth = load_truth(Path(args.truth))
    endpoints = sorted(truth)
    pooled = load_pooled(answers, "ref-")
    complete = {pid: rows for pid, rows in pooled.items() if all(ep in rows for ep in endpoints)}
    if len(complete) < 3:
        print(f"need >=3 complete prompts across {len(endpoints)} endpoints, have {len(complete)}", file=sys.stderr)
        return 2

    print(f"reference data: {answers.name}  {len(complete)} prompts x {len(endpoints)} endpoints")
    print(f"baseline to beat: AUROC {BASELINE_AUROC:.3f} (char-4gram, same data)\n")

    baseline = evaluate(pair_scores_from_text(complete, endpoints), truth)
    print("baseline (char-4gram):")
    print(f"  AUROC {baseline['auroc']:.3f}   same {baseline['same_mean']:.4f}  diff {baseline['diff_mean']:.4f} "
          f"(n={baseline['n_same']}/{baseline['n_diff']})")

    if args.metric_only or not args.verifier:
        print("\n(no --verifier given, so the gate is not evaluated)")
        return 0

    verifier = Path(args.verifier)
    if not verifier.is_absolute():
        verifier = ROOT / verifier
    if not verifier.is_file():
        print(f"missing verifier: {verifier}", file=sys.stderr)
        return 2

    encoder = make_encoder(verifier)
    trained = evaluate(pair_scores_from_text(complete, endpoints, encoder), truth)
    print("\nverifier:")
    print(f"  AUROC {trained['auroc']:.3f}   same {trained['same_mean']:.4f}  diff {trained['diff_mean']:.4f} "
          f"(n={trained['n_same']}/{trained['n_diff']})")
    print(f"  improvement over baseline: {trained['auroc'] - baseline['auroc']:+.3f}")

    passed = np.isfinite(trained["auroc"]) and trained["auroc"] >= args.threshold
    print()
    if passed:
        print(f"GATE PASSED: reference AUROC {trained['auroc']:.3f} >= {args.threshold:.2f}")
        print("The verifier transfers to unseen endpoints, so spending ep-* budget is justified.")
        print("Next: python training/select_prompts.py, then a 2-prompt Stage-A spend (16/endpoint).")
        return 0
    print(f"GATE FAILED: reference AUROC {trained['auroc']:.3f} < {args.threshold:.2f}")
    print(
        "\nDo NOT spend ep-* budget. The verifier learned the proxy distribution but does not transfer\n"
        "to the endpoints that matter. Record this in training-first-session/ROUND-2.md and treat the\n"
        "90% model term as not recoverable from this API's visible text."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
