#!/usr/bin/env python3
"""Build a submission using ONLY what the high-power test proved is detectable.

The verdict (`training-first-session/VERDICT.md`) established two measured facts on 648 purpose-built
reference answers:

  1. Subtle same-model vs different-model separation is at chance (AUROC 0.508). No threshold on text
     similarity works, so claiming "same model" for the subtle variants loses points.
  2. Gross divergence IS detectable with no overlap: `ref-07` (LoRA 1.5) has minimum MMD^2 0.0386
     against every other endpoint while all other pairs top out at 0.0122. Splitting just that
     endpoint off scores MCC 0.331.

So this builder does not pretend to solve the model question. It:

  * flags endpoints that are grossly divergent from the crowd, using robust z-scores on several
    per-endpoint statistics (median length, response-length spread, catalogue divergence), and
  * puts everything else in one family with one model label.

That is the highest-expected-value submission the evidence supports. Expected score is modest.

Two modes, because which one is right depends on whether the 33 real endpoints actually contain
gross outliers:

    --families gross     split off outliers, group the rest  (the recommended default)
    --families all-one   everything in one family/model      (honest fallback if no outlier stands out)
    --families all-distinct  every endpoint alone            (scores exactly 0; for format testing only)

Usage:
    python scripts/build_submission.py --answers data/answers.jsonl --out out/submission_v1.csv
    python scripts/build_submission.py --answers data/answers.jsonl --dry-run
    python scripts/build_submission.py --answers data/answers.jsonl --z 2.5 --out out/v2.csv
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


def load_answers(path: Path) -> dict[str, dict[str, str]]:
    """{prompt_id: {endpoint: response}}, keeping the first answer per (prompt, endpoint)."""
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
            if not endpoint or prompt_id is None or response is None:
                continue
            key = str(prompt_id)
            if endpoint not in by_prompt[key]:
                by_prompt[key][endpoint] = str(response)
    return dict(by_prompt)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def robust_z(values: dict[str, float]) -> dict[str, float]:
    """Median/MAD z-scores. MAD is not inflated by the outliers we are trying to detect."""
    keys = sorted(values)
    array = np.array([values[k] for k in keys], dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    if mad == 0:
        # Degenerate: fall back to standard deviation, and if that is zero too, no outliers.
        std = float(array.std())
        if std == 0:
            return {k: 0.0 for k in keys}
        return {k: float((values[k] - median) / std) for k in keys}
    return {k: 0.6745 * (values[k] - median) / mad for k in keys}


def per_endpoint_stats(data: dict[str, dict[str, str]], endpoints: list[str]) -> dict[str, dict[str, float]]:
    """Statistics that are gross enough to survive watermarks and sampling noise.

    Text similarity was tested to chance, so these are deliberately not similarity measures. They
    capture how much and how varied each endpoint's output is, which quantization and adapter
    changes perturb more reliably than content.
    """
    stats: dict[str, dict[str, float]] = {}
    for endpoint in endpoints:
        responses = [rows[endpoint] for rows in data.values() if endpoint in rows]
        if not responses:
            continue
        words = [len(tokens(r)) for r in responses]
        chars = [len(r) for r in responses]
        # Distinct 4-grams across all of this endpoint's answers: a crude "vocabulary size".
        vocab: set[str] = set()
        for r in responses:
            text = " ".join(str(r).lower().split())
            vocab.update(text[i : i + 4] for i in range(max(1, len(text) - 3)))
        stats[endpoint] = {
            "answers": float(len(responses)),
            "med_words": float(np.median(words)),
            "mean_words": float(np.mean(words)),
            "std_words": float(np.std(words)),
            "med_chars": float(np.median(chars)),
            "vocab": float(len(vocab)),
        }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", default="data/answers.jsonl")
    parser.add_argument("--out", default="out/submission_v1.csv")
    parser.add_argument("--z", type=float, default=1.5, help="|robust z| above which an endpoint is grossly divergent")
    parser.add_argument("--max-outliers", type=int, default=4, help="refuse to call more than this many outliers")
    parser.add_argument("--families", default="gross", choices=["gross", "all-one", "all-distinct"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = ROOT / answers
    if not answers.exists():
        print(f"answers file not found: {answers}", file=sys.stderr)
        return 2

    data = load_answers(answers)
    endpoints = sorted({ep for rows in data.values() for ep in rows if ep.startswith("ep-")})
    prefix = "ep-"
    if not endpoints:
        endpoints = sorted({ep for rows in data.values() for ep in rows if ep.startswith("ref-")})
        prefix = "ref-"
    if not endpoints:
        print("no endpoints found in the answers file", file=sys.stderr)
        return 2

    stats = per_endpoint_stats(data, endpoints)
    counts = {ep: int(stats[ep]["answers"]) for ep in endpoints}
    print(f"{answers.name}: {len(endpoints)} {prefix}* endpoints, {len(data)} prompt ids")
    print(f"answers per endpoint: {min(counts.values())}-{max(counts.values())}")
    if min(counts.values()) < 3:
        print(f"WARNING: only {min(counts.values())} answers for some endpoints; estimates will be noisy",
              file=sys.stderr)

    # Score each endpoint on each gross statistic, then combine.
    signals = ["med_words", "std_words", "med_chars", "vocab"]
    zs = {name: robust_z({ep: stats[ep][name] for ep in endpoints}) for name in signals}
    combined = {ep: max(abs(zs[name][ep]) for name in signals) for ep in endpoints}
    worst_signal = {ep: max(signals, key=lambda name: abs(zs[name][ep])) for ep in endpoints}
    z = {ep: zs[worst_signal[ep]][ep] for ep in endpoints}

    if args.families == "all-one":
        labels = {ep: ("F1", "F1_M1") for ep in endpoints}
        outliers: list[str] = []
    elif args.families == "all-distinct":
        labels = {ep: (f"F{i}", f"F{i}_M1") for i, ep in enumerate(endpoints, start=1)}
        outliers = list(endpoints)
    else:
        outliers = sorted((ep for ep in endpoints if combined[ep] >= args.z), key=lambda e: -combined[e])
        if len(outliers) > args.max_outliers:
            print(
                f"\nREFUSING: {len(outliers)} endpoints exceed |z|>={args.z}, but at most "
                f"{args.max_outliers} can plausibly be gross outliers. A large count means the spread is "
                f"uniform, so no threshold is meaningful here.\n"
                f"Options: raise --z (e.g. 2.5), or use --families all-one.",
                file=sys.stderr,
            )
            outliers = outliers[: args.max_outliers]
            print(f"Proceeding with only the {len(outliers)} most extreme: {outliers}", file=sys.stderr)
        core = [ep for ep in endpoints if ep not in outliers]
        if not core:
            print("ERROR: every endpoint was flagged as an outlier; use --families all-one", file=sys.stderr)
            return 2
        labels = {ep: ("F1", "F1_M1") for ep in core}
        for index, ep in enumerate(outliers, start=2):
            labels[ep] = (f"F{index}", f"F{index}_M1")

    print(f"\nmode: {args.families}")
    if outliers:
        print(f"set apart ({len(outliers)}): {outliers}")
    print(f"core group: {sum(1 for ep in endpoints if labels[ep][0] == 'F1')} endpoints")

    print(f"\n{'endpoint':<9}{'answers':>8}{'med_w':>7}{'std_w':>7}{'med_ch':>8}{'vocab':>7}{'z':>7}  signal       family/model")
    for ep in sorted(endpoints, key=lambda e: -combined[e]):
        s = stats[ep]
        family, model = labels[ep]
        print(f"{ep:<9}{counts[ep]:>8}{s['med_words']:>7.1f}{s['std_words']:>7.1f}"
              f"{s['med_chars']:>8.0f}{s['vocab']:>7.0f}{combined[ep]:>7.2f}  {worst_signal[ep]:<12} {family}/{model}")

    n_core = sum(1 for ep in endpoints if labels[ep][0] == "F1")
    total_pairs = len(endpoints) * (len(endpoints) - 1) // 2
    print(f"\nsame-family pairs claimed: {n_core * (n_core - 1) // 2} of {total_pairs}")
    print(
        "\nExpected score: modest (reference measurement: MCC 0.331 for splitting one gross outlier).\n"
        "The 90% model term needs the subtle variants separated, and that is where AUROC measured 0.5."
    )

    # Format checks before writing: a malformed CSV wastes a 30-minute submission slot.
    assert len(labels) == len(endpoints), "every endpoint must have a label exactly once"
    for family, model in labels.values():
        assert family and model, "blank label"
        assert "," not in family and "," not in model, "labels must not contain commas"
    by_model: dict[str, set[str]] = defaultdict(set)
    for family, model in labels.values():
        by_model[model].add(family)
    conflicts = {m: f for m, f in by_model.items() if len(f) > 1}
    assert not conflicts, f"a model label spans several families: {conflicts}"

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if args.dry_run:
        print(f"\nDRY RUN - would write {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        for ep in endpoints:
            family, model = labels[ep]
            writer.writerow([ep, family, model])
    print(f"\nwrote {out} ({len(endpoints)} endpoint rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
