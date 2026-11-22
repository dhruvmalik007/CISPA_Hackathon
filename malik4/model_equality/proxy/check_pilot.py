#!/usr/bin/env python3
"""Check the pilot batch before scaling up: do the disguises actually change the output?

The whole proxy-zoo premise is that a system prompt and a temperature shift change the *surface* of a
response while the underlying model stays the same. If the API ignores system prompts, `clean` and
`sys_concise` come back identical, there is nothing to learn, and a 7,680-request batch would buy a
training set of duplicates.

This script answers that with numbers instead of faith, for a cost of ~36 requests:

  1. per-disguise response length and variance  -> did the system prompt bite?
  2. pairwise similarity between disguises      -> are they distinguishable at all?
  3. self-similarity within a disguise          -> is the sampling actually stochastic?
  4. per-disguise obedience check               -> did "be concise" shorten the answer?

Exit code 1 if the disguises are indistinguishable, so a broken pilot cannot silently proceed.

Usage:
    python proxy/check_pilot.py --results proxy/pilot_results.jsonl
    python proxy/check_pilot.py --results proxy/pilot_results.jsonl --min-separation 0.02
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_custom_id(custom_id: str) -> dict[str, str]:
    parts = custom_id.split("|")
    if len(parts) != 6:
        return {}
    prefix, family, model, disguise, prompt, sample = parts
    return {
        "prefix": prefix,
        "family": family,
        "model": model,
        "disguise": disguise,
        "prompt": prompt,
        "sample": sample,
    }


def extract_text(row: dict) -> tuple[str | None, str | None]:
    """Return (visible content, reasoning content).

    Reasoning models can spend the entire token budget in `reasoning_content` and leave `content` as
    None. That is not a parsing bug - it is a real and disqualifying outcome, because the competition's
    signal is the visible 256-token answer. So the two are returned separately and the caller counts
    the empty-content case rather than silently falling back to reasoning text.
    """
    body = (row.get("response") or {}).get("body")
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            reasoning = message.get("reasoning_content")
            return (
                content if isinstance(content, str) else None,
                reasoning if isinstance(reasoning, str) else None,
            )
    choices = row.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content, message.get("reasoning_content")
    if isinstance(row.get("content"), str):
        return row["content"], None
    return None, None


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="proxy/pilot_results.jsonl")
    parser.add_argument("--min-separation", type=float, default=0.02,
                        help="disguises must differ by at least this much mean similarity")
    parser.add_argument("--max-concise-ratio", type=float, default=0.85,
                        help="sys_concise should be at most this fraction of clean's length")
    args = parser.parse_args()

    path = Path(args.results)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        print(f"results not found: {path}", file=sys.stderr)
        return 2

    # group[disguise][(model, prompt)] -> list of responses
    grouped: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
    errors = 0
    unparsed = 0
    empty_content = 0
    starved: list[str] = []
    total = 0
    finish_reasons: dict[str, int] = defaultdict(int)

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                unparsed += 1
                continue
            meta = parse_custom_id(str(row.get("custom_id", "")))
            if not meta:
                unparsed += 1
                continue
            body = (row.get("response") or {}).get("body") or {}
            choices = body.get("choices") or []
            reason = choices[0].get("finish_reason") if choices else None
            finish_reasons[str(reason)] += 1

            content, reasoning = extract_text(row)
            if content is None:
                errors += 1
                if reasoning:
                    # The budget went entirely to hidden reasoning: no visible answer to compare.
                    empty_content += 1
                    starved.append(meta["disguise"])
                continue
            grouped[meta["disguise"]][(meta["model"], meta["prompt"])].append(content)

    if not grouped:
        print(f"no usable responses in {path} ({total} lines, {errors} empty, {unparsed} unparsed)", file=sys.stderr)
        return 2

    print(f"{path.name}: {total} lines, {errors} without visible content, {unparsed} unparsed custom_id")
    print(f"finish_reason: {dict(finish_reasons)}")
    if empty_content:
        print(
            f"\n{empty_content}/{total} responses returned NO visible content - the token budget was "
            f"consumed by reasoning_tokens (by disguise: "
            f"{dict(sorted(((d, starved.count(d)) for d in set(starved))))})."
        )
        print("  Fix: raise --max-tokens and/or send reasoning_effort=none (already the default in")
        print("  make_proxy_batch.py after this was measured).")
    disguises = sorted(grouped)
    print(f"\ndisguises present: {disguises}\n")

    lengths: dict[str, np.ndarray] = {}
    print(f"{'disguise':<14}{'n':>5}{'mean_tok':>10}{'std_tok':>9}{'mean_chars':>12}")
    for disguise in disguises:
        texts = [t for rs in grouped[disguise].values() for t in rs]
        lengths[disguise] = np.array([len(tokens(t)) for t in texts], dtype=float)
        print(
            f"{disguise:<14}{len(texts):>5}{lengths[disguise].mean():>10.1f}"
            f"{lengths[disguise].std():>9.1f}{np.mean([len(t) for t in texts]):>12.0f}"
        )

    # Did the system prompt actually bite?
    if "clean" in lengths and "sys_concise" in lengths:
        ratio = lengths["sys_concise"].mean() / max(1e-9, lengths["clean"].mean())
        verdict = "OBEYED" if ratio <= args.max_concise_ratio else "NOT OBEYED"
        print(f"\nsys_concise / clean length ratio: {ratio:.2f}  -> system prompt {verdict}")
    else:
        ratio = None

    # Are the disguises distinguishable at all, and is sampling stochastic?
    print(f"\n{'pair':<32}{'mean_sim':>10}{'n_pairs':>9}")
    separation_rows = []
    keys = sorted({k for d in disguises for k in grouped[d]})
    for i, left in enumerate(disguises):
        for right in disguises[i + 1 :]:
            sims = []
            for key in keys:
                for a in grouped[left].get(key, []):
                    for b in grouped[right].get(key, []):
                        sims.append(jaccard(tokens(a), tokens(b)))
            if sims:
                separation_rows.append((left, right, float(np.mean(sims)), len(sims)))
                print(f"{left + ' vs ' + right:<32}{np.mean(sims):>10.3f}{len(sims):>9}")

    print(f"\n{'within-disguise self-similarity':<32}{'mean_sim':>10}{'n_pairs':>9}")
    self_rows = []
    for disguise in disguises:
        sims = []
        for key, responses in grouped[disguise].items():
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    sims.append(jaccard(tokens(responses[i]), tokens(responses[j])))
        if sims:
            self_rows.append((disguise, float(np.mean(sims)), len(sims)))
            print(f"{disguise:<32}{np.mean(sims):>10.3f}{len(sims):>9}")

    # Verdict.
    problems = []
    if empty_content:
        rate = empty_content / max(1, total)
        if rate > 0.02:
            problems.append(
                f"{empty_content}/{total} responses ({rate:.0%}) had no visible content because the "
                "token budget was consumed by reasoning; the comparison below is computed on a biased "
                "subset. Re-run with --max-tokens 1024 (or reasoning disabled)."
            )
    if ratio is not None and ratio > args.max_concise_ratio:
        problems.append(
            f"system prompt not obeyed (concise/clean length ratio {ratio:.2f} > {args.max_concise_ratio}); "
            "positives would be duplicates"
        )
    if separation_rows:
        weakest = min(row[2] for row in separation_rows)
        distinct = [row for row in separation_rows if row[2] < 1.0 - args.min_separation]
        print(
            f"\nweakest cross-disguise separation: {weakest:.3f} "
            f"({len(distinct)}/{len(separation_rows)} pairs differ by >= {args.min_separation})"
        )
        if not distinct:
            problems.append(
                "every disguise pair is near-identical; the API is not varying output, "
                "so there is nothing for a verifier to learn"
            )
    if self_rows and all(row[1] > 0.99 for row in self_rows):
        problems.append("responses are deterministic (within-disguise similarity ~1.0); "
                        "sampling is not stochastic, so repeats add no information")

    print()
    if problems:
        print("PILOT FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nDo NOT run the full batch. The premise that disguises create learnable variation is unmet.")
        return 1
    print("PILOT PASSED: disguises produce distinguishable, non-identical output. Safe to scale up.")
    print("next:  python proxy/make_proxy_batch.py --zoo proxy/zoo.json --samples 4 --out proxy/proxy_batch.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
