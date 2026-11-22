#!/usr/bin/env python3
"""Normalize Doubleword batch results into one flat training corpus.

The batch API returns each result keyed by `custom_id`, which already encodes the ground truth
(`ep|family|model|disguise|prompt|sample`). This script decodes it, extracts the VISIBLE answer, and
writes one JSON line per response with the labels attached. Pair construction happens at training
time so this file stays a simple corpus.

Two deliberate behaviours, both learned from the pilot:

  * responses whose `content` is None are counted and DROPPED, never silently backfilled from
    `reasoning_content`. The pilot had 9/36 such rows when reasoning was left enabled; training on
    reasoning text would teach the model to match hidden thoughts the scored endpoints do not emit.
  * every `(model, disguise)` cell is checked for completeness. A truncated batch would leave cells
    with fewer samples, which biases the positive pairs toward whichever cells survived.

Usage:
    python proxy/build_training_set.py --results proxy/pilot4_results.jsonl \
        --out training/proxy_pairs.jsonl
    python proxy/build_training_set.py --results proxy/pilot4_results.jsonl \
        --out training/proxy_pairs.jsonl --min-samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def decode_custom_id(custom_id: str) -> dict[str, str] | None:
    parts = custom_id.split("|")
    if len(parts) != 6:
        return None
    prefix, family, model, disguise, prompt, sample = parts
    return {
        "prefix": prefix,
        "family": family,
        "model": model,
        "disguise": disguise,
        "prompt": prompt,
        "sample": sample,
    }


def visible_content(row: dict) -> str | None:
    """Return the visible answer, or None. Never falls back to reasoning_content."""
    body = (row.get("response") or {}).get("body")
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            content = (choices[0].get("message") or {}).get("content")
            if isinstance(content, str) and content.strip():
                return content
    choices = row.get("choices") or []
    if choices and isinstance(choices[0], dict):
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str) and content.strip():
            return content
    content = row.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def finish_reason(row: dict) -> str | None:
    body = (row.get("response") or {}).get("body") or {}
    choices = body.get("choices") or row.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def prompt_text(row: dict) -> str:
    """Recover the prompt from the request body so the pair text can be built without a side file."""
    body = (row.get("request") or {}).get("body") or {}
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", default=["proxy/proxy_results.jsonl"])
    parser.add_argument("--out", default="training/proxy_pairs.jsonl")
    parser.add_argument("--min-samples", type=int, default=1,
                        help="minimum rows required per (model, disguise, prompt) cell")
    parser.add_argument("--allow-empty-cells", action="store_true",
                        help="write the corpus even if some cells are short (not recommended)")
    args = parser.parse_args()

    rows: list[dict] = []
    seen_ids: set[str] = set()
    stats = Counter()

    for name in args.results:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            print(f"missing results file: {path}", file=sys.stderr)
            return 2
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                stats["lines"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                custom_id = str(row.get("custom_id", ""))
                meta = decode_custom_id(custom_id)
                if meta is None:
                    stats["bad_custom_id"] += 1
                    continue
                if custom_id in seen_ids:
                    stats["duplicate"] += 1
                    continue
                seen_ids.add(custom_id)
                content = visible_content(row)
                if content is None:
                    stats["no_visible_content"] += 1
                    reason = finish_reason(row)
                    stats[f"empty_{reason}"] += 1
                    continue
                stats["kept"] += 1
                rows.append(
                    {
                        "prompt": prompt_text(row),
                        "response": content,
                        "text": f"[PROMPT] {prompt_text(row)}\n[RESPONSE] {content}",
                        "family": meta["family"],
                        "model": meta["model"],
                        "disguise": meta["disguise"],
                        "prompt_id": meta["prompt"],
                        "sample": meta["sample"],
                        "source": "doubleword",
                    }
                )

    print(f"lines read:        {stats['lines']}")
    print(f"kept:              {stats['kept']}")
    print(f"no visible answer: {stats['no_visible_content']} (dropped, never backfilled from reasoning)")
    print(f"duplicates:        {stats['duplicate']}")
    print(f"unparseable:       {stats['bad_json'] + stats['bad_custom_id']}")
    if stats["no_visible_content"]:
        print("  by finish_reason:", {k[6:]: v for k, v in stats.items() if k.startswith("empty_")})

    if not rows:
        print("no usable rows", file=sys.stderr)
        return 2

    # Balance check: a short cell would skew the positive pairs it participates in.
    cells: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        cells[(row["model"], row["disguise"], row["prompt_id"])] += 1
    counts = Counter(cells.values())
    short = {k: v for k, v in cells.items() if v < args.min_samples}
    print(f"\ncells: {len(cells)}  rows-per-cell distribution: {dict(sorted(counts.items()))}")
    print(f"cells below min-samples={args.min_samples}: {len(short)}")
    for key, value in list(short.items())[:8]:
        print(f"  {key}: {value}")

    models = sorted({r["model"] for r in rows})
    disguises = sorted({r["disguise"] for r in rows})
    prompts = sorted({r["prompt_id"] for r in rows})
    print(f"\nmodels: {len(models)} {models}")
    print(f"disguises: {len(disguises)} {disguises}")
    print(f"prompts: {len(prompts)}")

    # Positive pairs available for SupCon: same (model, prompt), different disguise.
    positives = 0
    per_cell: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        per_cell[(row["model"], row["prompt_id"])] += 1
    for count in per_cell.values():
        positives += count * (count - 1) // 2
    print(f"positive pairs (same model+prompt, differing disguise): {positives}")
    print(f"hard negatives available: 0  (this catalog has no same-family quantization variants)")

    if short and not args.allow_empty_cells:
        print(
            f"\nREFUSING to write: {len(short)} cell(s) have fewer than {args.min_samples} rows.\n"
            "Re-collect the missing rows, or pass --allow-empty-cells to accept a biased corpus.",
            file=sys.stderr,
        )
        return 1

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nwrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
