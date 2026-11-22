#!/usr/bin/env python3
"""Build the proxy model zoo from the LIVE Doubleword catalog.

Hardcoding model IDs is unsafe: `dw models list` returned 17 chat models on this date, and the docs
retired 20 older models on 30 Oct 2026. So the zoo is always derived from a captured catalog file.

Selection is driven by what the task actually needs, not by what looks impressive:

  * POSITIVES (same checkpoint, different disguise) need only ONE model - disguises are free.
  * NEGATIVES need breadth across genuinely different checkpoints. In this catalog every entry is a
    distinct checkpoint: there is no `X` and `X-FP8` pair, so the zoo CANNOT supply the hard negatives
    (same base, different quantization) that carry 90% of the competition score.

The script says that out loud in its output rather than implying the zoo covers the hard case.

Usage:
    python proxy/make_zoo.py --catalog proxy/catalog.jsonl --out proxy/zoo.json
    python proxy/make_zoo.py --catalog proxy/catalog.jsonl --out proxy/zoo.json --max 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Excluded by reason, not by taste. Both are real usage warnings from the catalog itself.
EXCLUDE = {
    "lightonai/LightOnOCR-2-1B-bbox-soup": "OCR-only; catalog warns it repeats the prompt",
    "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8": "vision-priced, most expensive entry here",
}

# Family extraction: the catalog uses vendor prefixes that are stable enough to group on.
FAMILY_RULES = [
    (r"^(Qwen)/", "qwen"),
    (r"^(zai-org)/", "glm"),
    (r"^(deepseek-ai)/", "deepseek"),
    (r"^(google)/", "gemma"),
    (r"^(meta-models|meta)/", "meta"),
    (r"^(moonshotai)/", "kimi"),
    (r"^(nvidia)/", "nemotron"),
]


def family_of(alias: str) -> str:
    for pattern, name in FAMILY_RULES:
        if re.match(pattern, alias, re.IGNORECASE):
            return name
    return alias.split("/")[0].lower()


def quant_of(alias: str) -> str:
    """Best-effort quantization tag from the alias. Absent means the provider default."""
    lowered = alias.lower()
    for tag in ("fp8", "nvfp4", "int8", "int4", "bf16"):
        if tag in lowered:
            return tag
    return "default"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="proxy/catalog.jsonl")
    parser.add_argument("--out", default="proxy/zoo.json")
    parser.add_argument("--max", type=int, default=6, help="how many distinct checkpoints to keep")
    parser.add_argument("--positives-model", default=None, help="alias to use as the positive source")
    args = parser.parse_args()

    catalog = Path(args.catalog)
    if not catalog.is_absolute():
        catalog = ROOT / catalog
    if not catalog.is_file():
        print(
            f"catalog not found: {catalog}\n"
            "Capture it first:  dw models list --type chat > proxy/catalog.jsonl",
            file=sys.stderr,
        )
        return 2

    models = []
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            alias = row.get("alias") or row.get("model_name")
            if not alias:
                continue
            models.append(
                {
                    "alias": alias,
                    "family": family_of(alias),
                    "quant": quant_of(alias),
                    "capabilities": row.get("capabilities", []),
                }
            )

    if not models:
        print(f"no models parsed from {catalog}", file=sys.stderr)
        return 2

    kept, dropped = [], []
    for model in models:
        if model["alias"] in EXCLUDE:
            dropped.append((model["alias"], EXCLUDE[model["alias"]]))
            continue
        kept.append(model)

    # One per family first, so breadth is not spent on near-duplicates, then fill by order.
    by_family: dict[str, list[dict]] = {}
    for model in kept:
        by_family.setdefault(model["family"], []).append(model)

    chosen: list[dict] = []
    for family in sorted(by_family, key=lambda f: (-len(by_family[f]), f)):
        chosen.append(by_family[family][0])
        if len(chosen) >= args.max:
            break
    if len(chosen) < args.max:
        remaining = [m for m in kept if m not in chosen]
        for model in remaining:
            if len(chosen) >= args.max:
                break
            chosen.append(model)

    positive = args.positives_model or chosen[0]["alias"]
    if not any(m["alias"] == positive for m in chosen):
        print(f"warning: --positives-model {positive} not in the chosen set; adding it", file=sys.stderr)
        match = next((m for m in kept if m["alias"] == positive), None)
        if match:
            chosen.append(match)

    print(f"catalog: {len(models)} chat models")
    print(f"chosen:  {len(chosen)} checkpoints across {len({m['family'] for m in chosen})} families\n")
    print(f"{'alias':<48}{'family':<12}{'quant':<10}role")
    for model in chosen:
        role = "POSITIVE source (disguises on this one)" if model["alias"] == positive else "negative"
        print(f"{model['alias']:<48}{model['family']:<12}{model['quant']:<10}{role}")

    if dropped:
        print("\nexcluded by reason:")
        for alias, reason in dropped:
            print(f"  {alias}: {reason}")

    # The honest caveat, printed every run so it cannot be forgotten.
    quants = {m["quant"] for m in chosen}
    same_family_variants = [
        (f, [m["alias"] for m in chosen if m["family"] == f])
        for f in {m["family"] for m in chosen}
    ]
    pairs = [(f, aliases) for f, aliases in same_family_variants if len(aliases) > 1]
    print(
        f"\nHARD-NEGATIVE CHECK: quantization tags present = {sorted(quants)}"
    )
    if pairs:
        print("  same-family groups available for hard negatives:")
        for family, aliases in pairs:
            print(f"    {family}: {aliases}")
    else:
        print(
            "  No family has two checkpoints, so this zoo CANNOT supply the hard negatives\n"
            "  (same base, different quantization) that carry 90% of the competition score.\n"
            "  It teaches 'same checkpoint different disguise' (positives) and cross-family\n"
            "  separation (the 10% term). Local GPU rows are required for the hard case."
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(catalog),
        "captured_models": len(models),
        "positives_model": positive,
        "can_supply_hard_negatives": bool(pairs),
        "models": chosen,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
