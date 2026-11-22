#!/usr/bin/env python3
"""Build the prompt battery, with the design rationale that makes it work.

The central design insight, and the reason a generic prompt set would fail:

    A *high-constraint* prompt suppresses deployment-level variation (system prompt, persona,
    decoding temperature, watermark) while preserving weight-level variation (quantization, LoRA).

That is exactly the discrimination the task demands. Consider "continue this sentence exactly":
a persona instruction has almost nothing to act on, so all deployments of one checkpoint produce
nearly identical text — but a quantized copy differs slightly in token choice and a LoRA adapter
differs more. Conversely a free-form "write a poem about X" prompt is dominated by persona and
sampling variance, which are *identity-preserving* disguises, so it burns budget measuring noise.

The battery therefore leans heavily on constrained continuations, and includes a small number of
probes for the other failure direction: tasks where a LoRA adapter would show clearly (style
transfer, structured output) so the method does not mistake "follows the format the same way" for
"same weights".

Three tiers, by purpose:

* `continuation/` - high constraint. Primary discriminator. Also the cheapest to compare because
  outputs are short and deterministic in structure.
* `structured/` - format-following (JSON, lists, arithmetic). Separates instruction tuning and
  adapters; catches LoRA that only alters formatting.
* `lexical/` - rare tokens, code, mixed scripts. Exposes quantization most directly, since quantized
  weights perturb low-frequency token choices first.

Usage:
    python prompts/build.py                # write prompts/battery.jsonl
    python prompts/build.py --tier continuation --out prompts/seed.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each entry: (id, tier, rationale, prompt). Rationale is written to the file so the reason for
# every spent query is auditable later - budget is irreversible.
BATTERY: list[tuple[str, str, str, str]] = [
    # ---------------- high-constraint continuations: the primary discriminator ----------------
    (
        "cont_01",
        "continuation",
        "Concrete narrative continuation. High constraint; persona has little to act on.",
        "Continue the following with no preamble and no explanation, writing 30 to 50 words of plain prose.\n\n"
        "The lighthouse keeper had not spoken to anyone in eleven days, and",
    ),
    (
        "cont_02",
        "continuation",
        "Enumerative continuation with a fixed subject; suppresses stylistic freedom.",
        "Continue this list, one item per line, no introduction:\n\n"
        "Three reasons a bridge might be closed for inspection:\n1.",
    ),
    (
        "cont_03",
        "continuation",
        "Procedural text, strongly constrained by real-world ordering.",
        "Complete the following instructions in under 40 words, no preamble:\n\n"
        "To replace a bicycle inner tube, first remove the wheel, then",
    ),
    (
        "cont_04",
        "continuation",
        "Dialogue continuation; tests voice while remaining tightly constrained by context.",
        'Continue this exchange with exactly two more lines, no narration:\n\n'
        '"Did you bring the map?"\n"I brought the map, but the road it shows no longer exists."\n',
    ),
    (
        "cont_05",
        "continuation",
        "Causal explanation; low stylistic freedom, high semantic constraint.",
        "Explain in one sentence of under 30 words why ice floats on water. No preamble.",
        ),
    (
        "cont_06",
        "continuation",
        "Numbered factual recall; the answer is nearly unique, so any token drift is a weight signal.",
        "List the first five prime numbers greater than 100, comma separated, nothing else.",
    ),
    (
        "cont_07",
        "continuation",
        "Definition under length pressure; constrained but not a lookup.",
        "In fewer than 25 words, define 'osmosis' for a secondary-school student. No preamble.",
    ),
    (
        "cont_08",
        "continuation",
        "Temporal ordering task; a single correct ordering, minimal stylistic freedom.",
        "Arrange these events chronologically and output only the letters, space separated: "
        "A) the fall of Constantinople B) the invention of the printing press C) the Magna Carta.",
    ),
    # ---------------- structured output: exposes adapter and instruction-tuning differences ------
    (
        "struct_01",
        "structured",
        "Strict JSON schema. Adapters often alter key naming or formatting, which is a model signal.",
        "Output only valid JSON with keys name, population, country for the capital of France. "
        "No markdown fences, no commentary.",
    ),
    (
        "struct_02",
        "structured",
        "Exact-count list. Violating the count is a strong signal of different weights.",
        "List exactly four two-syllable synonyms for 'quiet', one per line, no numbering, no commentary.",
    ),
    (
        "struct_03",
        "structured",
        "Deterministic arithmetic with a required format; catches tokenizer and precision drift.",
        "Compute 17 * 23 and 144 / 12. Output only the two results separated by a comma.",
    ),
    (
        "struct_04",
        "structured",
        "Table formatting. Formatting fidelity separates instruction-tuned variants.",
        "Produce a two-column markdown table with header 'Element | Symbol' and exactly three rows "
        "for the first three noble gases. No text before or after the table.",
    ),
    # ---------------- lexical / tokenization: where quantization shows first -------------------
    (
        "lex_01",
        "lexical",
        "Rare-token spelling. Quantized weights perturb low-frequency choices most.",
        "Write the word 'onomatopoeia' and then three other English words of Greek origin, "
        "comma separated, nothing else.",
    ),
    (
        "lex_02",
        "lexical",
        "Code output. Syntax and identifier choices are sensitive to weight precision.",
        "Write a Python one-liner that reverses a string s. Output only the code, no fences, no explanation.",
    ),
    (
        "lex_03",
        "lexical",
        "Non-ASCII script; exercises tokenizer and vocabulary behaviour.",
        "Translate 'the library closes at six' into French, German, and Japanese. "
        "One per line, label each with the language name and a colon.",
    ),
    (
        "lex_04",
        "lexical",
        "Punctuation-sensitive transformation.",
        "Rewrite this sentence in the past tense, keeping the same words where possible: "
        "'She runs to the station and catches the early train.'",
    ),
    (
        "lex_05",
        "lexical",
        "Character-level constraint. Token drift is very visible here.",
        "List five English words that contain the letter sequence 'ough', comma separated, nothing else.",
    ),
    (
        "lex_06",
        "lexical",
        "Fixed-length output in a rare format.",
        "Write a haiku about autumn. Exactly three lines, no title, no commentary.",
    ),
    # ---------------- probes for the opposite failure direction --------------------------------
    (
        "trap_01",
        "continuation",
        "A persona instruction normally changes tone. Under high constraint it should NOT change "
        "the continuation much - if it does, the pair will be wrongly split.",
        "Ignoring any stylistic preference you may have, continue with 30 to 50 words of plain prose, "
        "no preamble:\n\nThe river froze in November that year, and by January",
    ),
    (
        "trap_02",
        "structured",
        "Explicitly requests a neutral register, neutralising persona-driven style shifts.",
        "In a strictly neutral encyclopaedic register, write two sentences about the water cycle. "
        "No preamble, no first person.",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", default=None, help="only this tier (continuation/structured/lexical)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "prompts" / "battery.jsonl"))
    parser.add_argument("--format", choices=["jsonl", "txt", "csv"], default="jsonl")
    args = parser.parse_args()

    rows = [row for row in BATTERY if args.tier is None or row[1] == args.tier]
    if args.limit:
        rows = rows[: args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        with out.open("w", encoding="utf-8") as handle:
            for pid, tier, rationale, prompt in rows:
                handle.write(
                    json.dumps({"id": pid, "tier": tier, "rationale": rationale, "prompt": prompt})
                    + "\n"
                )
    elif args.format == "txt":
        with out.open("w", encoding="utf-8") as handle:
            for _pid, _tier, _rationale, prompt in rows:
                handle.write(prompt.replace("\n", " ") + "\n")
    else:
        import csv

        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "prompt"])
            for pid, _tier, _rationale, prompt in rows:
                writer.writerow([pid, prompt])

    by_tier: dict[str, int] = {}
    for _pid, tier, _rationale, _prompt in rows:
        by_tier[tier] = by_tier.get(tier, 0) + 1
    print(f"wrote {len(rows)} prompts to {out}")
    print(f"by tier: {by_tier}")
    print(f"one prompt to all 33 scored endpoints costs 33 responses; {len(rows)} prompts costs {len(rows) * 33}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
