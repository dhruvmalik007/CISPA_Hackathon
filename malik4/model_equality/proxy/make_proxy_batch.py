#!/usr/bin/env python3
"""Generate the Doubleword batch JSONL for the proxy zoo.

Every `custom_id` encodes the ground-truth tuple, so the labels survive the round trip through the
batch API without a side table:

    ep|family|model|disguise|prompt|sample

The `custom_id` cap is 64 characters (docs, "Prepare your batch file"), so family and model are short
codes defined in `zoo.json`, and prompt ids are abbreviated. The script refuses to write a file
containing an over-long id rather than letting the API reject it after upload.

Disguises are split into two groups with different meanings, and the distinction is the whole point:

  SAME MODEL  (positives)  clean, sys_*, temp_low/temp_high, no_think
             These are deployment-level differences the task says do NOT change identity.
  These are what the verifier must learn to ignore.

There is deliberately no "different quantization" disguise here: this catalog has no same-family
quantization pairs (verified — see RUNBOOK.md), so those hard negatives must come from local GPU rows.

Usage:
    python proxy/make_proxy_batch.py --zoo proxy/zoo.json --out proxy/pilot_batch.jsonl \
        --models 1 --disguises clean,sys_concise,temp_high --prompts-limit 6 --samples 2
    python proxy/make_proxy_batch.py --zoo proxy/zoo.json --out proxy/proxy_batch.jsonl --samples 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MAX_CUSTOM_ID = 64
MAX_TOKENS = 256  # the competition's cap, so proxy responses have the same shape as ep-* responses

# name -> (system prompt or None, temperature, top_p). All of these are identity-preserving.
DISGUISES: dict[str, tuple[str | None, float, float]] = {
    "clean": (None, 0.7, 1.0),
    "sys_concise": ("Be extremely concise. Answer directly with minimal words.", 0.7, 1.0),
    "sys_verbose": ("You are a thorough academic tutor. Explain your reasoning clearly.", 0.7, 1.0),
    "sys_formal": ("Respond in a strictly formal, impersonal tone.", 0.7, 1.0),
    "temp_low": (None, 0.2, 0.95),
    "temp_high": (None, 1.1, 0.85),
    "no_think": (None, 0.7, 1.0),  # also appends /no_think, a documented Qwen control
}

# Reasoning models spend the token budget on hidden reasoning BEFORE emitting content. Measured on the
# pilot: with max_tokens=256 the model used all 256 on `reasoning_tokens` and returned
# `content: None` on 9 of 36 requests. That is fatal for this task, whose whole signal is the visible
# text - an empty response carries no fingerprint, and the competition caps answers at 256 tokens.
#
# Two defences, both applied below:
#   1. disable reasoning where the provider allows it (`reasoning_effort: "none"`);
#   2. raise max_tokens so reasoning cannot starve the visible answer.
# The second alone is enough, but a model that reasons about the prompt tends to produce answers that
# describe the task instead of performing it, so both are used.
REASONING_OFF = {"reasoning_effort": "none"}
MIN_TOKENS_WITH_REASONING = 1024


def short(code: str, limit: int = 10) -> str:
    """Deterministic short code, so the custom_id stays under the 64-char cap."""
    cleaned = "".join(ch for ch in code if ch.isalnum())
    if len(cleaned) <= limit:
        return cleaned
    digest = hashlib.sha1(code.encode()).hexdigest()[:4]
    return f"{cleaned[:limit - 4]}{digest}"


def load_prompts(path: Path, limit: int | None) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("prompt")
            if not text:
                continue
            prompts.append((str(row.get("id") or f"p{len(prompts):03d}"), str(text)))
    if limit:
        prompts = prompts[:limit]
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zoo", default="proxy/zoo.json")
    parser.add_argument("--prompts", default="prompts/battery.jsonl")
    parser.add_argument("--out", default="proxy/proxy_batch.jsonl")
    parser.add_argument("--models", type=int, default=None, help="cap the number of checkpoints used")
    parser.add_argument("--disguises", default=None, help="comma list; default all")
    parser.add_argument("--prompts-limit", type=int, default=None)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--no-think-suffix", default=" /no_think")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="visible-answer budget; default 256 to match the competition cap, but it is "
                             "raised automatically when reasoning is left enabled")
    parser.add_argument("--keep-reasoning", action="store_true",
                        help="do NOT send reasoning_effort=none. Slower, costlier, and the pilot showed it "
                             "can consume the whole budget and return empty content.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    zoo_path = Path(args.zoo)
    if not zoo_path.is_absolute():
        zoo_path = ROOT / zoo_path
    prompts_path = Path(args.prompts)
    if not prompts_path.is_absolute():
        prompts_path = ROOT / prompts_path
    for path in (zoo_path, prompts_path):
        if not path.is_file():
            print(f"missing input: {path}", file=sys.stderr)
            return 2

    zoo = json.loads(zoo_path.read_text(encoding="utf-8"))
    models = zoo["models"]
    if args.models:
        models = models[: args.models]

    wanted = [d.strip() for d in args.disguises.split(",")] if args.disguises else list(DISGUISES)
    unknown = [d for d in wanted if d not in DISGUISES]
    if unknown:
        print(f"unknown disguises: {unknown}; available: {sorted(DISGUISES)}", file=sys.stderr)
        return 2

    prompts = load_prompts(prompts_path, args.prompts_limit)
    if not prompts:
        print(f"no prompts in {prompts_path}", file=sys.stderr)
        return 2

    disable_reasoning = not args.keep_reasoning
    if args.max_tokens:
        output_tokens = args.max_tokens
    elif disable_reasoning:
        output_tokens = MAX_TOKENS          # matches the competition's 256-token answer cap
    else:
        output_tokens = MIN_TOKENS_WITH_REASONING

    # Short, stable codes for each model so custom_id stays short.
    codes: dict[str, tuple[str, str]] = {}
    for model in models:
        codes[model["alias"]] = (short(model["family"], 6).upper(), short(model["alias"].split("/")[-1], 10))

    requests: list[dict] = []
    ids: set[str] = set()
    too_long: list[str] = []
    for model in models:
        fam_code, model_code = codes[model["alias"]]
        for disguise in wanted:
            system_prompt, temperature, top_p = DISGUISES[disguise]
            for prompt_id, prompt in prompts:
                user_text = prompt
                if disguise == "no_think":
                    user_text = prompt + args.no_think_suffix
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_text})
                for sample in range(args.samples):
                    custom_id = f"ep|{fam_code}|{model_code}|{disguise}|{short(prompt_id, 8)}|{sample}"
                    if len(custom_id) > MAX_CUSTOM_ID:
                        too_long.append(custom_id)
                        continue
                    if custom_id in ids:
                        continue  # a duplicate custom_id would collapse two rows in the results
                    ids.add(custom_id)
                    body = {
                        "model": model["alias"],
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": output_tokens,
                    }
                    if disable_reasoning:
                        body.update(REASONING_OFF)
                    requests.append(
                        {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": body,
                        }
                    )

    if too_long:
        print(f"refusing: {len(too_long)} custom_id(s) exceed {MAX_CUSTOM_ID} chars, e.g. {too_long[0]}", file=sys.stderr)
        return 2

    expected = len(models) * len(wanted) * len(prompts) * args.samples
    print(f"models:    {len(models)}")
    print(f"disguises: {len(wanted)}  {wanted}")
    print(f"prompts:   {len(prompts)}")
    print(f"samples:   {args.samples}")
    print(f"reasoning: {'DISABLED (reasoning_effort=none)' if disable_reasoning else 'enabled'}"
          f"  max_tokens: {output_tokens}")
    print(f"requests:  {len(requests)}  (expected {expected})")
    if len(requests) != expected:
        print(f"WARNING: {expected - len(requests)} row(s) were dropped as duplicates", file=sys.stderr)

    # Positive cells: each (model, disguise) should be a rewording of the same underlying answer.
    print(f"\ncells (model x disguise): {len(models) * len(wanted)}, each with {len(prompts)} prompts")
    print("all disguises are identity-preserving, so same (model, prompt) across disguises = POSITIVE pair")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if args.dry_run:
        print(f"\nDRY RUN - would write {len(requests)} requests to {out}")
        print("first request:", json.dumps(requests[0])[:300] if requests else "(none)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request) + "\n")
    size_mb = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({len(requests)} requests, {size_mb:.2f} MB)")
    if size_mb > 200:
        print(f"WARNING: {size_mb:.0f} MB exceeds the 200 MB batch limit; use dw files split", file=sys.stderr)
    print("\nnext:  dw files validate {} && dw files stats {}".format(out, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
