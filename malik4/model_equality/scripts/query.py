#!/usr/bin/env python3
"""Budget-safe query CLI for the model-equality challenge.

Modes:
    --mode budget      report remaining allowance per endpoint, sending nothing
    --mode reference  query ref-01..ref-09 (unlimited, never scored) - the development surface
    --mode scored     query ep-01..ep-33 (100 responses per endpoint, for the whole competition)

`--dry-run` reports the exact cost and sends nothing. Always run it first: the scored budget is
irreversible and 100 per endpoint is all a team gets.

The API key is read from HACKATHON_API_KEY, the project `.env`, or `--api-key`.

Examples:
    python scripts/query.py --mode budget
    python scripts/query.py --mode reference --prompts prompts/seed.txt --dry-run
    python scripts/query.py --mode reference --prompts prompts/seed.txt --repeats 2
    python scripts/query.py --mode scored --prompts prompts/final.txt --dry-run
    python scripts/query.py --mode scored --prompts prompts/final.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from met import client  # noqa: E402


def load_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit.strip()
    for name in ("HACKATHON_API_KEY", "MODEL_EQUALITY_API_KEY", "API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    # Fall back to the project .env files, which are not exported into the shell.
    for candidate in (ROOT.parent / ".env", ROOT.parent / "orchestrator-app" / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() in {"HACKATHON_API_KEY", "MODEL_EQUALITY_API_KEY"}:
                return value.strip().strip('"').strip("'")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["budget", "reference", "scored"], required=True)
    parser.add_argument("--prompts", default=None, help="prompt file for reference/scored modes")
    parser.add_argument("--answers", default=None, help="answers JSONL (default by mode)")
    parser.add_argument("--repeats", type=int, default=1, help="answers per prompt and endpoint")
    parser.add_argument("--max-per-endpoint", type=int, default=None, help="hard cap on queries per endpoint")
    parser.add_argument("--max-tokens", type=int, default=client.MAX_TOKENS)
    parser.add_argument("--parallel", type=int, default=client.MAX_PARALLEL)
    parser.add_argument("--dry-run", action="store_true", help="report cost, send nothing")
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    if not api_key:
        print(
            "No API key found. Set HACKATHON_API_KEY (or --api-key).\n"
            "Until then nothing can be queried, and the budget stays untouched.",
            file=sys.stderr,
        )
        return 2

    session = client.make_session(api_key)

    if args.mode == "budget":
        budgets = client.read_budget(session)
        used = sum(b.used for b in budgets.values())
        total = sum(b.limit for b in budgets.values())
        print(f"{'endpoint':<10} {'used':>6} {'limit':>6} {'remaining':>10}")
        for endpoint in sorted(budgets):
            budget = budgets[endpoint]
            print(f"{endpoint:<10} {budget.used:>6} {budget.limit:>6} {budget.remaining:>10}")
        print(f"\ntotal: {used}/{total} used across {len(budgets)} endpoints")
        return 0

    if not args.prompts:
        print("--prompts is required for reference/scored modes", file=sys.stderr)
        return 2
    prompt_path = Path(args.prompts)
    if not prompt_path.is_absolute():
        prompt_path = ROOT / prompt_path
    if not prompt_path.is_file():
        print(f"prompt file not found: {prompt_path}", file=sys.stderr)
        return 2

    prompts = client.load_prompts(prompt_path)
    if not prompts:
        print(f"no prompts in {prompt_path}", file=sys.stderr)
        return 2
    longest = max(len(p) for _, p in prompts)
    if longest > 4000:
        print(f"WARNING: longest prompt is {longest} chars; the server caps input at 1024 tokens", file=sys.stderr)

    if args.mode == "reference":
        endpoints = client.REFERENCE
        default_answers = ROOT / "data" / "reference_answers.jsonl"
    else:
        budgets_preview = client.read_budget(session)
        endpoints = [ep for ep in sorted(budgets_preview) if ep.startswith("ep-")]
        default_answers = ROOT / "data" / "answers.jsonl"
        if not args.dry_run:
            remaining = sum(budgets_preview[ep].remaining for ep in endpoints)
            print(f"scored budget remaining: {remaining} responses across {len(endpoints)} endpoints")
            if remaining < len(prompts) * len(endpoints):
                print(
                    f"NOTE: {len(prompts)} prompts x {len(endpoints)} endpoints = "
                    f"{len(prompts) * len(endpoints)} > {remaining} remaining; "
                    "the collection will be truncated round-robin",
                    file=sys.stderr,
                )

    answers_path = Path(args.answers) if args.answers else default_answers
    if not answers_path.is_absolute():
        answers_path = ROOT / answers_path
    store = client.AnswerStore(answers_path)

    summary = client.collect(
        api_key=api_key,
        prompts=prompts,
        store=store,
        endpoints=endpoints,
        repeats=args.repeats,
        max_per_endpoint=args.max_per_endpoint,
        max_tokens=args.max_tokens,
        parallel=args.parallel,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDRY RUN - nothing was sent.")
    print(f"answers file: {answers_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
