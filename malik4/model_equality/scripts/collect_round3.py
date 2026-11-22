#!/usr/bin/env python3
"""Collect a selected prompt battery on reference or scored endpoints.

The reference endpoints are free and have a 1,000,000 response limit. Scored endpoints are capped at
100 responses each; only run --mode scored after the selected prompts clear the reference gate.

Usage:
    python scripts/collect_round3.py --mode reference --prompts-file prompts/round3_ft01.jsonl \
        --repeats 8 --out data/ref_round3_ft01.jsonl
    python scripts/collect_round3.py --mode scored --prompts-file prompts/round3_ft01.jsonl \
        --repeats 8 --out data/ep_round3_ft01.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SERVER = "http://34.147.229.55"
TASK = "33-model-equality"
BASE = f"{SERVER}/{TASK}/v1"
API_KEY = "376ae623f4dfdf4cacd92c9864795292"
REFERENCE = [f"ref-{i:02d}" for i in range(1, 10)]
SCORED = [f"ep-{i:02d}" for i in range(1, 34)]
PARALLEL = 4


def load_prompts(path: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        prompts.append((str(row["id"]), str(row["prompt"])))
    return prompts


def load_existing(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((str(row["endpoint"]), str(row["id"]), int(row["repeat"])))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["reference", "scored"], default="reference")
    parser.add_argument("--prompts-file", default="prompts/round3_ft01.jsonl")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", default="data/ref_round3_ft01.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=PARALLEL)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompt_path = Path(args.prompts_file)
    if not prompt_path.is_absolute():
        prompt_path = ROOT / prompt_path
    prompts = load_prompts(prompt_path)
    endpoints = REFERENCE if args.mode == "reference" else SCORED
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out

    jobs = [
        (endpoint, pid, repeat, prompt)
        for pid, prompt in prompts
        for endpoint in endpoints
        for repeat in range(args.repeats)
    ]
    done = load_existing(out)
    todo = [job for job in jobs if (job[0], job[1], job[2]) not in done]
    print(
        f"{len(prompts)} prompts x {len(endpoints)} endpoints x {args.repeats} repeats "
        f"= {len(jobs)} answers",
        flush=True,
    )
    print(f"already collected: {len(done)}; to query: {len(todo)}", flush=True)
    if args.dry_run or not todo:
        print("DRY RUN - nothing sent" if args.dry_run else "nothing to do", flush=True)
        return 0

    session = requests.Session()
    session.headers["X-API-Key"] = args.api_key
    lock = threading.Lock()
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("a", encoding="utf-8")
    counter = [0]
    failures: list[str] = []

    def one(job: tuple[str, str, int, str]) -> None:
        endpoint, pid, repeat, prompt = job
        for _attempt in range(4):
            try:
                reply = session.post(
                    f"{BASE}/chat/completions",
                    json={
                        "model": endpoint,
                        "max_tokens": args.max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=180,
                )
            except requests.RequestException:
                time.sleep(3)
                continue
            if reply.status_code == 200:
                body = reply.json()
                row = {
                    "endpoint": endpoint,
                    "id": pid,
                    "repeat": repeat,
                    "prompt": prompt,
                    "response": body["choices"][0]["message"]["content"],
                    "finish_reason": body["choices"][0].get("finish_reason"),
                }
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    counter[0] += 1
                    if counter[0] % 10 == 0 or counter[0] == len(todo):
                        print(f"  {counter[0]}/{len(todo)} collected", flush=True)
                return
            if reply.status_code == 429:
                wait = reply.headers.get("Retry-After")
                time.sleep(int(wait) if wait else 5)
                continue
            failures.append(f"{endpoint} {pid} r{repeat}: HTTP {reply.status_code} {reply.text[:120]}")
            return

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(args.parallel, PARALLEL))) as pool:
            list(pool.map(one, todo))
    finally:
        handle.close()

    print(f"\nwrote {counter[0]} answers to {out}", flush=True)
    print(f"total rows now: {len(load_existing(out))}", flush=True)
    if failures:
        print(f"{len(failures)} failures, first 5:", *failures[:5], sep="\n  ", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
