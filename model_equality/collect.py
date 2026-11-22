"""Parallel, resumable, budget-safe collector for the model-equality challenge.

Input: a JSONL job file, one line per prompt:
    {"id": "det-1", "prompt": "...", "repeats": 2, "endpoints": ["ep-01", "ep-02"]}   # "endpoints" optional
    or with a multi-turn conversation instead of "prompt" (roles user/assistant, last one user):
    {"id": "fs-1", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}
Output: answers JSONL (one line per answer, with endpoint, id, repeat, prompt, response, latency, full body).

Safety:
  * Scored endpoints (ep-..) are only queried with --allow-scored.
  * --cap N: never let any endpoint's used budget go above N (checked against the server's /models before starting
    and updated after every answer). Default cap 0 for scored endpoints, so you must choose it explicitly.
  * Resumable: (endpoint, id, repeat) already in the output are skipped.

Examples:
    python collect.py jobs_ref.jsonl --out ref_answers.jsonl --endpoints ref-01,ref-02          # practice, free
    python collect.py jobs_p1.jsonl --out p1_answers.jsonl --endpoints all-ep --allow-scored --cap 12
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

API_KEY = "376ae623f4dfdf4cacd92c9864795292"
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1"
PARALLEL = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--endpoints", default="all-ref",
                    help="comma list, or all-ref / all-ep (used when a job line has no 'endpoints')")
    ap.add_argument("--allow-scored", action="store_true")
    ap.add_argument("--cap", type=int, default=0, help="max used budget per scored endpoint after this run")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if API_KEY == "YOUR_API_KEY_HERE":
        sys.exit("Set API_KEY first.")

    s = requests.Session()
    s.headers["X-API-Key"] = API_KEY
    models = s.get(f"{BASE_URL}/models", timeout=60).json()["data"]
    used = {m["id"]: m["budget"]["used"] for m in models}
    limit = {m["id"]: m["budget"]["limit"] for m in models}

    if a.endpoints == "all-ref":
        default_eps = sorted(e for e in used if e.startswith("ref-"))
    elif a.endpoints == "all-ep":
        default_eps = sorted(e for e in used if e.startswith("ep-"))
    else:
        default_eps = [e.strip() for e in a.endpoints.split(",") if e.strip()]

    jobs_in = [json.loads(l) for l in open(a.jobs, encoding="utf-8") if l.strip()]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in map(json.loads, open(out, encoding="utf-8")) if r}

    def allowed_total(ep):
        if ep.startswith("ep-"):
            return min(a.cap, limit[ep]) if a.allow_scored else used[ep]   # no new scored answers without the flag
        return limit[ep]

    planned = dict(used)
    jobs, skipped = [], 0
    max_rep = max((j.get("repeats", 1) for j in jobs_in), default=1)
    for rep in range(max_rep):                      # round-robin: repeat-major, so a stop leaves even coverage
        for j in jobs_in:
            if rep >= j.get("repeats", 1):
                continue
            for ep in j.get("endpoints", default_eps):
                if ep not in used:
                    sys.exit(f"unknown endpoint {ep}")
                if (ep, j["id"], rep) in done:
                    continue
                if planned[ep] + 1 > allowed_total(ep):
                    skipped += 1
                    continue
                planned[ep] += 1
                jobs.append((ep, j["id"], rep, j.get("messages") or [{"role": "user", "content": j["prompt"]}]))
    scored_new = {e: planned[e] - used[e] for e in planned if e.startswith("ep-") and planned[e] > used[e]}
    print(f"{len(jobs)} answers to collect, {len(done)} already done, {skipped} skipped by cap/flag")
    if scored_new:
        print(f"SCORED budget to spend: {sum(scored_new.values())} answers over {len(scored_new)} endpoints "
              f"(max {max(scored_new.values())} each; cap {a.cap})")
    if a.dry_run or not jobs:
        return

    lock, counter, exhausted = threading.Lock(), [0], set()
    fh = out.open("a", encoding="utf-8")
    t_start = time.time()

    def one(job):
        ep, pid, rep, messages = job
        prompt = messages[-1]["content"]
        while ep not in exhausted:
            t0 = time.time()
            try:
                r = s.post(f"{BASE_URL}/chat/completions", timeout=300, json={
                    "model": ep, "max_tokens": a.max_tokens, "messages": messages})
            except requests.RequestException as exc:
                print(f"  {ep}: unreachable ({type(exc).__name__}), retry in 30 s")
                time.sleep(30)
                continue
            if r.status_code == 200:
                body = r.json()
                rec = {"endpoint": ep, "id": pid, "repeat": rep, "prompt": prompt, "n_messages": len(messages),
                       "response": body["choices"][0]["message"]["content"],
                       "finish_reason": body["choices"][0].get("finish_reason"),
                       "latency_s": round(time.time() - t0, 2), "t": round(t0, 1), "budget": body.get("budget"),
                       "body_extra": {k: v for k, v in body.items() if k not in ("choices", "budget")}}
                with lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    counter[0] += 1
                    n = counter[0]
                    if n % 8 == 0 or n == len(jobs):
                        el = time.time() - t_start
                        print(f"  {n}/{len(jobs)}  {el / 60:.1f} min elapsed, ~{el / n * (len(jobs) - n) / 60:.1f} min left")
                return
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(int(r.headers["Retry-After"]))
            elif r.status_code == 429:
                with lock:
                    exhausted.add(ep)
                print(f"  {ep}: budget exhausted")
            elif r.status_code in (502, 503, 504):
                time.sleep(15)
            else:
                print(f"  {ep} {pid}: HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr)
                return

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, jobs))
    finally:
        fh.close()
    print(f"done -> {out}")


if __name__ == "__main__":
    main()