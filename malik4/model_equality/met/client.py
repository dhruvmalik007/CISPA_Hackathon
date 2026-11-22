"""Budget-aware client for the model-equality challenge.

The binding constraint is tiny: **100 responses per endpoint for the entire competition**
(33 endpoints, 4 concurrent requests, 256 max tokens). That is ~3,300 responses total, versus the
~10,000+ a naive design would want. Every query is therefore irreversible capital, and this module
is built around not wasting any:

* every request is preceded by a budget read, so it can never overspend;
* answers append to a JSONL file, so an interrupted run resumes instead of re-spending;
* `plan()` reports exactly what a run would cost *without* sending anything;
* the reference endpoints (`ref-01..ref-09`) are unlimited and free, so the method is developed and
  tuned there.

The pairing design matters more than the statistics. Because the same prompt is sent to every
endpoint, two responses to one prompt are directly comparable, which controls for prompt difficulty
entirely. With one sample per prompt, the effective sample size is the number of *prompts*, not the
number of responses.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import requests

SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1"

# Server limits, from task_template.py. Not tunable.
MAX_TOKENS = 256
MAX_PARALLEL = 4
MAX_INPUT_TOKENS = 1024
BUDGET_PER_ENDPOINT = 100

SCORED = [f"ep-{i:02d}" for i in range(1, 34)]
REFERENCE = [f"ref-{i:02d}" for i in range(1, 10)]


class BudgetError(RuntimeError):
    """Raised rather than overspending the per-endpoint allowance."""


@dataclass
class EndpointBudget:
    endpoint: str
    limit: int
    used: int
    reserved: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used - self.reserved)

    def as_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "limit": self.limit,
            "used": self.used,
            "remaining": self.remaining,
            "reserved": self.reserved,
        }


@dataclass
class AnswerStore:
    """Append-only answer log with an in-memory index of what has already been collected."""

    path: Path
    rows: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A truncated final line from a killed run must not poison the whole log.
                        continue

    @property
    def keys(self) -> set[tuple[str, str, int]]:
        return {(r["endpoint"], str(r["id"]), int(r.get("repeat", 0))) for r in self.rows}

    def append(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows.append(row)

    def count_by_endpoint(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["endpoint"]] = counts.get(row["endpoint"], 0) + 1
        return counts

    def unique_prompts(self) -> int:
        return len({str(r["id"]) for r in self.rows})


def make_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers["X-API-Key"] = api_key
    return session


def read_budget(session: requests.Session) -> dict[str, EndpointBudget]:
    """Server-reported remaining allowance per endpoint. Costs nothing."""
    reply = session.get(f"{BASE_URL}/models", timeout=60)
    if reply.status_code != 200:
        raise RuntimeError(f"budget read failed (HTTP {reply.status_code}): {reply.text[:300]}")
    out: dict[str, EndpointBudget] = {}
    for model in reply.json()["data"]:
        budget = model.get("budget") or {}
        out[model["id"]] = EndpointBudget(
            endpoint=model["id"],
            limit=int(budget.get("limit", BUDGET_PER_ENDPOINT)),
            used=int(budget.get("used", 0)),
        )
    return out


def plan(
    prompts: list[tuple[str, str]],
    endpoints: list[str],
    store: AnswerStore,
    repeats: int = 1,
    max_per_endpoint: int | None = None,
) -> dict:
    """What a run would cost, computed without sending a single request.

    Returns the exact job list plus, per endpoint, how many queries would be spent and how many are
    left afterwards. Callers must inspect `shortfall` before calling `collect`.
    """
    done = store.keys
    jobs: list[tuple[str, str, int, str]] = []
    per_endpoint: dict[str, int] = {ep: 0 for ep in endpoints}

    for repeat in range(repeats):
        for pid, prompt in prompts:
            for endpoint in endpoints:
                if (endpoint, pid, repeat) in done:
                    continue
                if max_per_endpoint is not None and per_endpoint[endpoint] >= max_per_endpoint:
                    continue
                per_endpoint[endpoint] += 1
                jobs.append((endpoint, pid, repeat, prompt))

    return {
        "jobs": jobs,
        "prompts": len(prompts),
        "endpoints": len(endpoints),
        "repeats": repeats,
        "per_endpoint": per_endpoint,
        "total_queries": len(jobs),
        "already_have": len(store.rows),
    }


def collect(
    api_key: str,
    prompts: list[tuple[str, str]],
    store: AnswerStore,
    endpoints: list[str],
    repeats: int = 1,
    max_per_endpoint: int | None = None,
    max_tokens: int = MAX_TOKENS,
    parallel: int = MAX_PARALLEL,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """Send prompts to endpoints, never exceeding the server's remaining allowance.

    `dry_run=True` performs the full budget check and returns the plan with zero requests sent,
    which is the only safe way to preview an experiment. The server is the authority on remaining
    budget: the plan is intersected with it before anything is sent.
    """
    session = make_session(api_key)
    budgets = read_budget(session)

    unknown = [ep for ep in endpoints if ep not in budgets]
    if unknown:
        raise ValueError(f"unknown endpoints {unknown}; server lists {sorted(budgets)}")

    work = plan(prompts, endpoints, store, repeats, max_per_endpoint)

    # Intersect the plan with the server's own accounting, then trim round-robin so an early stop
    # still leaves every endpoint equally covered.
    remaining = {ep: budgets[ep].remaining for ep in endpoints}
    planned: dict[str, int] = {ep: 0 for ep in endpoints}
    admitted: list[tuple[str, str, int, str]] = []
    dropped = 0
    for job in work["jobs"]:
        endpoint = job[0]
        if planned[endpoint] < remaining[endpoint]:
            planned[endpoint] += 1
            admitted.append(job)
        else:
            dropped += 1

    summary = {
        "would_send": len(admitted),
        "dropped_over_budget": dropped,
        "already_have": work["already_have"],
        "remaining_before": {ep: remaining[ep] for ep in endpoints},
        "remaining_after": {ep: remaining[ep] - planned[ep] for ep in endpoints},
        "dry_run": dry_run,
    }
    if verbose:
        print(
            f"prompts={work['prompts']} endpoints={work['endpoints']} repeats={repeats}\n"
            f"  already collected: {work['already_have']}\n"
            f"  would send: {len(admitted)}   dropped (over budget): {dropped}\n"
            f"  remaining after: {min(summary['remaining_after'].values())}"
            f"..{max(summary['remaining_after'].values())} per endpoint"
        )
    if dry_run or not admitted:
        summary["collected"] = 0
        return summary

    lock = threading.Lock()
    exhausted: set[str] = set()
    spent = [0]
    errors: list[str] = []

    def one(job: tuple[str, str, int, str]) -> None:
        endpoint, pid, repeat, prompt = job
        attempt = 0
        while endpoint not in exhausted:
            attempt += 1
            if attempt > 6:
                with lock:
                    errors.append(f"{endpoint} id {pid}: gave up after {attempt - 1} attempts")
                return
            try:
                reply = session.post(
                    f"{BASE_URL}/chat/completions",
                    timeout=180,
                    json={
                        "model": endpoint,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
            except requests.RequestException as exc:
                if verbose:
                    print(f"  network ({type(exc).__name__}); retrying in 20s")
                time.sleep(20)
                continue

            if reply.status_code == 200:
                body = reply.json()
                row = {
                    "endpoint": endpoint,
                    "id": pid,
                    "repeat": repeat,
                    "prompt": prompt,
                    "response": body["choices"][0]["message"]["content"],
                }
                with lock:
                    store.append(row)
                    spent[0] += 1
                    if verbose and (spent[0] % 25 == 0 or spent[0] == len(admitted)):
                        budget = body.get("budget", {})
                        print(
                            f"  {spent[0]}/{len(admitted)} "
                            f"({endpoint}: {budget.get('used', '?')}/{budget.get('limit', '?')})"
                        )
                return

            detail = ""
            if reply.headers.get("content-type", "").startswith("application/json"):
                detail = str(reply.json().get("detail", ""))[:200]
            else:
                detail = reply.text[:200]

            if reply.status_code == 429 and "Retry-After" in reply.headers:
                # Busy: a slot will free up. This is NOT budget exhaustion.
                time.sleep(int(reply.headers["Retry-After"]) or 5)
            elif reply.status_code == 429:
                # No Retry-After: this endpoint's allowance is gone.
                with lock:
                    if endpoint not in exhausted:
                        exhausted.add(endpoint)
                return
            elif reply.status_code in (502, 503, 504):
                time.sleep(15)
            else:
                # Invalid requests are rejected and not counted, so this costs no budget.
                with lock:
                    errors.append(f"{endpoint} id {pid}: HTTP {reply.status_code}: {detail}")
                return

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(parallel, MAX_PARALLEL))) as pool:
        list(pool.map(one, admitted))

    summary["collected"] = spent[0]
    summary["errors"] = errors[:20]
    summary["error_count"] = len(errors)
    summary["exhausted_endpoints"] = sorted(exhausted)
    summary["seconds"] = round(time.time() - started, 1)
    if verbose:
        print(f"collected {spent[0]} answers in {summary['seconds']}s; errors {len(errors)}")
        if exhausted:
            print(f"  budget exhausted on: {', '.join(sorted(exhausted))}")
    return summary


def load_prompts(path: Path) -> list[tuple[str, str]]:
    """One prompt per line in a .txt, or id,prompt in .csv / .jsonl."""
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [(row["id"], row["prompt"]) for row in csv.DictReader(handle)]
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [
                (str(row["id"]), row["prompt"])
                for row in (json.loads(line) for line in handle if line.strip())
            ]
    with path.open(encoding="utf-8") as handle:
        return [(str(i), line.rstrip("\n")) for i, line in enumerate(handle) if line.strip()]


def write_submission(rows: Iterable[tuple[str, str, str]], path: Path) -> Path:
    """CSV with endpoint,family,model. Endpoints sharing a model label must share a family."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)
    return path


def responses_by_prompt(store: AnswerStore) -> dict[str, dict[str, str]]:
    """prompt id -> {endpoint: response}. The paired layout the statistics rely on."""
    out: dict[str, dict[str, str]] = {}
    for row in store.rows:
        out.setdefault(str(row["id"]), {})[row["endpoint"]] = row["response"]
    return out
