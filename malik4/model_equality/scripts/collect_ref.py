#!/usr/bin/env python3
"""High-power collection on the FREE reference endpoints.

The reference endpoints have a 1,000,000 response limit, so the sample-size constraint that makes
the scored set so hard does not apply here. That means the honest question can be answered properly:
"with many prompts AND many repeats per prompt, does ANY statistic separate same-model from
different-model deployments?"

Every earlier test used falcon2's limited collections (10 and 40 prompts, almost no repeats). The
paper's method needs repeats per prompt to estimate a distribution rather than compare two points,
and it has never had them.

Prompts are drawn from a battery designed to stress the four deployment differences the task
describes, with a short answer length so many can be collected quickly:

  * structured/constrained prompts -> same model should reproduce the same *content*
  * open continuation           -> exposes sampling variance
  * identity/self-report         -> tests provider-injected system prompts
  * lexical/rare-token           -> quantization perturbs low-frequency choices first

Usage:
    python scripts/collect_ref.py --prompts 60 --repeats 8 --out data/ref_power.jsonl
    python scripts/collect_ref.py --prompts 60 --repeats 8 --dry-run
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
PARALLEL = 4  # the server's documented max
MAX_TOKENS = 96  # short answers: more prompts x repeats per unit time

# A battery spanning the behaviours the deployments differ on. Kept short so each answer is cheap.
PROMPTS: list[tuple[str, str]] = [
    ("num_list", "List the integers from 1 to 20 in order, comma separated, nothing else."),
    ("alpha", "Write the alphabet in uppercase, comma separated, nothing else."),
    ("months", "List the months of the year in order, comma separated, nothing else."),
    ("planets", "List the planets in order from the sun, comma separated, nothing else."),
    ("vowels", "List the vowels in order, comma separated, nothing else."),
    ("digits", "Write the digits 0 through 9, comma separated, nothing else."),
    ("weekdays", "List the days of the week, comma separated, nothing else."),
    ("colors", "Name five colors, comma separated, nothing else."),
    ("animals", "Name five animals, comma separated, nothing else."),
    ("fruits", "Name five fruits, comma separated, nothing else."),
    ("math", "Compute 7+5, 12-4, 6*3, 20/4. Output four numbers comma separated, nothing else."),
    ("arith2", "Compute 17*3, 49/7, 8+9, 100-45. Output four numbers comma separated, nothing else."),
    ("json", 'Output only JSON: {"city":"Paris","country":"France","pop":2148000}'),
    ("json2", 'Output only JSON with keys a,b,c set to 1,2,3.'),
    ("cap_fr", "What is the capital of France? Answer with the city name only."),
    ("cap_jp", "What is the capital of Japan? Answer with the city name only."),
    ("cap_br", "What is the capital of Brazil? Answer with the city name only."),
    ("syn_quiet", "List three synonyms for quiet, comma separated, nothing else."),
    ("syn_happy", "List three synonyms for happy, comma separated, nothing else."),
    ("syn_fast", "List three synonyms for fast, comma separated, nothing else."),
    ("rev_word", "Reverse the letters of the word 'strawberry'. Output only the result."),
    ("rev_word2", "Reverse the letters of the word 'mountain'. Output only the result."),
    ("count_v", "Count the vowels in 'extraordinary'. Output only the number."),
    ("upper", "Convert 'the quick brown fox' to uppercase. Output only the result."),
    ("cont_city", "Continue in one short sentence: The city at night is"),
    ("cont_river", "Continue in one short sentence: The river in winter is"),
    ("cont_library", "Continue in one short sentence: The library at closing time is"),
    ("cont_train", "Continue in one short sentence: The train arriving late is"),
    ("haiku", "Write a haiku about rain. Three lines only."),
    ("identity", "In one sentence: what model are you and who trained you?"),
    ("selfname", "What is your name? One word if possible."),
    ("translate_fr", "Translate 'good morning' into French. Output only the translation."),
    ("translate_de", "Translate 'good morning' into German. Output only the translation."),
    ("spell", "Spell 'necessary' one letter at a time, hyphen separated."),
    ("rhyme", "Give one word that rhymes with 'orange'. If none, write NONE."),
    ("odd", "List the odd numbers from 1 to 15, comma separated, nothing else."),
    ("even", "List the even numbers from 2 to 16, comma separated, nothing else."),
    ("prime", "List the primes below 20, comma separated, nothing else."),
    ("squares", "List the squares of 1 through 10, comma separated, nothing else."),
    ("pig", "Translate 'hello world' into pig latin. Output only the result."),
    ("leet", "Write 'hello' in leetspeak. Output only the result."),
    ("bin", "Write the number 10 in binary. Output only the digits."),
    ("hex", "Write the number 255 in hexadecimal. Output only the digits."),
    ("roman", "Write 1987 in Roman numerals. Output only the result."),
    ("palindrome", "Is 'racecar' a palindrome? Answer yes or no only."),
    ("fact", "Give one fact about the sun. One short sentence."),
    ("opinion", "Give one opinion about cities. One short sentence."),
    ("advice", "Give one piece of advice about sleep. One short sentence."),
    ("emoji", "Describe a sunset in three words, comma separated."),
    ("verb", "List three verbs, comma separated, nothing else."),
    ("adj", "List three adjectives, comma separated, nothing else."),
    ("num_word", "Write the number 42 in words. Output only the words."),
    ("num_word2", "Write the number 1984 in words. Output only the words."),
    ("capital_q", "Which city is the capital of Italy? Answer with the city name only."),
    ("chem", "What is the chemical symbol for gold? Answer with the symbol only."),
    ("geo", "What is the largest ocean? Answer with the name only."),
    ("unit", "How many centimetres in a metre? Output only the number."),
    ("time", "How many minutes in two hours? Output only the number."),
    ("zero", "What is 0 divided by 5? Output only the number."),
    ("perm", "List all permutations of the letters a and b, comma separated."),
    ("nl", "Write the numbers 1 to 5 each on its own line. Nothing else."),
]


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
    parser.add_argument("--prompts", type=int, default=60, help="how many prompts from the battery")
    parser.add_argument("--repeats", type=int, default=8, help="responses per prompt per endpoint")
    parser.add_argument("--out", default="data/ref_power.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--api-key", default=API_KEY)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / args.out

    prompts = PROMPTS[: args.prompts]
    jobs = [
        (ep, pid, repeat, prompt)
        for pid, prompt in prompts
        for ep in REFERENCE
        for repeat in range(args.repeats)
    ]
    done = load_existing(out)
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]

    print(f"{len(prompts)} prompts x {len(REFERENCE)} endpoints x {args.repeats} repeats = {len(jobs)} answers")
    print(f"already collected: {len(done)}; to query: {len(todo)}")

    if args.dry_run:
        first = todo[0] if todo else None
        print(f"\nDRY RUN - nothing sent. First job would be: {first}")
        return 0
    if not todo:
        print("nothing to do")
        return 0

    session = requests.Session()
    session.headers["X-API-Key"] = args.api_key
    lock = threading.Lock()
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = out.open("a", encoding="utf-8")
    counter = [0]
    failures: list[str] = []
    exhausted: set[str] = set()

    def one(job: tuple[str, str, int, str]) -> None:
        endpoint, pid, repeat, prompt = job
        if endpoint in exhausted:
            return
        for attempt in range(4):
            try:
                reply = session.post(
                    f"{BASE}/chat/completions",
                    json={
                        "model": endpoint,
                        "max_tokens": MAX_TOKENS,
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
                    if counter[0] % 50 == 0 or counter[0] == len(todo):
                        print(f"  {counter[0]}/{len(todo)} collected")
                return
            if reply.status_code == 429:
                wait = reply.headers.get("Retry-After")
                time.sleep(int(wait) if wait else 5)
                continue
            failures.append(f"{endpoint} {pid} r{repeat}: HTTP {reply.status_code} {reply.text[:120]}")
            return

    try:
        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            list(pool.map(one, todo))
    finally:
        handle.close()

    print(f"\nwrote {counter[0]} answers to {out}")
    print(f"total rows now: {len(load_existing(out))}")
    if failures:
        print(f"{len(failures)} failures, first 5:", *failures[:5], sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
