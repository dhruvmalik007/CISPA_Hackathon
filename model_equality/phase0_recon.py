"""Phase 0 recon on the PRACTICE endpoints only (ref-01 ... ref-09: no query limit, never scored).

Answers the questions that decide the plan, without spending any scored budget:
  1. What does /models list (are the ref endpoints there, what budget fields)?
  2. What does a raw response contain beyond the text (usage / prompt_tokens, logprobs, headers, finish_reason)?
  3. Is the default decoding greedy? (same prompt 5x on several ref endpoints)
  4. Do extra request fields work (logprobs, n, multi-turn with an assistant turn, system role)?
  5. Where is the 1,024-token input limit, and does it differ between ref-01 (plain) and ref-03 (system prompt)?
     The error message may also state the token count.
  6. Same prompt on all 9 refs: latency, length, usage side by side.

Everything goes to phase0_log.jsonl (one JSON line per request, with status, headers, body and latency).
A summary is printed at the end. Run: python phase0_recon.py     (pip install requests)
"""
import json
import sys
import time
from pathlib import Path

import requests

API_KEY = "376ae623f4dfdf4cacd92c9864795292"
SERVER_URL = "http://34.147.229.55"
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL.rstrip('/')}/{TASK_ID}/v1"
LOG = Path("phase0_log.jsonl")
REFS = [f"ref-{i:02d}" for i in range(1, 10)]
PROMPT = "Write a short story (about 150 words) about a lighthouse keeper who finds a message in a bottle."

session = requests.Session()
session.headers["X-API-Key"] = API_KEY
log = LOG.open("a", encoding="utf-8")


def post(body, tag):
    """One request with retries. Never touches scored endpoints."""
    if not str(body.get("model", "")).startswith("ref-"):
        raise SystemExit(f"refusing to query {body.get('model')}: this script only queries ref-* endpoints")
    while True:
        t0 = time.time()
        try:
            r = session.post(f"{BASE_URL}/chat/completions", json=body, timeout=180)
        except requests.RequestException as exc:
            print(f"  unreachable ({type(exc).__name__}), retry in 30 s")
            time.sleep(30)
            continue
        dt = time.time() - t0
        if r.status_code == 429 and "Retry-After" in r.headers:
            time.sleep(int(r.headers["Retry-After"]))
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(15)
            continue
        try:
            data = r.json()
        except ValueError:
            data = {"raw_text": r.text[:2000]}
        rec = {"tag": tag, "request": body, "status": r.status_code, "latency_s": round(dt, 3),
               "headers": dict(r.headers), "body": data}
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        return rec


def text(rec):
    try:
        return rec["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def detail(rec):
    b = rec["body"]
    return b.get("detail", b) if isinstance(b, dict) else b


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def msg(content, ep, max_tokens=256, **extra):
    return {"model": ep, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}], **extra}


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        raise SystemExit("Set API_KEY first.")
    summary = {}

    # 1. /models
    r = session.get(f"{BASE_URL}/models", timeout=60)
    models = r.json()
    log.write(json.dumps({"tag": "models", "status": r.status_code, "headers": dict(r.headers), "body": models}) + "\n")
    ids = [m.get("id") for m in models.get("data", [])]
    print(f"[1] /models lists {len(ids)} ids: {ids[:5]} ... {ids[-5:]}")
    print("    first entry, full:", json.dumps(models.get("data", [{}])[0]))
    print("    top-level keys:", list(models.keys()))

    # 2. Raw response anatomy
    rec = post(msg(PROMPT, "ref-01"), "anatomy")
    print(f"\n[2] raw response from ref-01 (HTTP {rec['status']}, {rec['latency_s']} s)")
    body = dict(rec["body"]) if isinstance(rec["body"], dict) else {}
    if "choices" in body:
        ch = dict(body["choices"][0])
        ch["message"] = {k: (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
                         for k, v in ch.get("message", {}).items()}
        body["choices"] = [ch]
    print("    body:", json.dumps(body, ensure_ascii=False)[:1500])
    print("    headers:", {k: v for k, v in rec["headers"].items()
                           if k.lower() not in ("date", "connection", "keep-alive")})
    summary["usage_field"] = "usage" in rec["body"] if isinstance(rec["body"], dict) else False

    # 3. Determinism: same prompt 5x
    print("\n[3] determinism: same prompt 5x")
    for ep in ["ref-01", "ref-03", "ref-06", "ref-08", "ref-02"]:
        outs = [text(post(msg(PROMPT, ep), f"determinism:{ep}")) for _ in range(5)]
        outs = [o for o in outs if o is not None]
        uniq = len(set(outs))
        pref = [common_prefix(outs[0], o) for o in outs[1:]]
        print(f"    {ep}: {uniq} distinct of {len(outs)}; shared-prefix chars with 1st: {pref}")
        summary[f"distinct_{ep}"] = uniq
    print("    -> if ref-01/03/06/02 give 1 distinct and ref-08 gives 5: the default is greedy and only the T=1 disguise samples")

    # 4. Extra request fields
    print("\n[4] extra request fields")
    tests = {
        "logprobs": msg("Say hello.", "ref-01", 16, logprobs=True, top_logprobs=5),
        "n=2": msg("Say hello.", "ref-01", 16, n=2),
        "multi-turn+assistant": {"model": "ref-01", "max_tokens": 16, "messages": [
            {"role": "user", "content": "Pick a number."},
            {"role": "assistant", "content": "7"},
            {"role": "user", "content": "Now pick another one."}]},
        "assistant-last(prefill)": {"model": "ref-01", "max_tokens": 16, "messages": [
            {"role": "user", "content": "Name a colour."},
            {"role": "assistant", "content": "The colour is"}]},
        "system-role": {"model": "ref-01", "max_tokens": 16, "messages": [
            {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Say hello."}]},
        "max_tokens=1": msg("Say hello.", "ref-01", 1),
    }
    limit_max_tokens = 16
    for name, body in tests.items():
        rec = post(body, f"field:{name}")
        if name == "max_tokens=1" and rec["status"] == 200:
            limit_max_tokens = 1
        b = rec["body"] if isinstance(rec["body"], dict) else {}
        extra = ""
        if name == "logprobs" and "choices" in b:
            extra = f"logprobs returned: {b['choices'][0].get('logprobs') is not None}"
        if name == "n=2" and "choices" in b:
            extra = f"choices returned: {len(b['choices'])}"
        print(f"    {name:24s} HTTP {rec['status']}  {extra}  {'' if rec['status'] == 200 else str(detail(rec))[:200]}"
              f"{'  text=' + repr(text(rec)[:60]) if text(rec) else ''}")

    # 5. Input-length limit (binary search on number of ' the' repeats)
    print(f"\n[5] input-length limit (binary search; max_tokens={limit_max_tokens})")
    for ep in ["ref-01", "ref-03", "ref-05"]:
        lo, hi, first_err = 0, 1400, None
        while hi - lo > 1:
            mid = (lo + hi) // 2
            rec = post(msg("Reply OK." + " the" * mid, ep, limit_max_tokens), f"limit:{ep}:{mid}")
            if rec["status"] == 200:
                lo = mid
            else:
                hi = mid
                first_err = first_err or detail(rec)
        print(f"    {ep}: longest accepted = {lo} repeats; first error: {str(first_err)[:200]}")
        summary[f"limit_{ep}"] = lo
    print("    -> a lower limit on ref-03/05 than ref-01 means the limit counts the hidden system prompt"
          " (free system-prompt detector); same limit everywhere means it counts only our text")

    # 6. Same prompt on all refs
    print("\n[6] same prompt on all 9 refs")
    for ep in REFS:
        rec = post(msg(PROMPT, ep), f"allrefs:{ep}")
        t = text(rec) or ""
        b = rec["body"] if isinstance(rec["body"], dict) else {}
        print(f"    {ep}: HTTP {rec['status']} {rec['latency_s']:6.2f}s  chars={len(t):4d}  usage={b.get('usage')}"
              f"  finish={b.get('choices', [{}])[0].get('finish_reason') if b.get('choices') else None}"
              f"  start={t[:70]!r}")

    print("\nSummary:", json.dumps(summary))
    print(f"Full log: {LOG.resolve()}  (send it back for analysis)")


if __name__ == "__main__":
    try:
        main()
    finally:
        log.close()