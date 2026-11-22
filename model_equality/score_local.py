"""Score collected answers under a LOCAL copy of a model (and an fp8-rounded copy of it).

For every answer: rebuild the exact conversation (from the jobs file), apply the model's chat template, and compute the
log-probability of the answer's tokens. Per endpoint this gives:
  lp@T     mean log-prob per token at temperature T (log_softmax(logits / T)); the best T is the endpoint's fitted temperature
  top1     fraction of answer tokens that are the local model's argmax (temperature-free)
  llr_fp8  mean per-token [log p_bf16 - log p_fp8] at the fitted T. > 0: closer to the bf16 model; < 0: closer to fp8.
           For answers sampled from the bf16 model its expectation is KL(bf16 || fp8) > 0; from an fp8 model it is < 0.

    python score_local.py --answers ref_calib.jsonl --jobs jobs_ref_calib.jsonl \
        --model HuggingFaceTB/SmolLM2-360M-Instruct --fp8 tensor --practice practice_answers.csv
    python score_local.py --answers p1_answers.jsonl --jobs jobs_p1.jsonl --model Qwen/Qwen2.5-1.5B-Instruct \
        --endpoints ep-01,ep-02,... --fp8 tensor

Needs torch + transformers. Uses the GPU if there is one (bf16), else CPU (float32). Small models run fine on CPU.
Writes per-answer rows to scores_<model>_<answers>.jsonl and prints the per-endpoint summary.
"""
import argparse
import collections
import copy
import csv
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]


def fp8_round_(model, mode):
    """Round every Linear weight (not lm_head) to float8 e4m3 and back, with a per-tensor or per-output-channel scale."""
    n = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                w = mod.weight.data.float()
                s = (w.abs().amax() if mode == "tensor" else w.abs().amax(dim=1, keepdim=True)) / 448.0
                s = torch.clamp(s, min=1e-12)
                mod.weight.data.copy_(((w / s).to(torch.float8_e4m3fn).float() * s).to(mod.weight.dtype))
                n += 1
    return n


def conversation(rec, jobs):
    j = jobs.get(rec["id"])
    if j and "messages" in j:
        return j["messages"]
    if j and "prompt" in j:
        return [{"role": "user", "content": j["prompt"]}]
    return [{"role": "user", "content": rec["prompt"]}]


@torch.no_grad()
def token_logprobs(model, ids, n_prompt, temps):
    logits = model(ids.unsqueeze(0)).logits[0, n_prompt - 1:-1].float()      # predicts tokens n_prompt .. end
    tgt = ids[n_prompt:]
    out = {T: torch.log_softmax(logits / T, -1).gather(1, tgt[:, None])[:, 0] for T in temps}
    top1 = (logits.argmax(-1) == tgt).float()
    return out, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--jobs", required=True, help="the jobs file the answers came from (to rebuild multi-turn prompts)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoints", default="all", help="comma list, a prefix like ref- / ep-, or all")
    ap.add_argument("--fp8", choices=["none", "tensor", "channel"], default="tensor")
    ap.add_argument("--max-answers", type=int, default=0, help="per endpoint, 0 = all")
    ap.add_argument("--practice", help="practice_answers.csv, to label the summary with the true kinds")
    a = ap.parse_args()

    jobs = {j["id"]: j for j in map(json.loads, open(a.jobs, encoding="utf-8")) if j}
    recs = [json.loads(l) for l in open(a.answers, encoding="utf-8") if l.strip()]
    if a.endpoints != "all":
        keep = [e.strip() for e in a.endpoints.split(",")]
        recs = [r for r in recs if any(r["endpoint"] == k or (k.endswith("-") and r["endpoint"].startswith(k))
                                       for k in keep)]
    if a.max_answers:
        cnt = collections.Counter()
        recs = [r for r in recs if (cnt.update([r["endpoint"]]) or cnt[r["endpoint"]] <= a.max_answers)]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).to(dev).eval()
    qmodel = None
    if a.fp8 != "none":
        qmodel = copy.deepcopy(model)
        print(f"fp8 ({a.fp8} scale) applied to {fp8_round_(qmodel, a.fp8)} Linear layers")
    print(f"{len(recs)} answers, model {a.model} on {dev}")

    tag = f"{Path(a.model).name}_{Path(a.answers).stem}"
    rows = []
    with open(f"scores_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for k, r in enumerate(recs):
            resp = r.get("response") or ""
            if not resp.strip():
                continue
            prompt_text = tok.apply_chat_template(conversation(r, jobs), tokenize=False, add_generation_prompt=True)
            p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp, add_special_tokens=False)["input_ids"][:300]
            ids = torch.tensor(p_ids + r_ids, device=dev)
            lp, top1 = token_logprobs(model, ids, len(p_ids), TEMPS)
            row = {"endpoint": r["endpoint"], "id": r["id"], "n": len(r_ids), "top1": float(top1.sum()),
                   "lp": {str(T): float(v.sum()) for T, v in lp.items()}}
            if qmodel is not None:
                qlp, _ = token_logprobs(qmodel, ids, len(p_ids), TEMPS)
                row["llr"] = {str(T): float((lp[T] - qlp[T]).sum()) for T in TEMPS}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(recs)}")

    kinds = {}
    if a.practice:
        for row in csv.reader(open(a.practice, encoding="utf-8-sig")):
            if row and row[0].startswith("ref-"):
                kinds[row[0]] = f"{row[2]:3s} {row[3] if len(row) > 3 else ''}"
    by = collections.defaultdict(list)
    for row in rows:
        by[row["endpoint"]].append(row)
    rng = random.Random(0)
    head = "endpoint  kind                              tokens  bestT  lp@bestT  lp@0.7  lp@1.0   top1"
    if qmodel is not None:
        head += "   llr_fp8 (x1000/token) +- se"
    print("\n" + head)
    for e in sorted(by):
        rs = by[e]
        ntok = sum(x["n"] for x in rs)
        lpT = {T: sum(x["lp"][str(T)] for x in rs) / ntok for T in TEMPS}
        bestT = max(TEMPS, key=lambda T: lpT[T])
        line = (f"{e:9s} {kinds.get(e, ''):33s} {ntok:6d}  {bestT:4.1f}  {lpT[bestT]:8.3f}  {lpT[0.7]:6.3f}  "
                f"{lpT[1.0]:6.3f}  {sum(x['top1'] for x in rs) / ntok:.3f}")
        if qmodel is not None:
            key = str(bestT)
            est = sum(x["llr"][key] for x in rs) / ntok
            boots = []
            for _ in range(300):
                smp = [rs[rng.randrange(len(rs))] for _ in rs]
                boots.append(sum(x["llr"][key] for x in smp) / max(1, sum(x["n"] for x in smp)))
            se = (sum((b - est) ** 2 for b in boots) / len(boots)) ** 0.5
            line += f"   {est * 1000:+8.3f} +- {se * 1000:.3f}"
        print(line)
    print(f"\nPer-answer scores: scores_{tag}.jsonl")


if __name__ == "__main__":
    main()