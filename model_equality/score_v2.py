"""Per-answer local scoring with several stats per temperature (for the model-equality challenge).

For each answer and each T: sum log p_T(token), sum entropy H(p_T) at each position (expected -logp if the endpoint
sampled exactly from p_T), sum fp8 log-likelihood ratio, top1 hits. Writes v2_<model>_<answers>.jsonl.
    python score_v2.py --answers ep_p2.jsonl --jobs jobs_p2.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --endpoints ep-01,ep-02
"""
import argparse
import copy
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from score_local import conversation, fp8_round_

TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


NF4 = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
       -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
       0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


@torch.no_grad()
def quant_round_(model, kind):
    """Round-to-nearest symmetric quantization of Linear weights (not lm_head): int8 per output channel, int4 groups of 128."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            w = mod.weight.data.float()
            if kind == "nf4":
                lv = torch.tensor(NF4, device=w.device)
                o, i = w.shape
                g = w.reshape(-1, 64)
                sc = g.abs().amax(-1, keepdim=True).clamp(min=1e-12)
                idx = ((g / sc)[..., None] - lv).abs().argmin(-1)
                mod.weight.data.copy_((lv[idx] * sc).reshape(o, i))
            elif kind == "int8":
                sc = (w.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12)
                mod.weight.data.copy_((w / sc).round().clamp(-127, 127) * sc)
            else:
                o, i = w.shape
                g = w.reshape(o, i // 128, 128)
                sc = (g.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-12)
                mod.weight.data.copy_(((g / sc).round().clamp(-8, 7) * sc).reshape(o, i))


@torch.no_grad()
def stats(model, ids, n_prompt):
    logits = model(ids.unsqueeze(0)).logits[0, n_prompt - 1:-1].float()
    tgt = ids[n_prompt:]
    out = {}
    for T in TEMPS:
        ls = torch.log_softmax(logits / T, -1)
        out[T] = (ls.gather(1, tgt[:, None])[:, 0], -(ls.exp() * ls).sum(-1))
    rank = (logits > logits.gather(1, tgt[:, None])).sum(-1)
    return out, (logits.argmax(-1) == tgt).float().sum().item(), {k: int((rank >= k).sum()) for k in (2, 5, 20, 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoints", default="all")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--quants", default="", help="extra simulated quantizations: int8,int4")
    ap.add_argument("--max-per-ep", type=int, default=0)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    jobs = {j["id"]: j for j in map(json.loads, open(a.jobs, encoding="utf-8")) if j}
    recs = [json.loads(l) for l in open(a.answers, encoding="utf-8") if l.strip()]
    if a.endpoints != "all":
        keep = set(a.endpoints.split(","))
        recs = [r for r in recs if r["endpoint"] in keep]
    out_path = Path(f"v2{a.tag}_{Path(a.model).name}_{Path(a.answers).stem}.jsonl")
    done = set()
    if out_path.exists():
        done = {(r["endpoint"], r["id"], r.get("repeat", 0)) for r in map(json.loads, open(out_path)) if r}
    recs = [r for r in recs if (r["endpoint"], r["id"], r.get("repeat", 0)) not in done]
    if a.max_per_ep:
        cnt = {}
        recs = [r for r in recs if cnt.setdefault(r["endpoint"], [0]) and (cnt[r["endpoint"]].__setitem__(0, cnt[r["endpoint"]][0] + 1) or cnt[r["endpoint"]][0] <= a.max_per_ep)]
    print(f"{len(recs)} new answers to score with {a.model}", flush=True)
    if not recs:
        return
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16 if a.fp16 else torch.float32).to("cuda").eval()
    qmodel = None
    if a.fp8:
        qmodel = copy.deepcopy(model)
        fp8_round_(qmodel, "tensor")
    extra = {}
    for qn in [q for q in a.quants.split(",") if q]:
        extra[qn] = copy.deepcopy(model)
        quant_round_(extra[qn], qn)
    with out_path.open("a", encoding="utf-8") as fh:
        for r in recs:
            resp = r.get("response") or ""
            if not resp.strip():
                continue
            pt = tok.apply_chat_template(conversation(r, jobs), tokenize=False, add_generation_prompt=True)
            p_ids = tok(pt, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp, add_special_tokens=False)["input_ids"][:300]
            ids = torch.tensor(p_ids + r_ids, device="cuda")
            s, top1, ranks = stats(model, ids, len(p_ids))
            row = {"endpoint": r["endpoint"], "id": r["id"], "repeat": r.get("repeat", 0), "n": len(r_ids),
                   "top1": top1, "rank_ge": ranks, "lp": {str(T): float(v[0].sum()) for T, v in s.items()},
                   "H": {str(T): float(v[1].sum()) for T, v in s.items()}}
            if qmodel is not None:
                q, _, _ = stats(qmodel, ids, len(p_ids))
                row["llr"] = {str(T): float((s[T][0] - q[T][0]).sum()) for T in TEMPS}
            for qn, qm in extra.items():
                q, _, _ = stats(qm, ids, len(p_ids))
                row["llr_" + qn] = {str(T): float((s[T][0] - q[T][0]).sum()) for T in TEMPS}
            fh.write(json.dumps(row) + "\n")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
