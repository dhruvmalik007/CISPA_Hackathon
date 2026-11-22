"""Builds the submission from the local-likelihood scores (v2_*.jsonl from score_v2.py).

Per family (base model identified by best likelihood): every endpoint is compared to the family median prompt by prompt
(paired, so prompt difficulty cancels).
  * quantized: paired quantization llr (log p_base - log p_quant per token) clearly below the family -> "Q-<type>" model
    (same quantization type in one family = same model).
  * LoRA: paired log-likelihood clearly below the family (answers less likely under the base model), but not so far below
    that it looks like temperature-1 sampling -> own model (one adapter each).
  * temperature-like (fitted T >= 1, or very far below, or many rank>=100 tokens) -> base model "S".
  * otherwise S only if clearly above the family median (d > S_D): weight changes (weak LoRA, quantization) make the
    answers a little less likely under the base model, prompt-level disguises do not. The rest -> own model "V".
Thresholds calibrated on the reference endpoints (practice_answers.csv).

    python final.py ref        # practice check -> final_ref.csv (score with score_practice.py)
    python final.py ep         # scored endpoints -> submission_final.csv
"""
import csv
import json
import statistics as st
import sys

from paired import paired
from summarize import summarize

Q_Z, LORA_D, LORA_Z, TEMP_D, S_D = -3.0, -0.06, -2.5, -0.8, 0.035

FAMILIES = {
    "ref": [("P", "v2_SmolLM2-360M-Instruct_ref_p2.jsonl", [f"ref-{i:02d}" for i in range(1, 10)])],
    "ep": [("A", "v2_Qwen2.5-3B-Instruct_ep_p2.jsonl",
            "ep-01 ep-02 ep-04 ep-12 ep-14 ep-17 ep-18 ep-23 ep-24 ep-25 ep-33".split()),
           ("B", "v2_Qwen2.5-1.5B-Instruct_ep_p2.jsonl",
            "ep-03 ep-06 ep-07 ep-08 ep-20 ep-21 ep-22 ep-26 ep-28 ep-30 ep-32".split()),
           ("C", "v2_Llama-3.2-1B-Instruct_ep_p2.jsonl",
            "ep-05 ep-09 ep-10 ep-11 ep-13 ep-15 ep-16 ep-19 ep-27 ep-29 ep-31".split())],
}


def decide(path, eps, fam):
    summ = summarize(path, set(eps))
    lp = paired(path, set(eps), "0.7", "lp")
    qkeys = [k for k in json.loads(open(path).readline()) if k.startswith("llr")]
    q = {k: paired(path, set(eps), "0.7", k) for k in qkeys}
    rk = paired(path, set(eps), "100", "rank")
    labels, lora = {}, 0
    for e in eps:
        if e not in lp:
            labels[e] = (f"{fam}S", "no data")
            continue
        d, s, n = lp[e]
        zq = {k: q[k][e][0] / q[k][e][1] for k in qkeys if e in q[k]}
        kq = min(zq, key=zq.get) if zq else None
        zr = rk[e][0] / rk[e][1] if e in rk and rk[e][1] > 0 else 0.0
        temp_like = summ[e]["T"] >= 1.0 or d <= TEMP_D or zr > 4
        info = f"zr100={zr:+.1f} d={d:+.3f} z={d / s:+.1f} T={summ[e]['T']} n={n} " + " ".join(f"{k}:{v:+.1f}" for k, v in zq.items())
        if temp_like:
            labels[e] = (f"{fam}S", "TEMP  " + info)
        elif kq and zq[kq] < Q_Z:
            labels[e] = (f"{fam}Q{kq[3:] or '_fp8'}", "QUANT " + info)
        elif d < LORA_D and d / s < LORA_Z and not temp_like:
            lora += 1
            labels[e] = (f"{fam}L{lora}", "LORA  " + info)
        elif temp_like or d > S_D:
            labels[e] = (f"{fam}S", "same  " + info)
        else:
            lora += 1
            labels[e] = (f"{fam}V{lora}", "VAR   " + info)
    return labels


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ep"
    rows = []
    for fam, path, eps in FAMILIES[which]:
        if not __import__("os").path.exists(path):
            continue
        print(f"== family {fam}: {path}")
        for e, (m, info) in decide(path, eps, fam).items():
            print(f"  {e} {m:10s} {info}")
            rows.append((e, fam, m))
    out = "final_ref.csv" if which == "ref" else "submission_final.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(sorted(rows))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
