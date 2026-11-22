"""Submission v2: cluster endpoints within each family by their fitted sampling temperature (tfit.py).

A LoRA adapter (and 4-bit quantization) flattens the model's distribution relative to the base model, so the base model
needs a higher temperature to explain the answers; the shift grows with adapter strength. On the reference endpoints:
base disguises 0.709-0.723, fp8 0.717, LoRA 0.3 0.744, LoRA 0.7 0.791, LoRA 1.5 0.880, temperature-1 sampling 1.016.
  * T >= TEMP_T                  -> base model (temperature/decoding disguise)
  * T <= base band, quant llr z < Q_Z (fp8/int8 llr) -> 8-bit quantized model (one per family)
  * shifted T and 4-bit (nf4/int4) llr z < Q_Z -> 4-bit quantized model (4-bit also flattens the distribution)
  * T <= base band               -> base model
  * otherwise one model per T band (endpoints of one family in the same band = same adapter/strength)
    python final2.py ref | ep
"""
import csv
import json
import sys

from final import FAMILIES
from paired import paired
from tfit import tfit

TEMP_T, BASE_T, Q_Z = 0.95, 0.735, -3.0
BANDS = [(0.735, 0.775, "L1"), (0.775, 0.84, "L2"), (0.84, 0.95, "L3")]


def decide(path, eps, fam):
    tf = tfit(path, set(eps))
    qkeys = [k for k in json.loads(open(path).readline()) if k.startswith("llr")]
    q = {k: paired(path, set(eps), "0.7", k) for k in qkeys}
    lp = paired(path, set(eps), "0.7", "lp")
    out = {}
    for e in eps:
        t, s = tf[e]
        zq = {k: q[k][e][0] / q[k][e][1] for k in qkeys if e in q[k]}
        kq = min(zq, key=zq.get)
        info = f"T={t:.3f}+-{s:.3f} d={lp[e][0]:+.3f} " + " ".join(f"{k[3:] or '_fp8'}:{v:+.1f}" for k, v in zq.items())
        if t >= TEMP_T:
            out[e] = (f"{fam}S", "TEMP  " + info)
        elif t <= BASE_T and min(zq.get("llr", 0), zq.get("llr_int8", 0)) < Q_Z:
            out[e] = (f"{fam}Q8", "QUANT " + info)
        elif t > BASE_T and min(zq.get("llr_nf4", 0), zq.get("llr_int4", 0)) < Q_Z:
            out[e] = (f"{fam}Q4", "QUANT4 " + info)
        elif t <= BASE_T:
            out[e] = (f"{fam}S", "base  " + info)
        else:
            band = next(b for lo, hi, b in BANDS if lo < t <= hi)
            out[e] = (f"{fam}{band}", "SHIFT " + info)
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "ep"
    rows = []
    for fam, path, eps in FAMILIES[which]:
        print(f"== family {fam}")
        for e, (m, info) in sorted(decide(path, eps, fam).items(), key=lambda x: (x[1][0], x[0])):
            print(f"  {e} {m:9s} {info}")
            rows.append((e, fam, m))
    out = "final2_ref.csv" if which == "ref" else "submission_final2.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        w.writerows(sorted(rows))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
