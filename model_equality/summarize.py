"""Per-endpoint summary of v2_*.jsonl: fitted T, lp@T per token, top1, quantization llr (x1000/token, +- bootstrap se)."""
import collections
import csv
import json
import random
import sys

TEMPS = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2"]


def boot(rs, f, rng, k=200):
    return (sum((f([rs[rng.randrange(len(rs))] for _ in rs]) - f(rs)) ** 2 for _ in range(k)) / k) ** .5


def summarize(path, eps=None):
    by = collections.defaultdict(list)
    for r in map(json.loads, open(path)):
        if eps is None or r["endpoint"] in eps:
            by[r["endpoint"]].append(r)
    res, rng = {}, random.Random(0)
    for e in sorted(by):
        rs = by[e]
        n = sum(x["n"] for x in rs)
        lp = {T: sum(x["lp"][T] for x in rs) / n for T in TEMPS}
        bT = max(TEMPS, key=lambda T: lp[T])
        d = {"n_ans": len(rs), "T": float(bT), "lp": lp[bT], "lp07": lp["0.7"], "lp10": lp["1.0"],
             "top1": sum(x["top1"] for x in rs) / n,
             "lp_se": boot(rs, lambda s: sum(x["lp"][bT] for x in s) / sum(x["n"] for x in s), rng)}
        for k in [k for k in rs[0] if k.startswith("llr")]:
            f = lambda s, k=k: sum(x[k][bT] for x in s) / sum(x["n"] for x in s) * 1000
            d[k], d[k + "_se"] = f(rs), boot(rs, f, rng)
        res[e] = d
    return res


def show(res, kinds={}):
    for e, d in res.items():
        q = "  ".join(f"{k[3:] or '_fp8'}={d[k]:+6.2f}+-{d[k + '_se']:.2f}" for k in d if k.startswith("llr") and not k.endswith("_se"))
        print(f"{e} {kinds.get(e, ''):30s} n={d['n_ans']:3d} T={d['T']} lp={d['lp']:.3f}+-{d['lp_se']:.3f} "
              f"lp1.0={d['lp10']:.3f} top1={d['top1']:.3f} {q}")


if __name__ == "__main__":
    kinds = {}
    if len(sys.argv) > 2:
        kinds = {r[0]: r[2] + " " + r[3] for r in csv.reader(open(sys.argv[2])) if r and r[0].startswith("ref")}
    show(summarize(sys.argv[1]), kinds)
