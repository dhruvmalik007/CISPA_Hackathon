"""Paired per-prompt comparison: for each endpoint, mean over prompts of [lp/token(ep, prompt) - median over family
endpoints of lp/token(., prompt)] at a fixed temperature, with a standard error from the paired differences.
Also the same for the quantization llr's."""
import collections
import json
import statistics as st
import sys


def paired(path, eps=None, T="0.7", key="lp"):
    v = collections.defaultdict(dict)
    for r in map(json.loads, open(path)):
        if (eps is None or r["endpoint"] in eps) and r["n"] > 20:
            val = (r["rank_ge"][T] if "rank_ge" in r else None) if key == "rank" else (r[key][T] if key in r else None)
            if val is not None:
                v[r["endpoint"]][r["id"]] = val / r["n"]
    E = sorted(v)
    ids = {i for e in E for i in v[e]}
    ref = {i: st.median([v[e][i] for e in E if i in v[e]]) for i in ids}
    out = {}
    for e in E:
        d = [v[e][i] - ref[i] for i in v[e]]
        out[e] = (st.mean(d), st.stdev(d) / len(d) ** .5 if len(d) > 1 else 9, len(d))
    return out


if __name__ == "__main__":
    import csv
    kinds = {r[0]: r[2] + " " + r[3] for r in csv.reader(open("practice_answers.csv")) if r and r[0].startswith("ref")}
    T = sys.argv[2] if len(sys.argv) > 2 else "0.7"
    for key in sys.argv[3].split(",") if len(sys.argv) > 3 else ["lp"]:
        print(f"--- {key} @T={T}")
        for e, (m, s, n) in paired(sys.argv[1], T=T, key=key).items():
            print(f"{e} {kinds.get(e, ''):30s} {m * (1000 if key != 'lp' else 1):+8.3f} +- {s * (1000 if key != 'lp' else 1):.3f}  z={m / s:+.1f} n={n}")
