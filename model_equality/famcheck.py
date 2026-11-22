import collections, glob, json, sys
fam = {}
for l in open("sub01_families_only.csv").readlines()[1:]:
    e, f, _ = l.strip().split(",")
    fam[e] = f
tab = collections.defaultdict(dict)
for p in sorted(glob.glob(f"v2_*_{sys.argv[1]}.jsonl")):
    m = p.split("_")[1]
    by = collections.defaultdict(list)
    for r in map(json.loads, open(p)):
        by[r["endpoint"]].append(r)
    for e, rs in by.items():
        n = sum(x["n"] for x in rs)
        tab[e][m] = max(sum(x["lp"][T] for x in rs) / n for T in rs[0]["lp"])
models = sorted({m for d in tab.values() for m in d})
print("ep    fam " + " ".join(f"{m[:18]:>18s}" for m in models))
for e in sorted(tab, key=lambda e: (fam.get(e, "?"), e)):
    print(f"{e} {fam.get(e,'?')}   " + " ".join(f"{tab[e].get(m, float('nan')):18.3f}" for m in models))
