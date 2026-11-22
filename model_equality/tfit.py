"""Fine temperature fit per endpoint: parabola through the per-token lp(T) around the grid maximum, with bootstrap se."""
import collections, json, random, sys, csv
TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def fit(rs):
    n = sum(x["n"] for x in rs)
    y = [sum(x["lp"][str(T)] for x in rs) / n for T in TEMPS]
    i = max(range(len(TEMPS)), key=lambda k: y[k])
    if i == 0 or i == len(TEMPS) - 1:
        return TEMPS[i]
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return TEMPS[i] + (0.1 * 0.5 * (a - c) / den if den else 0)


def tfit(path, eps=None):
    by = collections.defaultdict(list)
    for r in map(json.loads, open(path)):
        if (eps is None or r["endpoint"] in eps) and r["n"] > 20:
            by[r["endpoint"]].append(r)
    rng = random.Random(0)
    out = {}
    for e, rs in sorted(by.items()):
        t = fit(rs)
        bs = [fit([rs[rng.randrange(len(rs))] for _ in rs]) for _ in range(100)]
        out[e] = (t, (sum((b - t) ** 2 for b in bs) / len(bs)) ** .5)
    return out


if __name__ == "__main__":
    kinds = {r[0]: r[2] + " " + r[3] for r in csv.reader(open("practice_answers.csv")) if r and r[0].startswith("ref")}
    lab = {r[0]: r[2] for r in csv.reader(open(sys.argv[2]))} if len(sys.argv) > 2 else {}
    for e, (t, s) in tfit(sys.argv[1]).items():
        print(f"{e} {kinds.get(e, lab.get(e, '')):30s} T={t:.3f} +- {s:.3f}")
