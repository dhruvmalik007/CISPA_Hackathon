"""Analyse collected answers (from collect.py) for the model-equality challenge.

    python analyze.py p1_answers.jsonl                       # scored endpoints: report + distance matrices
    python analyze.py ref_calib.jsonl --practice practice_answers.csv   # practice: also measures how well each signal separates

Signals (per pair of endpoints):
  cat_mmd   delta-kernel MMD^2 on the probe answers (AgentProv-style): per question, within-a + within-b - 2*cross
            agreement, averaged over questions. 0 = same answer distribution. Needs >= 2 answers per question.
  cat_cross 1 - cross agreement (chance that a random answer of a equals a random answer of b), averaged over questions.
  text      1 - cosine similarity of word counts on the free-text prompts, averaged over prompts.
  style     difference in format behaviour (extra chatter after the answers, answers that break the format, length).
Output: report_<input>.txt and dist_<signal>_<input>.csv, plus nearest neighbours of every endpoint on screen.
Needs only numpy.
"""
import argparse
import collections
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

PROBES = {  # id: (number of real questions, the worked-example answers the model might echo)
    "probeA": (15, ["leek", "4", "no"]),
    "probeB": (12, ["note_open", "phone_call"]),
    "probeC": (12, ["dark", "tennis"]),
    "probeD": (12, ["8", "8"]),
}
NUMERIC = {("probeA", 1), ("probeA", 8)} | {("probeD", i) for i in range(1, 13)}
LINE = re.compile(r"^\s*#?\s*(\d{1,2})\s*[.):]\s*(.*\S)?\s*$")
WORD = re.compile(r"[a-zà-ÿ0-9_]+")


def norm(probe, q, text):
    t = (text or "").strip().lower()
    t = re.sub(r"^(the|a|an)\s+", "", t)
    if (probe, q) in NUMERIC:
        m = re.search(r"\d+(?:\.\d+)?", t)
        return m.group(0) if m else "<nonum>"
    t = re.split(r"[,(;]| - |\.\s", t)[0]
    t = t.strip(" .!?'\"*`")
    words = t.split()
    keep = 1 if probe == "probeC" else 2
    return " ".join(words[:keep]) if words else "<empty>"


def question_texts(prompt):
    """Question number -> phrase, from the last user message (used to strip echoed phrases like 'The weather today is')."""
    out = {}
    for ln in (prompt or "").splitlines():
        m = LINE.match(ln)
        if m and m.group(2):
            out[int(m.group(1))] = re.sub(r"^complete with one word:\s*", "", m.group(2).strip(), flags=re.I).lower()
    return out


def parse_probe(probe, response, prompt=""):
    """Returns ({question number: normalized answer}, style dict). Keeps the model's own numbering, because when a model
    echoes the worked example it usually continues with the real question numbers."""
    n_q, example = PROBES[probe]
    qtext = question_texts(prompt)
    lines = response.splitlines()
    numbered, chatter = [], False
    for ln in lines:
        m = LINE.match(ln)
        if m and not chatter:
            numbered.append((int(m.group(1)), m.group(2) or ""))
        elif numbered and ln.strip():
            chatter = True                 # text after the list (or a second list after chatter): ignored
    echoed = False
    k = len(example)
    if numbered and norm(probe, 0, numbered[0][1]) == example[0]:
        hits = sum(norm(probe, 0, a) == example[i] for i, (_, a) in enumerate(numbered[:k]))
        if hits >= math.ceil(k / 2):
            numbered, echoed = numbered[k:], True
    answers = {}
    for q, a in numbered:
        if 1 <= q <= n_q and q not in answers:
            t = a.strip()
            ph = qtext.get(q, "")
            if ph and t.lower().startswith(ph):
                t = t[len(ph):]
            answers[q] = norm(probe, q, t)
    style = {"n_answers": len(numbered), "chatter": float(chatter), "echo": float(echoed),
             "overflow": float(any(q > n_q for q, _ in numbered)), "len": len(response)}
    return answers, style


def load(path):
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    eps = sorted({r["endpoint"] for r in recs})
    cat = collections.defaultdict(lambda: collections.defaultdict(list))   # ep -> (probe,q) -> [answers]
    style = collections.defaultdict(list)
    text = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # ep -> prompt -> words
    raw = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        ep, pid, resp = r["endpoint"], r["id"], r["response"] or ""
        raw[ep][pid].append(resp)
        if pid in PROBES:
            ans, st = parse_probe(pid, resp, r.get("prompt", ""))
            for q, a in ans.items():
                cat[ep][(pid, q)].append(a)
            style[ep].append(st)
        else:
            text[ep][pid].update(WORD.findall(resp.lower()))
    return eps, cat, style, text, raw


def mmd_pair(ca, cb):
    """Delta-kernel MMD^2 averaged over shared questions (unbiased within terms) and 1 - cross agreement."""
    mm, cr = [], []
    for key in set(ca) & set(cb):
        a, b = ca[key], cb[key]
        cross = sum(x == y for x in a for y in b) / (len(a) * len(b))
        cr.append(1 - cross)
        if len(a) >= 2 and len(b) >= 2:
            wa = sum(a[i] == a[j] for i in range(len(a)) for j in range(i + 1, len(a))) / (len(a) * (len(a) - 1) / 2)
            wb = sum(b[i] == b[j] for i in range(len(b)) for j in range(i + 1, len(b))) / (len(b) * (len(b) - 1) / 2)
            mm.append(wa + wb - 2 * cross)
    return (float(np.mean(mm)) if mm else float("nan")), (float(np.mean(cr)) if cr else float("nan"))


def perm_p(ca, cb, n_perm, rng):
    obs, _ = mmd_pair(ca, cb)
    if math.isnan(obs):
        return float("nan")
    keys = [k for k in set(ca) & set(cb) if len(ca[k]) >= 2 and len(cb[k]) >= 2]
    ge = 0
    for _ in range(n_perm):
        pa, pb = {}, {}
        for k in keys:
            pool = ca[k] + cb[k]
            rng.shuffle(pool)
            pa[k], pb[k] = pool[:len(ca[k])], pool[len(ca[k]):]
        ge += mmd_pair(pa, pb)[0] >= obs - 1e-12
    return (1 + ge) / (1 + n_perm)


def cosine(c1, c2):
    keys = set(c1) | set(c2)
    v1 = np.array([c1[k] for k in keys], float)
    v2 = np.array([c2[k] for k in keys], float)
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(v1 @ v2 / d) if d else float("nan")


def style_vec(sts):
    if not sts:
        return np.zeros(4)
    return np.array([np.mean([s["chatter"] for s in sts]), np.mean([s["overflow"] for s in sts]),
                     np.mean([s["echo"] for s in sts]), np.mean([s["len"] for s in sts]) / 1000])


def auc(pos, neg):
    """P(distance of a same-model pair < distance of a different-model pair); 1 = perfect separation."""
    pos, neg = [x for x in pos if not math.isnan(x)], [x for x in neg if not math.isnan(x)]
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(p < n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("--practice", help="practice_answers.csv: score how well each signal separates same/different")
    ap.add_argument("--perm", type=int, default=0, help="permutation p-values for cat_mmd with this many shuffles")
    a = ap.parse_args()
    eps, cat, style, text, raw = load(a.answers)
    n = len(eps)
    tag = Path(a.answers).stem
    D = {k: np.full((n, n), np.nan) for k in ("cat_mmd", "cat_cross", "text", "style")}
    sv = {e: style_vec(style[e]) for e in eps}
    for i, ei in enumerate(eps):
        for j, ej in enumerate(eps):
            if j < i:
                for k in D:
                    D[k][i, j] = D[k][j, i]
                continue
            m, c = mmd_pair(cat[ei], cat[ej]) if i != j else (0.0, mmd_pair(cat[ei], cat[ei])[1])
            D["cat_mmd"][i, j], D["cat_cross"][i, j] = m, c
            cs = [cosine(text[ei][p], text[ej][p]) for p in set(text[ei]) & set(text[ej])]
            D["text"][i, j] = 1 - float(np.nanmean(cs)) if cs else float("nan")
            D["style"][i, j] = float(np.abs(sv[ei] - sv[ej]).sum())

    out = []
    pr = lambda s="": (print(s), out.append(s))
    pr(f"{n} endpoints, {sum(len(v) for e in eps for v in raw[e].values())} answers from {a.answers}")
    pr("\nPer endpoint: answers | probe answers parsed | chatter | overflow | echo | mean length | self-disagreement")
    for e in eps:
        nans = sum(len(v) for v in raw[e].values())
        parsed = sum(len(v) for v in cat[e].values())
        s = sv[e]
        pr(f"  {e}: {nans:3d} | {parsed:4d} | {s[0]:.2f} | {s[1]:.2f} | {s[2]:.2f} | {s[3] * 1000:6.0f} | "
           f"{D['cat_cross'][eps.index(e), eps.index(e)]:.2f}")
    pr("  (self-disagreement = chance two answers of the same endpoint to the same question differ; "
       "high values hint at T=1 sampling)")

    if "identity" in next(iter(raw.values()), {}):
        pr("\nIdentity answers (first 160 chars):")
        for e in eps:
            for t in raw[e].get("identity", [])[:1]:
                pr(f"  {e}: {t[:160]!r}")

    pr("\nNearest neighbours (cat_mmd | text):")
    for i, e in enumerate(eps):
        order = [j for j in np.argsort(np.nan_to_num(D["cat_mmd"][i], nan=9)) if j != i][:5]
        pr(f"  {e}: " + "  ".join(f"{eps[j]}({D['cat_mmd'][i, j]:.3f}|{D['text'][i, j]:.2f})" for j in order))

    if a.perm:
        rng = random.Random(0)
        pr(f"\nPermutation p-values on cat_mmd ({a.perm} shuffles; small p = different answer distributions):")
        P = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                P[i, j] = P[j, i] = perm_p(cat[eps[i]], cat[eps[j]], a.perm, rng)
        D["perm_p"] = P
        for i, e in enumerate(eps):
            pr(f"  {e}: " + " ".join(f"{P[i, j]:.2f}" if j != i else " -- " for j in range(n)))

    for k, M in D.items():
        with open(f"dist_{k}_{tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([""] + eps)
            for i, e in enumerate(eps):
                w.writerow([e] + [f"{x:.4f}" for x in M[i]])

    if a.practice:
        truth = {}
        for row in csv.reader(open(a.practice, encoding="utf-8-sig")):
            if row and row[0].startswith("ref-"):
                truth[row[0]] = (row[1] + "/" + row[2], row[3] if len(row) > 3 else "")
        pr("\nPractice check: separation of same-model vs different-model pairs (AUC 1 = perfect, 0.5 = useless)")
        for k, M in D.items():
            if k == "perm_p":
                M = -M                           # small p = different, so negate to read as a distance
            pos, neg = [], []
            for i in range(n):
                for j in range(i + 1, n):
                    if eps[i] in truth and eps[j] in truth:
                        (pos if truth[eps[i]][0] == truth[eps[j]][0] else neg).append(M[i, j])
            pr(f"  {k:10s} AUC = {auc(pos, neg):.3f}   same-model mean {np.nanmean(pos):.4f}   "
               f"different mean {np.nanmean(neg):.4f}")
        base = [e for e in eps if e in truth and truth[e][0].endswith("/S")]
        pr("\n  Distance from each endpoint to the base cluster (mean over the base endpoints, itself excluded):")
        for k in ("cat_mmd", "cat_cross", "text", "style"):
            pr(f"    {k}:")
            for e in eps:
                if e not in truth:
                    continue
                d = [D[k][eps.index(e), eps.index(b)] for b in base if b != e]
                pr(f"      {e} {truth[e][1]:28s} {np.nanmean(d):.4f}")

    Path(f"report_{tag}.txt").write_text("\n".join(out), encoding="utf-8")
    print(f"\nSaved report_{tag}.txt and dist_*_{tag}.csv")


if __name__ == "__main__":
    main()