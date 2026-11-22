# Round 2 — what the measurements actually support

Written after round 1 was built. Every number here is reproduced by a script in `scripts/`; nothing is
estimated from intuition.

## The diagnosis in one line

**Membership is broken, not group size.** The ladder hedge (try several sizes, keep the best score)
was worth +0.19 to +0.29 in simulation, but **+0.000 on real reference data** — because the metric
picks the wrong *endpoints*, and no amount of trying different sizes fixes a wrong membership.

Evidence, running every rung through the organisers' own scorer on `ref-01..ref-09`:

| Submission | MCC |
|---|---|
| the truth (ref-01/03/05/06/08 grouped) | **1.000** |
| group of 8 (most central) | 0.331 |
| group of 7 | 0.021 |
| group of 6 | 0.231 |
| group of 5 ← the correct size | **0.031** |
| group of 4 | 0.000 |
| group of 3 | 0.037 |
| group of 2 | 0.000 |
| **best rung of the ladder** | **0.331** |

Note the k=5 row: the ladder tried the *correct* size and scored 0.031, because centrality ordered
`ref-02` (quantized — a different model) as more central than `ref-01` (plain — same model). The size
was right and the members were wrong.

The centrality ranking makes the error visible:

```
least -> most central:  ref-07 (0.212)  ref-01 (0.389)  ref-04 (0.396)  ref-05 (0.409)
                        ref-06 (0.427)  ref-09 (0.435)  ref-03 (0.488)  ref-02 (0.505)
                        ref-08 (0.509)
```

`ref-01` is a *same-model* endpoint yet ranks second-least central. `ref-02` is a *different* model
yet ranks second-most. Text similarity simply does not track model identity here, which is the same
finding as `VERDICT.md` from a different angle.

## Why the ladder is still worth running (cheaply)

Only the best score per team counts, and 48 submissions/day are allowed. So the ladder costs nothing
but wall-clock time — a low rung cannot damage your standing. It is a **free hedge against unknown
group size**, not a method.

Run it, but expect it to match rather than beat round 1.

## What will not work, and I will not propose it

| Idea | Why it fails |
|---|---|
| More prompts for the same metric | High-power test: AUROC **0.508** at 72 answers/endpoint. 10× the budget of a real submission and still chance. |
| RUT (rank-based uniformity test, the PDF's second reference) | Requires a **locally deployed authentic model** to compare against. We have none — that is the whole point of a black-box API. |
| Prompt engineering to force determinism | `temperature`/`seed` are ignored and system messages rejected by the API (PDF p.3). All sampling is at T=1. |
| Bigger MMD sample pools | Tested: pooling across prompts gives **negative** MMD² because within-endpoint pairs span different topics while between-endpoint pairs share a prompt. |
| Logprob-based distance (MET's actual estimator) | The API returns `choices[0].message.content` only — verified in `official/query_reference.py`. No logits, no token IDs. |

## The one genuinely untested dimension

Across everything I ran, the number of **prompts** never exceeded 9 complete ones. Repeats were tested
to 8 per prompt; prompt *count* was not. That is the only axis with no measurement behind it.

It is worth one bounded experiment, because it is cheap on the free reference set:

```bash
# free: reference endpoints have a 1,000,000 limit
cd model-equality
../orchestrator-app/.venv/bin/python scripts/collect_ref.py --prompts 61 --repeats 1 \
    --out data/ref_wide.jsonl
../orchestrator-app/.venv/bin/python scripts/power_eval.py --answers data/ref_wide.jsonl --min-repeats 1
```

If AUROC stays near 0.5 at 40-60 prompts, the question is closed for good and the honest recommendation
is to leave the remaining 88 responses/endpoint unspent. If it rises above ~0.65, that is the signal
round 3 should scale.

## Round 2 submission plan

1. **Round 1 first** (`out/submission_v1.csv`) — establishes the scored baseline.
2. **Wait 30 minutes** (cooldown is per accepted submission).
3. **Run the ladder** in order, largest group first:

```bash
for f in out/ladder/ladder_k32_d0.csv out/ladder/ladder_k25_d0.csv out/ladder/ladder_k19_d0.csv \
         out/ladder/ladder_k14_d0.csv out/ladder/ladder_k10_d0.csv out/ladder/ladder_k07_d0.csv \
         out/ladder/ladder_k05_d0.csv; do
  echo "=== $f ==="
  ./scripts/submit.sh "$f"
  sleep 1800          # 30-minute cooldown
done
```

4. **Stop early** if any score exceeds ~0.25: the ladder has found the right size and the remaining
   rungs only spend cooldowns.

## What round 1's score tells you

The score inverts to the true group size, roughly:

| If group size is k | true same-model pairs | expected score from "32+1" |
|---|---|---|
| 1 (all distinct) | 0 | 0.000 |
| 5 | 10 | 0.031 |
| 12 | 66 | 0.061 |
| 19 | 171 | 0.079 |
| 26 | 325 | 0.063 |
| 33 (all same) | 528 | 0.000 |

So a score near **0.00** is genuinely ambiguous: it means either "all distinct" or "all same", and no
further submission from this family will help. A score around **0.06-0.08** tells you the truth has one
group of ~12-26 endpoints, and the ladder's k=19 or k=25 rungs are the ones worth watching.

That is the one piece of information round 1 buys, and it costs a submission slot that was unused.

## Filtering the ladder before you spend cooldowns

Each rung is one submission. Pre-screen them on the free reference set first so you are not spending a
30-minute slot on a rung you could have known was bad:

```bash
# official scorer over every rung shape, on reference data
../orchestrator-app/.venv/bin/python - <<'PY'
import sys, csv, subprocess, tempfile
from pathlib import Path
eps = [f"ref-{i:02d}" for i in range(1, 10)]
truth = {r["endpoint"]: r["model"] for r in csv.DictReader(open("official/practice_answers.csv"))}

def score(sizes):
    labels, it = {}, iter(eps)
    for gi, s in enumerate(sizes, 1):
        for _ in range(s):
            labels[next(it)] = (f"F{gi}", f"F{gi}_M1")
    for ep in it:
        labels[ep] = (f"F{ep}", f"F{ep}_M1")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.csv"
        with p.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["endpoint", "family", "model"])
            for ep in eps: w.writerow([ep, *labels[ep]])
        r = subprocess.run([sys.executable, "official/score_practice.py", str(p)], capture_output=True, text=True)
    return next((float(l.rsplit(":", 1)[1]) for l in r.stdout.splitlines() if l.startswith("Practice score")), float("nan"))

for k in (8, 7, 6, 5, 4, 3, 2):
    print(f"group of {k}: MCC {score([k]):.3f}")
PY
```

## Honest expectation

Round 2 will most likely land between **0.00 and 0.10** on the public split. The ladder may add a little
if the true group size happens to be one of the rungs' sizes *and* centrality picks decent members, but
the reference test shows it matching rather than beating a single submission.

There is no SOTA path here without a signal that this API does not expose. Saying otherwise would be
inventing a result.
