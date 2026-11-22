# Task 3: Model equality testing — working notes

Source: `Model Equality Testing.pdf` (European AI Championship Paris 2026), plus the official
files in `official/`, cloned from `huggingface.co/datasets/SprintML/model_equality` and verified
byte-identical.

## The binding constraint is the query budget, not compute

**100 responses per endpoint for the entire competition.** 33 endpoints, so ~3,300 responses total.

| Limit | Value | Consequence |
|---|---|---|
| Responses per endpoint | **100 total, ever** | Irreversible. Every query is capital. |
| Concurrent requests | 4 | Throughput ceiling, not a real constraint here |
| Max output tokens | 256 | Bounds response length |
| Max input tokens | 1,024 | Bounds prompt length |
| Submissions | 1 per 30 min | ~48/day, plenty; but a *failed* submission also burns the cooldown |

This means the method cannot be the one in the cited paper (MET). MET fits output *distributions*
and needs thousands of samples per endpoint to estimate them. With one sample per prompt, the
effective sample size is the number of **prompts**, so the design must be paired and the statistics
must be exact at small n.

## Free development surface

`ref-01`..`ref-09` are **unlimited and never scored**, with published answers. All method
development happens there. The scored endpoints are only touched once the method is frozen.

The published answers reveal the trap structure, and it is not what "similar text" would suggest:

| Endpoint | Disguise | Same model? |
|---|---|---|
| ref-01 | plain | reference |
| ref-02 | quantized (fp8) | **different** |
| ref-03 | system prompt | same |
| ref-04 | LoRA strength 0.7 | **different** |
| ref-05 | persona | same |
| ref-06 | watermark | same |
| ref-07 | LoRA strength 1.5 | **different** |
| ref-08 | sampling temperature 1.0 | same |
| ref-09 | LoRA strength 0.3 | **different** |

Two things follow. **LoRA at *any* strength counts as a different model** — including strength 0.3,
which produces text very close to the base. And **four quite different disguises leave the model
identical**. A method that measures "how similar is the text" fails in both directions: it
under-merges watermarked pairs and over-merges lightly-adapted LoRA ones.

## Method

```
answers.jsonl -> paired similarity per endpoint pair -> paired sign-flip permutation test
             -> Benjamini-Hochberg FDR -> union-find components -> endpoint,family,model
```

**Paired by prompt.** The same prompt goes to every endpoint, so each pair is compared on identical
inputs. This removes prompt difficulty from the comparison: a hard prompt is hard for both
endpoints, and only the residual difference is evidence. It is also why one sample per prompt is
usable at all.

**Similarity** is character 4-gram Jaccard on normalised text. Chosen deliberately over exact match:
exact match would label every watermarked pair "different", which is the known trap. A watermark is
a small insertion and costs little overlap; a genuinely different continuation costs almost all of
it.

**Test.** Paired sign-flip permutation test on the mean paired similarity. Exact for any n, no
distributional assumption, and it detects *both* directions — significantly dissimilar is also
evidence of different models. P-values are smoothed by +1 to avoid the zero that would break the FDR
step.

**Multiple testing is not optional.** 528 pairs at α=0.05 invents ~26 "same" pairs by chance. The
step-up Benjamini-Hochberg correction is applied over all pairs, and a pair must additionally clear
an absolute similarity floor — without that floor, a uniformly *dissimilar* pair can be
"significantly dissimilar" and would be misread as agreement.

**Family vs model.** MCC_model is 90% of the score and is computed only over pairs inside one true
family. Families are therefore derived with a looser threshold (FDR 0.25, floor 0.35) than models
(FDR 0.05, floor 0.50): a missed family link costs little, while over-merging families would let
cross-family "same model" claims destroy the 90% term.

## Validated offline against the organisers' own scorer

`scripts/e2e_check.py` writes synthetic answers in the exact format `query_reference.py` produces,
runs the real `solve.py` as a subprocess, and scores with `official/score_practice.py`.

```
Practice score (MCC over 36 pairs, same model or not): 1.000
Right: 36 pairs, wrong: 0
```

Recovery as a function of prompt count (measured, not assumed):

| Prompts | Practice MCC | Note |
|---|---|---|
| 5 | 0.000 | too few; detects nothing (and invents nothing — 0 false positives) |
| 8 | **1.000** | minimum for perfect recovery |
| 12 | 1.000 | |
| 20 | 1.000 | |
| 30-40 | 1.000 | |

**So 15-25 prompts is the target on the real endpoints**, leaving most of the 100-response budget
unspent as a reserve. One prompt to all 33 endpoints costs 33 responses; 20 prompts costs 660 of
3,300 (20%).

Caveat on the 1.000: the practice set is a single family, so `MCC_family` is mathematically
degenerate there (one pair, one true negative, no positives) and `score_practice.py` scores only the
model question. A unit test documents this rather than hiding it.

## What is blocked

**No API key.** The server answers `401 {"detail":"Missing X-API-Key"}`, and `HACKATHON_API_KEY` is
set in neither `.env`. Nothing has been queried, so **the budget is untouched at 100/100 per
endpoint**.

To proceed, a key is needed. Then:

```bash
# 1. sanity: budget should be 100 per endpoint, 0 used
python scripts/query.py --mode budget

# 2. DEVELOP on the free reference endpoints only
python scripts/query.py --mode reference --prompts prompts/seed.txt --dry-run   # preview cost
python scripts/query.py --mode reference --prompts prompts/seed.txt
python scripts/solve.py --answers data/reference_answers.jsonl --out out/ref.csv --json
python official/score_practice.py out/ref.csv

# 3. only once the reference MCC is satisfactory:
python scripts/query.py --mode scored --prompts prompts/final.txt --dry-run
python scripts/query.py --mode scored --prompts prompts/final.txt
python scripts/solve.py --answers data/answers.jsonl --out out/submission.csv --json
python official/submission.py    # mind the 30-minute cooldown
```

`--dry-run` performs the full budget check and returns the plan **without sending anything**. That
is the only safe way to preview an experiment, and it is enforced in `met/client.py`: the plan is
intersected with the server's own accounting before any request goes out.

## Files

| Path | Purpose |
|---|---|
| `official/` | organisers' scripts, byte-identical to the Hub |
| `met/client.py` | budget-safe querying; `plan()` never spends, `collect()` cannot overspend |
| `met/stats.py` | similarity, permutation test, BH correction, MCC, competition score |
| `met/simulate.py` | offline harness mirroring the published trap structure |
| `scripts/solve.py` | answers -> submission, production entry point |
| `scripts/query.py` | CLI for budget / reference / scored modes |
| `scripts/e2e_check.py` | synthetic end-to-end against the organisers' scorer |
| `tests/test_met.py` | 32 tests: statistics, decision rule, budget safety |

## Honest gaps

- The method has **never touched a real endpoint**. Every number above comes from the simulator,
  which encodes the published structure but cannot reproduce the real models' text. Expect the
  real similarity separation to be worse than the simulated one, which is exactly why the reference
  endpoints are the next step and not a formality.
- The similarity floor and FDR thresholds (0.50 / 0.05 / 0.35 / 0.25) are set from the simulator.
  They must be recalibrated on `ref-01..ref-09` before being used on scored endpoints.
- The family/model split cannot be validated on practice, because practice has one family. That
  logic is reasoned rather than measured, and the 90% weight on the model term means a family
  error is cheap — but it is untested.
