# Round 4 — gate results

## Gate 1

### Existing proxy-trained verifier

`training/verifier.pt` on `data/ref_power.jsonl`:

- reference AUROC: **0.354**
- char-4gram baseline: 0.512
- gate: **FAIL** (needs 0.65)

The verifier learned the Doubleword proxy distribution but does not transfer to the reference endpoints.

### SetPrint

Collected 5 base prompts x 5 perturbations x 9 ref endpoints = **225** answers in
`data/setprint_ref.jsonl`. Embedder: `sentence-transformers/all-mpnet-base-v2`.

Per-base edit/change distances:

| base | AUROC | spectral k | practice MCC |
|---|---:|---:|---:|
| 1 | 0.581 | 2 | 0.000 |
| 2 | 0.338 | 2 | 0.000 |
| 3 | 0.615 | 2 | 0.000 |
| 4 | 0.400 | 2 | 0.000 |
| 5 | 0.508 | 2 | 0.000 |

Leave-one-base-out mean MCC: **0.000**.
All-base mean MCC: **0.000**.
Gate 1: **FAIL** (needs 0.85).

The response-change vectors did not separate the five same-model S endpoints from the four different
variants.

## Gate 2

### Doubleword proxy zoo

`training/proxy_pairs.jsonl`: 4 models x 3 disguises x 12 prompts x 3 samples = 432 responses.
All-mpnet energy distance over the 12 `(family, model, disguise)` cells:

- pairwise AUROC (same model vs different model): **0.685**
- leave-one-family-out mean MCC: **0.294**
- spectral eigengap selected **k=2** for every held-out family, not the true four models
- Gate 2: **FAIL** (needs 0.85)

### Structural blocker

The Doubleword catalog has **no same-family quantization/LoRA pairs**. The proxy can supply
same-checkpoint/disguise positives and cross-family easy negatives, but not the within-family hard
negatives that determine the 90% model term. This is recorded in `proxy/RUNBOOK.md` and was confirmed
live here.

The missing asset is a **local GPU proxy zoo** with:
- two or more open-weight base families,
- LoRA adapters / int4 or int8 variants of the same base,
- a KGW or equivalent watermark variant,
- held-out families and held-out within-family variants.

## Budget

No `ep-*` responses were spent in this round. The remaining budget is still approximately
**87 responses per endpoint** (the 33 round-3 `ft01` responses plus one diagnostic are the only new
spend).

No Task 2 submission was made from round 4.

## Files

- `data/setprint_ref.jsonl` — 225 SetPrint reference answers
- `scripts/setprint_eval.py` — SetPrint distances, LOO, spectral clustering, proxy Gate 2
- `out/setprint_eval.json` — full gate evidence
- `training/verifier.pt` — the failed proxy-trained verifier
- `proxy/RUNBOOK.md` — catalog/hard-negative limitation

## What is needed to continue

Access to the local GPU proxy environment (JURECA reserved A100s or Modal GPU) to generate the
hard-negative corpus. Once that exists, Methods 7 and 13 can be trained and re-tested against Gate 1;
the current Doubleword-only proxy cannot satisfy Gate 2.
