# Proxy findings — a real signal, and the reason it may not transfer

## Headline

On Doubleword's hosted endpoints, the **same char-4gram metric that measured AUROC 0.508 on
`ref-01..ref-09` reaches AUROC 0.887** for separating same-model from different-model pairs:

```
432 responses, 4 models x 3 disguises x 12 prompts x 3 samples, 0 empty

SAME model (different disguise): n=12   mean 0.6183   range 0.567 - 0.693
DIFFERENT model:                 n=54   mean 0.5728   range 0.525 - 0.617
AUROC (same => more similar): 0.887
margin on means: +0.0454
```

That is the first above-chance signal in this project, measured on real API responses.

**And it is consistent per model** — every model shows the same pattern, so it is not one endpoint
driving it:

| model | within-model similarity (disguise pairs) |
|---|---|
| DeepSeek-V4.1-Flash-1017 | 0.640 |
| Kimi-K3.1-Pro | 0.623 |
| GLM-5.3 | 0.610 |
| Qwen3.8-27B-FP8 | 0.601 |

### A caveat I have to correct from the smaller run

At n=3 the ranges did **not** overlap (same-min 0.537 > diff-max 0.503), which looked like perfect
separation. **At n=12 they do overlap**: same-model pairs run 0.567-0.693 while different-model pairs
reach 0.617. So the effect is a *distributional shift of about +0.045*, not a clean gap. Any threshold
placed in 0.567-0.617 will make errors.

AUROC 0.887 means: pick a random same-model pair and a random different-model pair, and the same-model
one scores higher 88.7% of the time. Useful, far from perfect, and nothing like the 1.000 the n=3 run
implied. The n=3 result was luck.

## Why this does not immediately mean the competition is solvable

The reference endpoints (`ref-01..ref-09`) are the organisers' own proxy for the scored set, and on
them the identical metric is at chance. Two readings:

1. **The reference set is too small.** 9 endpoints, one family, 5 of them the same model. That is 10
   same-model pairs against 26 different-model pairs, dominated by a watermark endpoint that no text
   metric tolerates.
2. **The scored endpoints are not like the reference set.** `ref-*` is described as "a separate small
   model", while `ep-*` self-report as Qwen/Llama/Claude/Gemma — much larger and more distinct.

Reading 2 is plausible and would be good news. It is also *unverified*, which is why the next step is a
cheap submission, not a bigger zoo.

## The signal is weaker than the reference-vs-proxy comparison suggests

Both numbers are computed on different data with different pair structures, so they are not directly
comparable:

| | ref-01..ref-09 | proxy zoo |
|---|---|---|
| endpoints | 9 | 4 models x 3 disguises = 12 cells |
| same-model pairs | 10 (all disguises **and** one watermark) | 12 (disguises only, no watermark) |
| AUROC | 0.508 | 0.887 |

The reference set contains a **watermark** endpoint, which the proxy has none of. My earlier work
showed the watermark endpoint `ref-06` dominates every distance metric. So a large part of the
ref-vs-proxy difference is that the proxy is *easier*: no watermark, and no quantization variants.

**That is the honest reading: the proxy proves "system-prompt disguises are learnable", which was
already the easy half.** It says nothing about watermarks or quantization, which are the parts that
actually decided the reference scores.

## Bugs found and fixed during the pilot

| Bug | Evidence | Fix |
|---|---|---|
| **Reasoning tokens starved the answer** | 9 of 36 responses had `content: None` and `finish_reason: length`; `completion_tokens_details.reasoning_tokens` was 256 of 256 | Send `reasoning_effort: "none"` by default; raise `max_tokens` to 1024 when reasoning is kept. After the fix: **0 of 36** empty, all `finish_reason: stop` |
| Batch generator had no reasoning control | `--keep-reasoning` did not exist | Added, with the measured default being off |
| Pilot checker silently dropped empty rows | Reported "9 without usable text" and then compared 9 vs 12 vs 6 samples, biasing every mean | Now counts `empty_content` separately and **fails the pilot** if more than 2% of rows are starved |
| `custom_id` could exceed the API's 64-char cap | not triggered, but unchecked | Generator refuses to write any id over 64 chars and dedupes |

## What the pilot could NOT confirm

**The `sys_concise` system prompt is only weakly obeyed.** concise/clean length ratio was **0.91**
against a 0.85 threshold. That is a *test-design* problem, not an API problem: the battery prompts
already demand short answers ("Answer with the city name only"), leaving almost nothing to compress.
A system prompt that says "be concise" cannot shorten a five-token answer.

So the honest conclusion is: **the disguise mechanism works well enough to be learnable** (the three
same-model pairs differ in *surface* form — 0.537 to 0.673 similarity — while staying closer to each
other than to any other model), but the specific "concise" check was mis-specified. The next pilot
should use prompts with room to vary, or drop that specific assertion.

## What to do next, in order

1. **Scale the pilot before scaling the batch**: 4 models × 3 disguises × 12 prompts × 3 samples
   ≈ 432 requests. That gives ~12 same-model pairs instead of 3, so the separation claim stops being
   an n=3 curiosity. Cost is trivial.
2. **Then decide on training.** If non-overlap holds at 12+ pairs across 4 models, a contrastive
   verifier has something real to learn and the full zoo is justified.
3. **Do not train yet.** A verifier trained on 3 positive pairs would memorise them.

## The one experiment that settles the competition question

The proxy says same-model disguise detection is *possible*. Whether it works on `ep-*` is untested,
and the reference set suggests it does not. The cheapest discriminator is a **single submission** using
the reference-set-cheapest rule (already built: `out/submission_v1.csv`), because the returned score
tells us the true group size. That costs one 30-minute slot and zero query budget, and it converts an
open question into a number.

That is strictly better than spending 64 responses per endpoint on a zoo trained against an
unverified premise.

## Commands used (all verified working)

```bash
# toolchain
python3 -m venv .venv-dw && .venv-dw/bin/pip install dw-cli==0.1.25
.venv-dw/bin/dw models list --type chat > proxy/catalog.jsonl

# zoo (derived from the live catalog, never hardcoded)
python proxy/make_zoo.py --catalog proxy/catalog.jsonl --out proxy/zoo.json

# pilot
python proxy/make_proxy_batch.py --zoo proxy/zoo.json --prompts prompts/battery.jsonl \
    --models 3 --disguises clean,sys_concise --prompts-limit 8 --samples 2 \
    --out proxy/pilot3_batch.jsonl
.venv-dw/bin/dw files stats proxy/pilot3_batch.jsonl
.venv-dw/bin/dw batches run proxy/pilot3_batch.jsonl --watch --output-id proxy/.pilot3_id
.venv-dw/bin/dw batches results "$(cat proxy/.pilot3_id)" -o proxy/pilot3_results.jsonl
python proxy/check_pilot.py --results proxy/pilot3_results.jsonl
```

Batch turnaround measured: **96 requests completed in seconds**, not the 24 h SLA. The `completion_window`
is 24h but actual completion was near-instant for a small batch, so the expensive part of this
approach is not wall-clock time.

## Facts worth keeping

- Catalog: **18 chat models**, all ≥27B, none with a same-family quantization pair. So this API cannot
  supply the hard negatives (same base, different quantization) that carry 90% of the score.
- `dw files stats` reported **1,257 input tokens** for 36 requests — cheap enough that the pilot
  approach (measure before scaling) costs essentially nothing and should always be used.
- No `--completion-window 1h` exists; `completion_window` is `24h` and the fast path is a different
  mechanism (`service_tier: "flex"`).
