# Doubleword proxy zoo — runbook and findings

## Two things that change the plan, found before spending anything

### 1. The `ep-*` endpoints apparently self-identify by family — but it is hallucination

The `identity` prompt on the scored endpoints returns plausible-looking families:

| Self-reported | count | endpoints |
|---|---|---|
| Qwen | 17 | ep-01, 03, 04, 06, 07, 12, 14, 17, 18, 22, 24, 25, 26, 28, 30, 32, 33 |
| Llama | 6 | ep-10, 13, 15, 27, 29, 31 |
| unknown | 5 | ep-09, 16, 19, 21, 23 |
| Anthropic | 3 | ep-02, 08, 20 |
| Meta | 2 | ep-05, 11 |

This looked like the break we needed. It is not. Three independent tests kill it:

**a. It is anti-correlated with text similarity.** AUROC of "same self-report ⇒ more similar text" is
**0.298** — below chance, and *inverted* relative to a real signal.

**b. It is unstable across repeats.** On the reference set, where repeats exist:

```
ref-01 (truth: same model)  -> [('unknown', 1), ('smollm', 1)]     <- different answer each time
ref-08 (truth: same model)  -> [('unknown', 1), ('google', 1)]
ref-02 (truth: different)   -> [('unknown', 2)]                    <- consistently nothing
```

The same endpoint reports different families on repeat calls.

**c. It scores MCC 0.000 against known truth.** Feeding these families into the organisers' scorer on
`ref-01..ref-09` gives `0.000`, with 10 wrong pairs — every one of them a same-model pair wrongly
split because the self-reports differ.

**Conclusion: the identity answers are generated text, not metadata.** They are exactly the
"confident wrong answer" this task is built to punish. Do not use them, and do not build a proxy zoo
to predict them.

### 2. The catalog is not what the proposed recipe assumes

`dw models list --type chat` returns **17 chat models, all ≥27B**, and the IDs are Doubleword IDs, not
HuggingFace ones:

```
moonshotai/Kimi-K3.1-Pro                        meta/muse-spark-open
deepseek-ai/DeepSeek-V4.1-Flash-1017            deepseek-ai/DeepSeek-V4.1-Flash
meta-models/Muse-Glimmer-30B                   deepseek-ai/DeepSeek-V4-Pro-0813
zai-org/GLM-5.3                                 zai-org/GLM-5.3-Flash
Qwen/Qwen3.8-27B-FP8                            nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
Qwen/Qwen3.6-35B-A3B-FP8                        google/gemma-4-31B-it
Qwen/Qwen3.5-35B-A3B-FP8-dottxt                 Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
lightonai/LightOnOCR-2-1B-bbox-soup             moonshotai/kimi-k3
zai-org/GLM-5.2-FP8
```

Neither `Qwen/Qwen3-30B-A3B-Instruct-2507` nor `meta-llama/Llama-3.3-70B-Instruct` (the IDs in the
proposed recipe) exists here. And **20 older models were retired on 30 Oct 2026**, so a hardcoded list
would rot. `dw models list` is the only authority.

**No same-family quantization pairs are available.** The proposed plan assumed `-FP8` vs base pairs
exist. In this catalog every model is a distinct checkpoint: there is no `X` and `X-FP8` pair. So
Doubleword cannot supply the *hard negatives* (same base, different quantization) that carry 90% of
the score. It can only supply:
- same checkpoint + different disguise → **positives** (which the API can do)
- different checkpoints → **easy negatives**, mostly cross-family

That inverts the proposed division of labour. The part a hosted API can do is the *easy* part.

---

## What the proxy zoo is still worth

A contrastive verifier trained on "same checkpoint, different system prompt / temperature ⇒ same
embedding" is a genuine hypothesis, and it is the part the API does cheaply. But the negatives it
learns to reject are cross-family, which is the 10% term.

So the runbook below is **calibrated to prove or kill the hypothesis cheaply**, not to build the full
7,680-request zoo before knowing whether the idea transfers. Full zoo first, gate second, is the
expensive order.

## Phase 0 — install (done)

```bash
python3 -m venv .venv-dw
.venv-dw/bin/pip install dw-cli==0.1.25
.venv-dw/bin/dw --version          # dw 0.1.25
.venv-dw/bin/dw account current    # Precise Inference 6957 (personal)
```

## Phase 1 — capture the catalog

```bash
cd model-equality
../.venv-dw/bin/dw models list --type chat > proxy/catalog.jsonl
wc -l proxy/catalog.jsonl          # 17
```

Build `proxy/zoo.json` from it, choosing by what is actually useful:

- **Positives** need only ONE model — disguises are free. Start with the cheapest capable one.
- **Negatives** want breadth: pick 4-6 across distinct families
  (`Qwen/*`, `zai-org/GLM-*`, `deepseek-ai/*`, `google/gemma-4-31B-it`, `meta-models/*`).
- Avoid `lightonai/LightOnOCR-2-1B-bbox-soup` (OCR-only, carries a usage warning about repeating the
  prompt) and `Qwen/Qwen3-VL-235B-A22B-Instruct-FP8` (vision-priced, most expensive here).

```bash
python proxy/make_zoo.py --catalog proxy/catalog.jsonl --out proxy/zoo.json
```

## Phase 2 — SMALL pilot batch first (the cost control)

The proposed 4 models × 6 disguises × 40 prompts × 8 samples = 7,680 requests is ~1.2 M output tokens.
That is affordable, but pointless to spend before knowing the labels separate.

**Pilot: 1 model × 4 disguises × 6 prompts × 2 samples = 48 requests.** Cost is negligible, and it
answers two questions: does the API honour a system prompt, and does `temperature` visibly change
output?

```bash
python proxy/make_proxy_batch.py --zoo proxy/zoo.json --prompts prompts/battery.jsonl \
    --models 1 --disguises clean,sys_concise,temp_high --prompts-limit 6 --samples 2 \
    --out proxy/pilot_batch.jsonl
../.venv-dw/bin/dw files validate proxy/pilot_batch.jsonl
../.venv-dw/bin/dw files stats proxy/pilot_batch.jsonl        # tokens + cost, nothing spent
../.venv-dw/bin/dw batches run proxy/pilot_batch.jsonl --watch --output-id proxy/.pilot_id
../.venv-dw/bin/dw batches results "$(cat proxy/.pilot_id)" -o proxy/pilot_results.jsonl
```

Then check the pilot actually produced distinguishable text before scaling up:

```bash
python proxy/check_pilot.py --results proxy/pilot_results.jsonl
```

It reports, per disguise: response length, whether the system prompt is obeyed, and the pairwise
similarity between disguises. If `clean` and `sys_concise` are *identical*, the API is ignoring the
system prompt and the whole positive-construction premise is dead — better to learn that for $0.01
than after 7,680 requests.

## Phase 3 — full batch, only if the pilot shows separation

```bash
python proxy/make_proxy_batch.py --zoo proxy/zoo.json --prompts prompts/battery.jsonl \
    --samples 4 --out proxy/proxy_batch.jsonl
../.venv-dw/bin/dw files validate proxy/proxy_batch.jsonl
../.venv-dw/bin/dw files stats proxy/proxy_batch.jsonl
../.venv-dw/bin/dw batches run proxy/proxy_batch.jsonl --watch --output-id proxy/.batch_id
../.venv-dw/bin/dw batches results "$(cat proxy/.batch_id)" -o proxy/proxy_results.jsonl
```

`custom_id` format (capped at **64 chars** by the API):

```
ep|family|model|disguise|prompt_id|sample_idx
```

## Phase 4 — training set

```bash
python proxy/build_training_set.py --results proxy/proxy_results.jsonl \
    --out training/proxy_pairs.jsonl
```

Asserts every `(model, disguise)` cell is complete before writing.

## Phase 5 — train

```bash
npx skills add huggingface/skills@train-sentence-transformers -g -y
python training/train_verifier.py --pairs training/proxy_pairs.jsonl \
    --model microsoft/deberta-v3-base --epochs 3 --out training/verifier.pt
```

## Phase 6 — THE GATE (do not skip)

```bash
python training/eval_transfer.py --verifier training/verifier.pt \
    --answers data/ref_power.jsonl
```

Exits non-zero if reference AUROC < **0.65**. Baseline to beat: **0.508** for every hand-built metric
at maximum power. If this fails, the proxy zoo hypothesis is dead and `ep-*` budget stays unspent.

## Phase 7 — only if the gate passes

Two-stage spend, not one: 2 prompts × 8 samples (16/endpoint), check consistency, then the rest.

## Cost note

Batch is ≈50% below realtime. There is **no `--completion-window 1h`** — the docs specify
`completion_window: "24h"` only. The faster path is a different mechanism: async tier via
`service_tier: "flex"` (≈1 min TTFT, ≈25% below realtime). Batch is the right choice here.

## What I am not doing

- Not hardcoding model IDs (they change; 20 were retired on 30 Oct 2026).
- Not using the `identity` self-reports — measured MCC 0.000, unstable across repeats, anti-correlated
  with text similarity.
- Not building the 7,680-request zoo before the pilot proves the labels separate.
- Not claiming the verifier will work. It is a hypothesis with a plausible mechanism, against evidence
  that every text statistic so far sits at chance.
