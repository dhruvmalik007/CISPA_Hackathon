# Round 3 — constrained-prompt result

## Reference experiment

Merged the available reference prompts into one 60-prompt set:

- `ref_p2` — 40 prompts
- `ref_calib` — 10 prompts
- `ref_power` — 10 high-repeat prompts

Per-prompt AUROC was measured separately for exact match, normalized edit distance, and char-3-gram MMD. The round-3 claim that the top 8–10 prompts would each clear 0.80 did not reproduce:

- Only `ft01` cleared the individual practice gate: edit-distance AUROC **0.808**, best thresholded practice MCC **0.935** (`official/score_practice.py` confirmed 0.935).
- `ft02` was next at 0.734; `probeB`/`probeD` were 0.734/0.721.
- Averaging the top 5, 8, or 10 prompts lowered the practice MCC to **0.721–0.734**. The signal was concentrated in one prompt, not the top-k set.

`ft01` prompt:

```text
Continue this encyclopedia article for about 200 words: 'Photosynthesis is the process by which green plants'
```

Calibration file: `data/ref_round3_ft01_r1.jsonl`
Threshold: `0.871066` on normalized edit distance.

## Scored application

- Collected one `ft01` response on each of the 33 scored endpoints: `data/ep_round3_ft01_r1.jsonl`.
- Applied the reference threshold to the 33×33 edit-distance matrix.
- Result: 10 model clusters; `M1` contains 20 endpoints and ten endpoints are singletons.
- Submission: `out/submission_round3.csv`.
- Validator: `valid: format matches the organisers' contract`.
- Submission id: **3207**.
- Public score: **0.0**.
- Server message: previous best on task 2 still leads.

## What this means

The reference gate was real but did not transfer to the scored endpoints. A threshold that reached MCC 0.935 on the nine practice endpoints produced a wrong 33-endpoint partition. This is consistent with the earlier finding that reference membership is not a stable predictor of scored membership; the one-prompt result is reference-set overfitting, not a scoring signal.

Budget spent: 33 `ft01` responses plus one diagnostic `ep-01` request. Roughly 86–87 responses per endpoint remain. No further task-2 budget should be spent on this path.

The unlearning result remains the strongest team result at **0.0496** public.
