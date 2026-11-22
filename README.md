# CISPA Hackathon — FrenchyFighters

Source snapshot imported from the JURECA A100 working tree:

```text
jureca05:/p/scratch/training2666/falcon1/FrenchyFighters
```

The repository keeps source, prompts, compact experiment manifests, and runbooks only.
Model weights, datasets, virtual environments, caches, generated outputs, checkpoints,
and multi-hundred-MB archives are intentionally excluded.

## Areas

| Path | Task | Contents |
| --- | --- | --- |
| `Hackathon_Watermark_Localisation_Task/` | Task 1 | Watermark localisation/attack harness, `wmloc` utilities, submission template, release assets |
| `model_equality/` | Task 2 | Model-equality testing client, scoring, prompt/job builders, phase-0 reconstruction, compact ref/p1/calib results |
| `unlearning/` | Task 3 | NPO/SCRUB/SalUn variants, target fitting, diagnostics, repair, and submission scripts |
| `malik4/` | Task 2/3 | Personal experiments, round-3/4 evaluators, verifier code, local-proxy pairs, gate reports |

## Confirmed results at import time

- Task 3 (unlearning): best public score **0.2**, checkpoint `stage6-match-final`.
- Task 2: public submission `0.0`; Gate 1 did not pass (`0.727` proxy-trained and
  `0.531/0.612` ref-LOO folds), so the remaining `ep-*` query budget was not spent.
- Task 1: GPU-side watermark-localisation source imported; large train/test archives
  remain on JURECA and are deliberately excluded.

## Reproduction

Each task directory contains its own `README`, `requirements.txt`/`pyproject.toml`,
and run scripts. The missing data and checkpoints live at the JURECA paths documented in
the runbooks. No credentials or private keys are part of this repository.

## Attribution

The import is split into four commits for the FrenchyFighters contributors:
`falcon1`, `falcon2`, `kopparapu1`, and `malik4`. Commit trailers record
CommandCodeBot as the automation co-author.
