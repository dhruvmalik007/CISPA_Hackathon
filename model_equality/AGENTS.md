# Model equality (task 33) - notes

- Python with torch/transformers: `../unlearning/.venv/bin/python` (this task's .venv has no torch). Set
  `HF_HOME=/p/scratch/training2666/falcon1/FrenchyFighters/.cache HF_HUB_OFFLINE=1`. RTX 8000 (Turing): use fp32/fp16, not bf16.
- Base models (best local likelihood): family A = Qwen/Qwen2.5-3B-Instruct, B = Qwen/Qwen2.5-1.5B-Instruct,
  C = unsloth/Llama-3.2-1B-Instruct (ep lists in final.py); reference endpoints = HuggingFaceTB/SmolLM2-360M-Instruct.
- Pipeline: collect.py (jobs_p2.jsonl, 40 long free-text prompts) -> score_v2.py (per-answer lp/entropy/rank and
  simulated-quantization llr's) -> final.py (paired per-prompt comparison to the family median, rules calibrated on
  ref: `python final.py ref && python score_practice.py final_ref.csv`) -> submission_final.csv.
- Server latency ~20 s per answer (4 parallel) -> ~11 answers/min. Public leaderboard uses very few pairs.
