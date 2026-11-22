#!/usr/bin/env python3
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/p/scratch/training2666/falcon1/FrenchyFighters/malik4/model_equality")
DATA = ROOT / "data/setprint_ref.jsonl"
TRUTH = ROOT / "official/practice_answers.csv"
WORK = ROOT / "output/ref_loo"
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "training"))
from eval_transfer import make_encoder, auroc

rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
LABEL = {"ref-01":"S","ref-03":"S","ref-05":"S","ref-06":"S","ref-08":"S",
         "ref-02":"T1","ref-04":"T3","ref-07":"T4","ref-09":"T2"}
truth = {}
with TRUTH.open() as handle:
    for row in csv.DictReader(handle):
        truth[row["endpoint"]] = (row["family"], row["model"])

def pair(row):
    return {"text": str(row["prompt"]) + "\n" + str(row["response"]),
            "model": LABEL[str(row["endpoint"])], "prompt_id": str(row["id"]),
            "disguise": str(row["endpoint"]), "family": "P", "source": "ref-setprint"}

ids = sorted({str(row["id"]) for row in rows})
aurocs = []
for pid in ids:
    train_path = WORK / f"train_{pid}.jsonl"
    held_path = WORK / f"held_{pid}.jsonl"
    verifier_path = WORK / "verifier_loo.pt"
    train_path.write_text("\n".join(json.dumps(pair(row)) for row in rows if str(row["id"]) != pid) + "\n")
    held = [row for row in rows if str(row["id"]) == pid]
    held_path.write_text("\n".join(json.dumps(row) for row in held) + "\n")
    train = subprocess.run([sys.executable, "training/train_verifier.py", "--pairs", str(train_path),
                            "--epochs", "3", "--batch-size", "48", "--out", str(verifier_path)],
                           cwd=ROOT, capture_output=True, text=True)
    if train.returncode != 0:
        print(f"FOLD {pid} TRAIN_FAIL", flush=True); print(train.stderr[-800:], flush=True); continue
    encoder = make_encoder(verifier_path)
    endpoints = sorted({str(row["endpoint"]) for row in held})
    texts = {str(row["endpoint"]): str(row["prompt"]) + "\n" + str(row["response"]) for row in held}
    same, diff = [], []
    for i, left in enumerate(endpoints):
        for right in endpoints[i+1:]:
            value = encoder(texts[left], texts[right])
            if truth[left] == truth[right]:
                same.append(value)
            else:
                diff.append(value)
    value = auroc(__import__("numpy").array(same), __import__("numpy").array(diff))
    aurocs.append(value)
    print(f"FOLD {pid} AUROC {value:.3f} same={len(same)} diff={len(diff)}", flush=True)

valid = [value for value in aurocs if value == value]
mean = sum(valid) / len(valid) if valid else float("nan")
print(f"LOO mean AUROC {mean:.3f} n={len(valid)} gate={'PASS' if valid and mean >= 0.85 else 'FAIL'}", flush=True)
