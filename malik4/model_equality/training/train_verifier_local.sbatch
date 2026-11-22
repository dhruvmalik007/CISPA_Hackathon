#!/bin/bash
#SBATCH --job-name=task2-verifier-local
#SBATCH --partition=dc-gpu
#SBATCH --account=training2666
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=output/task2-verifier-local-%j.out
#SBATCH --error=output/task2-verifier-local-%j.err
set -euo pipefail
cd /p/scratch/training2666/falcon1/FrenchyFighters/malik4/model_equality
source /p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv/bin/activate
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python training/train_verifier.py \
  --pairs training/local_proxy_pairs.jsonl \
  --model microsoft/deberta-v3-base \
  --epochs 3 \
  --out training/verifier_local.pt
python training/eval_transfer.py \
  --verifier training/verifier_local.pt \
  --answers data/setprint_ref.jsonl \
  --threshold 0.85
