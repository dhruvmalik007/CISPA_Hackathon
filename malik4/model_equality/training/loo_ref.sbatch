#!/bin/bash
#SBATCH --job-name=task2-ref-loo
#SBATCH --partition=dc-gpu
#SBATCH --account=training2666
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=output/task2-ref-loo-%j.out
#SBATCH --error=output/task2-ref-loo-%j.err
set -euo pipefail
cd /p/scratch/training2666/falcon1/FrenchyFighters/malik4/model_equality
source /p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv/bin/activate
python training/loo_ref.py
