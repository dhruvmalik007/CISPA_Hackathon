#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/training2666/malik4/unlearning/output/%j.out
#SBATCH --error=/p/scratch/training2666/malik4/unlearning/output/%j.err
#SBATCH --job-name=malik4-scrub

set -uo pipefail
MY=/p/scratch/training2666/malik4/unlearning
U=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning
cd $MY || exit 1
export PYTHONPATH=$MY
echo "host=$(hostname) job=$SLURM_JOB_ID cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}"
mkdir -p output/gpu-metrics
( while true; do echo "$(date -Is),$(hostname),$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr "\n" ";")" >> output/gpu-metrics/${SLURM_JOB_ID}.csv; sleep 20; done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null || true" EXIT
$U/.venv/bin/python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
$U/.venv/bin/python -m unlearning.scrub \
  --data-dir . --output-dir output/stage6-match \
  --epochs 3 --lr 5e-5 --unfreeze-blocks 2 --anchor 1e-2 \
  --forget-objective match --match-kl-weight 1.0 --match-stats-weight 0.1 \
  --eval-steps 20 --forget-loss-cap 1.5 --retain-min-agreement 0.9 \
  --min-retain-score 0.9 --eval-limit 512
echo "DONE rc=$?"
