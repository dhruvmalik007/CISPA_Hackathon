#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=00:25:00
#SBATCH --output=/p/scratch/training2666/malik4/unlearning/output/%j.out
#SBATCH --error=/p/scratch/training2666/malik4/unlearning/output/%j.err
#SBATCH --job-name=malik4-salun-smoke

# SalUn smoke test on one A100: proves the mask, masked SGD, and both loss terms actually run on
# the cluster before spending time on a hyperparameter sweep. Uses a small forget subset so a
# failure surfaces in minutes.
set -uo pipefail

MY=/p/scratch/training2666/malik4/unlearning
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
cd "$MY" || exit 1
export PYTHONPATH="$MY"
mkdir -p output/salun-smoke output/gpu-metrics

echo "host=$(hostname) job=${SLURM_JOB_ID} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} gpus=${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

( while true; do
    echo "$(date -Is),$(hostname),$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 15
  done ) >> "output/gpu-metrics/${SLURM_JOB_ID}.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

echo "==================== salun smoke (300 images, 1 epoch, mask+train) ===================="
"$VENV/bin/python" -m unlearning.salun \
  --data-dir . \
  --output-dir output/salun-smoke \
  --train-limit 300 \
  --eval-limit 128 \
  --epochs 1 \
  --batch-size 8 \
  --lr 5e-3 \
  --keep-ratio 0.20 \
  --compute-mask 2>&1
echo "smoke_rc=$?"

echo "==================== ARTEFACTS ===================="
ls -la output/salun-smoke/
echo "--- summary.json ---"
cat output/salun-smoke/summary.json 2>&1 | head -60
echo "DONE"
