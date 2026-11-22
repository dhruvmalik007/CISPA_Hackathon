#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=00:40:00
#SBATCH --output=/p/scratch/training2666/malik4/unlearning/output/%j.out
#SBATCH --error=/p/scratch/training2666/malik4/unlearning/output/%j.err
#SBATCH --job-name=malik4-fullrun

# Full pipeline on an A100:
#   1. task_template.py (pinned, byte-identical to official) -> builds the model, runs the
#      server's exact preprocessing, predicts, and writes the baseline submission.pt
#   2. unlearning.baseline_profile -> profile orig over the WHOLE forget set
#   3. unlearning.evaluate -> PDF-formula score for orig vs sub
set -uo pipefail

MY=/p/scratch/training2666/malik4/unlearning
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
cd "$MY" || exit 1
export PYTHONPATH="$MY"
mkdir -p output/full output/gpu-metrics

echo "host=$(hostname) job=${SLURM_JOB_ID} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} gpus=${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# VRAM/RAM sampler: JURECA does not cgroup GPU devices, so record what is actually visible.
( while true; do
    echo "$(date -Is),$(hostname),$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 20
  done ) >> "output/gpu-metrics/${SLURM_JOB_ID}.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

echo "==================== STAGE 1: task_template.py (full) ===================="
"$VENV/bin/python" task_template.py 2>&1
echo "stage1_rc=$?"

echo "==================== STAGE 2: profile orig, all 3500 ===================="
"$VENV/bin/python" unlearning/baseline_profile.py \
  --data-dir . --limit 3500 --batch-size 64 --altered 1 \
  --output output/full/profile_all.json 2>&1
echo "stage2_rc=$?"

echo "==================== STAGE 3: PDF score, full forget set ===================="
"$VENV/bin/python" unlearning/evaluate.py \
  --data-dir . --orig model.pt --sub submission.pt \
  --limit 3500 --batch-size 64 --altered-variants 1 \
  --output-dir output/full 2>&1
echo "stage3_rc=$?"

echo "==================== ARTEFACTS ===================="
ls -la output/full/
echo "DONE rc=0"
