#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=00:40:00
#SBATCH --output=/p/scratch/training2666/malik4/unlearning/output/%j.out
#SBATCH --error=/p/scratch/training2666/malik4/unlearning/output/%j.err
#SBATCH --job-name=malik4-salun

# ORGANISER RULE: one node with four GPUs maximum per participant at a time, and they cancel
# jobs and revoke access for offenders. This script therefore requests the minimum that works:
#   1 node, 1 task, 1 GPU, 8 CPUs.
# PEER SAFETY: JURECA does not cgroup GPU devices, so a login/compute node's nvidia-smi shows
# every card. CUDA_VISIBLE_DEVICES is pinned to the Slurm-allocated device so this process cannot
# touch a peer's GPU, and PYTHONDONTWRITEBYTECODE keeps __pycache__ out of the shared venv that
# belongs to the team lead (falcon1), which is writable to us.
#
# SalUn on the real forget set: mask -> train -> PDF-formula evaluation, all inside malik4/.
set -uo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

MY=/p/scratch/training2666/malik4/unlearning
# The team venv is a READ-ONLY reference. Nothing is installed or written into it.
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
cd "$MY" || exit 1
export PYTHONPATH="$MY"
mkdir -p output/salun-full output/gpu-metrics

# Allocate exactly the one GPU Slurm handed us, even if more are visible on the node.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "host=$(hostname) job=${SLURM_JOB_ID} nodes=${SLURM_NNODES:-?} tasks=${SLURM_NTASKS:-?}"
echo "cuda_visible=${CUDA_VISIBLE_DEVICES} gpus_on_node=${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Fairness preflight: refuse to proceed if this user somehow has another job live.
LIVE=$(squeue -u "$USER" -h -t RUNNING,CONFIGURING 2>/dev/null | grep -v "^ *${SLURM_JOB_ID} " | wc -l)
if [ "$LIVE" -gt 0 ]; then
  echo "ABORT: $LIVE other job(s) live for $USER; organiser rule violated" >&2
  exit 3
fi

# VRAM/RAM sampler so actual usage is on record.
( while true; do
    echo "$(date -Is),$(hostname),$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 15
  done ) >> "output/gpu-metrics/${SLURM_JOB_ID}.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

PY="$VENV/bin/python"

echo "==================== STAGE 1: SalUn, full forget set ===================="
"$PY" -m unlearning.salun \
  --data-dir . \
  --output-dir output/salun-full \
  --epochs 4 \
  --batch-size 16 \
  --eval-batch-size 64 \
  --lr 5e-3 \
  --keep-ratio 0.20 \
  --anchor-weight 0.5 \
  --cutmix-weight 0.5 \
  --forget-objective soft-runner-up \
  --projection fixed \
  --eval-limit 512 \
  --log-every 10 \
  --compute-mask
echo "stage1_rc=$?"

echo "==================== STAGE 2: PDF-formula evaluation ===================="
if [ -f output/salun-full/submission.pt ]; then
  "$PY" unlearning/evaluate.py \
    --data-dir . \
    --orig model.pt \
    --sub output/salun-full/submission.pt \
    --limit 3500 --batch-size 64 --altered-variants 1 \
    --output-dir output/salun-full/eval 2>&1 | tail -30
  echo "stage2_rc=$?"
else
  echo "stage2_skipped: no submission.pt"
fi

echo "==================== ARTEFACTS ===================="
ls -la output/salun-full/ | head -20
echo "--- summary.json (head) ---"
head -50 output/salun-full/summary.json 2>&1
echo "--- eval report ---"
cat output/salun-full/eval/report.json 2>&1 | head -40
echo "DONE"
