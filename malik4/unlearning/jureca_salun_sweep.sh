#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=01:30:00
#SBATCH --output=/p/scratch/training2666/malik4/unlearning/output/%j.out
#SBATCH --error=/p/scratch/training2666/malik4/unlearning/output/%j.err
#SBATCH --job-name=malik4-salun-sweep

# SalUn hyperparameter sweep. The paper's defaults are gamma=median (50% sparsity) and lr in
# [5e-4, 5e-2], so the arms bracket that range. Each arm gets its own mask file because the mask
# depends on keep_ratio but not on lr.
set -uo pipefail

MY=/p/scratch/training2666/malik4/unlearning
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
cd "$MY" || exit 1
export PYTHONPATH="$MY"
mkdir -p output/salun-sweep output/gpu-metrics

echo "host=$(hostname) job=${SLURM_JOB_ID}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

( while true; do
    echo "$(date -Is),$(hostname),$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
    sleep 15
  done ) >> "output/gpu-metrics/${SLURM_JOB_ID}.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

PY="$VENV/bin/python"

# Arms: keep_ratio lr epochs anchor cutmix objective projection
# The paper's defaults are gamma=median (50% sparsity) and lr in [5e-4, 5e-2], so the lr arms
# bracket that range. `random-label` is the paper's own objective and `proximal` is its RL_pro
# variant; both exist so the soft runner-up deviation is falsifiable rather than assumed better.
ARMS=(
  "0.20 5e-3 4 0.5 0.5 soft-runner-up fixed"
  "0.10 5e-3 4 0.5 0.5 soft-runner-up fixed"
  "0.20 5e-2 4 0.5 0.5 soft-runner-up fixed"
  "0.20 5e-4 4 0.5 0.5 soft-runner-up fixed"
  "0.20 5e-3 4 0.1 0.5 soft-runner-up fixed"
  "0.20 5e-3 4 0.5 0.0 soft-runner-up fixed"
  "0.20 5e-3 4 0.5 0.5 random-label fixed"
  "0.20 5e-3 4 0.5 0.5 soft-runner-up proximal"
)

for arm in "${ARMS[@]}"; do
  read -r keep lr epochs anchor cutmix objective projection <<< "$arm"
  tag="k${keep}_lr${lr}_a${anchor}_c${cutmix}_${objective}_${projection}"
  echo "==================== ARM $tag ===================="
  "$PY" -m unlearning.salun \
    --data-dir . \
    --output-dir "output/salun-sweep/$tag" \
    --epochs "$epochs" \
    --batch-size 16 \
    --eval-batch-size 64 \
    --lr "$lr" \
    --keep-ratio "$keep" \
    --anchor-weight "$anchor" \
    --cutmix-weight "$cutmix" \
    --forget-objective "$objective" \
    --projection "$projection" \
    --eval-limit 512 \
    --compute-mask 2>&1 | tail -25
  echo "arm_rc=$? tag=$tag"
done

echo "==================== SWEEP TABLE ===================="
"$PY" - "$MY" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]) / "output" / "salun-sweep"
rows = []
for summary_path in sorted(root.glob("*/summary.json")):
    data = json.loads(summary_path.read_text())
    rows.append({
        "arm": summary_path.parent.name,
        "best_epoch": data["best"]["epoch"],
        "proxy": round(data["best"]["proxy_score"], 5),
        "ok": data["submission"]["ok"],
        "kept": round(data["mask"].get("effective_ratio_after_gate", data["mask"]["effective_ratio"]), 4),
    })
rows.sort(key=lambda r: r["proxy"], reverse=True)
print(f"{'arm':40s} {'proxy':>9s} {'ep':>3s} {'kept':>6s} ok")
for row in rows:
    print(f"{row['arm']:40s} {row['proxy']:9.5f} {row['best_epoch']:3d} {row['kept']:6.3f} {row['ok']}")
(root / "sweep_table.json").write_text(json.dumps(rows, indent=2))
PYEOF

echo "DONE"
