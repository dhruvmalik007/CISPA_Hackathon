#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --time=01:00:00
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err

# Generic job: everything after "job.sh" is passed to python.
#   sbatch --job-name=r2_m2.2 job.sh unlearn.py --method soft ... --out runs/r2_m2.2
#   sbatch --job-name=diag    job.sh diagnose.py --out diag_r2 --kinds flip crop color --subs ...
cd /p/scratch/training2666/falcon1/FrenchyFighters/unlearning || exit 1
source .venv/bin/activate
srun python "$@"
echo "done!"