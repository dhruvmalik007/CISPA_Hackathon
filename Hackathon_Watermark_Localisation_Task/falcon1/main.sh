#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --reservation=cispahack
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

cd "$SLURM_SUBMIT_DIR" || exit 1
source ../.venv/bin/activate
export PYTHONPATH=..
srun python step2_shapes.py --data ../release
