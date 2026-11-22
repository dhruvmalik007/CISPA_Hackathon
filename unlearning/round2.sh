#!/bin/bash
# Round 2: four 1-GPU jobs in parallel. Run from the unlearning folder on the login node: bash round2.sh
set -e
cd /p/scratch/training2666/falcon1/FrenchyFighters/unlearning
mkdir -p output runs

if [ -f proxy.npz ]; then PROXY="--proxy-npz proxy.npz"; else PROXY="--proxy-weight 0.3"; echo "no proxy.npz: using mosaics at weight 0.3"; fi
B="--method soft --diag diag/per_image.npz --aug-prob 0.3 $PROXY"
FULL="--full-classes 38 29 7 8 0 30 33 31 6 --full-loss 6"

sbatch --job-name=r2_m2.2      job.sh unlearn.py $B --loss-lo 0.3 --loss-hi 6 --gamma 2 --out runs/r2_m2.2
sbatch --job-name=r2_m3.3      job.sh unlearn.py $B --loss-lo 1.0 --loss-hi 8 --gamma 2 --out runs/r2_m3.3
sbatch --job-name=r2_m2.2_full job.sh unlearn.py $B --loss-lo 0.3 --loss-hi 6 --gamma 2 $FULL --out runs/r2_m2.2_full
sbatch --job-name=r2_m2.2_b4   job.sh unlearn.py $B --loss-lo 0.3 --loss-hi 6 --gamma 2 --train-blocks 4 --out runs/r2_m2.2_b4
squeue -u "$USER"