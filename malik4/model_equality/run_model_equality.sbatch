#SBATCH --output=/p/scratch/training2666/falcon1/FrenchyFighters/malik4/model_equality/output/%j.out
# ...
cd /p/scratch/training2666/falcon1/FrenchyFighters/malik4/model_equality
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/p/scratch/training2666/falcon1/FrenchyFighters/.cache/huggingface
export PYTHONDONTWRITEBYTECODE=1
VENV=/p/scratch/training2666/falcon1/FrenchyFighters/unlearning/.venv
