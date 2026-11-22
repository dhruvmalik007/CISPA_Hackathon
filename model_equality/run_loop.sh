#!/bin/bash
# usage: run_loop.sh GPU MODEL QUANTS EPS   -- scores new ep_p2.jsonl answers every pass until stop file exists
export HF_HOME=/p/scratch/training2666/falcon1/FrenchyFighters/.cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$1
while [ ! -f STOP_LOOP ]; do
  ../unlearning/.venv/bin/python score_v2.py --answers ep_p2.jsonl --jobs jobs_p2.jsonl --model $2 --fp8 --quants "$3" --endpoints $4 2>&1 | grep -v Loading
  sleep 60
done
