#!/bin/bash
cd /p/scratch/training2666/falcon1/FrenchyFighters/model_equality
export HF_HOME=/p/scratch/training2666/falcon1/FrenchyFighters/.cache HF_HUB_OFFLINE=1
A=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
while [ ! -f STOP_LOOP ]; do ../unlearning/.venv/bin/python score_v2.py --answers ep_p2.jsonl --jobs jobs_p2.jsonl --model Qwen/Qwen2.5-3B-Instruct --quants int8,int4 --tag b --endpoints $A 2>&1 | grep -v Loading; sleep 30; done
