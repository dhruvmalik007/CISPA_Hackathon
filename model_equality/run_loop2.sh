#!/bin/bash
# Final scoring loop: B and C on GPU 0 (one after the other), A on GPU 1. Stops when STOP_LOOP exists.
export HF_HOME=/p/scratch/training2666/falcon1/FrenchyFighters/.cache HF_HUB_OFFLINE=1
A=ep-01,ep-02,ep-04,ep-12,ep-14,ep-17,ep-18,ep-23,ep-24,ep-25,ep-33
B=ep-03,ep-06,ep-07,ep-08,ep-20,ep-21,ep-22,ep-26,ep-28,ep-30,ep-32
C=ep-05,ep-09,ep-10,ep-11,ep-13,ep-15,ep-16,ep-19,ep-27,ep-29,ep-31
PY=../unlearning/.venv/bin/python
if [ "$1" = A ]; then
  while [ ! -f STOP_LOOP ]; do CUDA_VISIBLE_DEVICES=1 $PY score_v2.py --answers ep_p2.jsonl --jobs jobs_p2.jsonl --model Qwen/Qwen2.5-3B-Instruct --fp8 --quants nf4,int4 --endpoints $A 2>&1 | grep -v Loading; sleep 30; done
else
  while [ ! -f STOP_LOOP ]; do
    CUDA_VISIBLE_DEVICES=0 $PY score_v2.py --answers ep_p2.jsonl --jobs jobs_p2.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --fp8 --quants nf4,int8,int4 --endpoints $B 2>&1 | grep -v Loading
    CUDA_VISIBLE_DEVICES=0 $PY score_v2.py --answers ep_p2.jsonl --jobs jobs_p2.jsonl --model unsloth/Llama-3.2-1B-Instruct --fp8 --quants nf4,int8,int4 --endpoints $C 2>&1 | grep -v Loading
    sleep 30
  done
fi
