#!/usr/bin/env bash
set -Eeuo pipefail
source /home/xianhao/venvs/easyr1-ppu/bin/activate-easyr1
cd /home/xianhao/projects/EasyR1
export EASYR1_VIDEO_OPD_PREPROCESS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
base="experiments/video_opd_phase1"
python "$base/eval_reasoning_behavior.py" --model teacher --condition text \
  --start 0 --end 250 --batch 16 --tp 4 \
  --output "$base/opd_eval_runs/behavior_teacher_text_part1.jsonl"
