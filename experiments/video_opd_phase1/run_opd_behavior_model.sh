#!/usr/bin/env bash
set -Eeuo pipefail
source /home/xianhao/venvs/easyr1-ppu/bin/activate-easyr1
cd /home/xianhao/projects/EasyR1
export EASYR1_VIDEO_OPD_PREPROCESS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
model="$1"
tp="${2:-1}"
base="experiments/video_opd_phase1"
mkdir -p "$base/opd_eval_runs"
for condition in video text; do
  if [[ "$model" == teacher ]]; then
    if [[ "$condition" == video ]]; then batch=4; else batch=8; fi
  else
    if [[ "$condition" == video ]]; then batch=8; else batch=16; fi
  fi
  python "$base/eval_reasoning_behavior.py" \
    --model "$model" --condition "$condition" --batch "$batch" --tp "$tp" \
    --output "$base/opd_eval_runs/behavior_${model}_${condition}.jsonl"
done
