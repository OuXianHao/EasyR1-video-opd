#!/usr/bin/env bash
set -Eeuo pipefail

source /home/xianhao/venvs/easyr1-ppu/bin/activate-easyr1
cd /home/xianhao/projects/EasyR1
source experiments/video_opd_phase1/high.env

base="$PWD/experiments/video_opd_phase1"
output="$PWD/outputs/video_opd_1k_natural"
log="$base/logs/1k_natural_16ppu.log"
run_id="video-opd-1k-natural-16ppu-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ray_tmpdir="/tmp/eopd-1k-$$"
mkdir -p "$base/logs" "$output" "$ray_tmpdir"
exec > >(tee -a "$log") 2>&1

export EASYR1_SMOKE_RUN_ID="$run_id"
export EASYR1_VIDEO_OPD_PREPROCESS=1
export EASYR1_VIDEO_OPD_DEBUG=1
export EASYR1_OPD_TIMING=1
export RAY_TMPDIR="$ray_tmpdir"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
unset RAY_ADDRESS

python experiments/video_opd_phase1/monitor_memory.py "$base/logs/1k_natural_16ppu_memory.csv" &
monitor_pid=$!
cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  python /home/xianhao/datasets/easyr1_smoke/cleanup_owned.py "$run_id" || true
  date -u +%FT%TZ > "$output/end_utc.txt"
}
trap cleanup EXIT
date -u +%FT%TZ > "$output/start_utc.txt"

command=(
  python -m verl.trainer.main
  config=examples/config.yaml
  "data.train_files=$base/pilot_1k_natural_easyr1.jsonl"
  "data.val_files=$base/qualification_500_easyr1.jsonl"
  data.prompt_key=prompt data.answer_key=answer
  data.format_prompt=null data.shuffle=true data.seed=20260926
  data.filter_overlong_prompts=false
  data.rollout_batch_size=32 data.mini_rollout_batch_size=null data.val_batch_size=2
  data.max_prompt_length=3072 data.max_response_length=16
  "data.min_pixels=4096" "data.max_pixels=$VIDEO_OPD_MAX_PIXELS" data.video_fps=2.0
  algorithm.opd.enabled=true
  worker.actor.model.model_path=/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct
  worker.actor.global_batch_size=32
  worker.actor.micro_batch_size_per_device_for_update=1
  worker.actor.micro_batch_size_per_device_for_experience=1
  worker.actor.use_torch_compile=false
  worker.actor.fsdp.torch_dtype=bf16
  worker.actor.optim.strategy=adamw_bf16
  worker.actor.optim.lr=1.0e-6
  worker.teacher.model.model_path=/home/xianhao/models/Qwen3-VL-32B-Instruct
  worker.teacher.model.enable_gradient_checkpointing=false
  worker.teacher.fsdp.torch_dtype=bf16
  worker.teacher.offload.offload_params=true
  worker.rollout.n=1 worker.rollout.temperature=1.0
  worker.rollout.top_p=1.0 worker.rollout.top_k=-1
  worker.rollout.tensor_parallel_size=1
  worker.rollout.gpu_memory_utilization=0.22
  worker.rollout.enforce_eager=true
  worker.rollout.enable_chunked_prefill=true
  worker.rollout.max_num_batched_tokens=3200
  worker.rollout.max_model_len=3200
  trainer.n_gpus_per_node=16 trainer.nnodes=1 trainer.total_epochs=1 trainer.max_steps=null
  trainer.val_before_train=false trainer.val_freq=-1
  trainer.run_final_validation=false
  trainer.save_freq=8 trainer.save_limit=-1
  trainer.save_final_checkpoint=true trainer.find_last_checkpoint=false
  'trainer.logger=["console","file"]'
  "trainer.save_checkpoint_path=$output"
  trainer.experiment_name=video_opd_1k_natural_16ppu
)
printf '%q ' "${command[@]}" > "$output/command.txt"
printf '\n' >> "$output/command.txt"
printf '%s\n' "$VIDEO_OPD_BUDGET" > "$output/visual_budget.txt"
printf '%s\n' "$VIDEO_OPD_EXPECTED_VISUAL_TOKENS" > "$output/expected_visual_tokens.txt"
echo "RUN_ID=$run_id"
echo "MAIN_SHELL_PID=$$"
echo "LOG=$log"
echo "CHECKPOINT_PATH=$output"
"${command[@]}"
