"""Read-only deterministic reasoning evaluation of the fixed 500 QA diagnostic set."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from verl.utils.dataset import process_video

from reasoning_format import parse_answer, parse_regions, reasoning_prompt

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
MODELS = {
    "base": Path("/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct"),
    "teacher": Path("/home/xianhao/models/Qwen3-VL-32B-Instruct"),
    **{f"step{step}": ROOT / f"outputs/video_opd_1k_reasoning/global_step_{step}/actor/huggingface"
       for step in (16, 31, 47, 62)},
}


def make_input(row, tokenizer, condition):
    body = reasoning_prompt(row).split("<video>\n", 1)[1]
    if condition == "video":
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "\n" + body}]}]
    else:
        messages = [{"role": "user", "content": body}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    item = {"prompt_token_ids": prompt_ids}
    visual_tokens = 0
    grid = None
    if condition == "video":
        video, metadata = process_video(row["frame_paths"], 4096, 147456, 2.0, return_metadata=True)
        assert video.shape[0] == 32
        item["multi_modal_data"] = {"video": [(video, metadata)]}
        visual_tokens = int(video.shape[-2] * video.shape[-1] // 64)
        grid = [16, int(video.shape[-2] // 16), int(video.shape[-1] // 16)]
    return item, visual_tokens, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--condition", choices=("video", "text"), required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    os.environ["EASYR1_VIDEO_OPD_PREPROCESS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rows = [json.loads(line) for line in (BASE / "qualification_500.jsonl").read_text().splitlines()]
    assert len(rows) == 500 and 0 <= args.start < args.end <= 500
    model_path = MODELS[args.model]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    engine = LLM(
        model=str(model_path), dtype="bfloat16", tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.72, max_model_len=4608,
        max_num_batched_tokens=4608, enforce_eager=True,
        disable_custom_all_reduce=True, enable_chunked_prefill=True,
        mm_processor_cache_gb=0, mm_processor_kwargs={"do_resize": False},
        trust_remote_code=False, disable_log_stats=True,
    )
    # Greedy decoding; same response cap and prompt as the reasoning rollout.
    params = SamplingParams(temperature=0, max_tokens=1024, detokenize=False,
                            logit_bias={processor.video_token_id: -100})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["sample_id"] for line in args.output.read_text().splitlines() if line}
    with args.output.open("a") as f:
        for lo in range(args.start, args.end, args.batch):
            selected = [row for row in rows[lo:min(lo + args.batch, args.end)] if row["sample_id"] not in done]
            if not selected:
                continue
            prepared = [make_input(row, tokenizer, args.condition) for row in selected]
            tick = time.monotonic()
            outputs = engine.generate([x[0] for x in prepared], params, use_tqdm=False)
            batch_sec = time.monotonic() - tick
            for row, (_, visual_tokens, grid), result in zip(selected, prepared, outputs):
                token_ids = list(result.outputs[0].token_ids)
                response = tokenizer.decode(token_ids, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                allowed = [chr(ord("A") + i) for i in range(len(row["options"]))]
                parsed = parse_answer(response, allowed)
                region = parse_regions(response, allowed)
                # Token region counts match the established qualification classifier.
                from verl.trainer.reasoning_diagnostics import analyze_response
                detail = analyze_response(tokenizer, token_ids)
                item = {
                    "sample_id": row["sample_id"], "original_video_id": row["original_video_id"],
                    "type": row["type"], "ground_truth": row["answer"],
                    "model": args.model, "condition": args.condition,
                    "response": response, "response_token_ids": token_ids,
                    "response_tokens": len(token_ids),
                    "reasoning_tokens": detail["reasoning_tokens"],
                    "answer_tokens": detail["answer_tokens"],
                    "format_valid": region is not None, "parsed_answer": parsed,
                    "correct": parsed == row["answer"],
                    "truncated": len(token_ids) >= 1024,
                    "frame_count": 32 if args.condition == "video" else 0,
                    "visual_tokens": visual_tokens, "video_grid_thw": grid,
                    "decode_seconds_per_batch": batch_sec,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                done.add(row["sample_id"])
            print(f"{args.model}/{args.condition} {len(done)} samples, last batch {batch_sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
