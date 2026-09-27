"""Reuse the frozen Natural 1K rows with the qualified reasoning prompt."""

import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoProcessor

from build_reasoning_qualification_128 import expected_visual_tokens, write
from reasoning_format import reasoning_prompt

BASE = Path(__file__).resolve().parent
STUDENT = "/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct"
PROBE_SEED = 20260927


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    train = read(BASE / "pilot_1k_natural.jsonl")
    qualification = read(BASE / "qualification_500.jsonl")
    reasoning_qualification = read(BASE / "reasoning_qualification_128.jsonl")
    assert len(train) == 1000 and len({r["sample_id"] for r in train}) == 1000
    train_videos = {r["original_video_id"] for r in train}
    assert not (train_videos & {r["original_video_id"] for r in qualification})
    assert not (train_videos & {r["original_video_id"] for r in reasoning_qualification})
    assert all(len(r["frame_paths"]) == 32 for r in train)
    visual = Counter(expected_visual_tokens(row) for row in train)
    assert set(visual) <= {2288, 2304}, visual

    prepared = [{
        "prompt": reasoning_prompt(row), "answer": row["answer"],
        "videos": [row["frame_paths"]], "sample_id": row["sample_id"],
        "video_id": row["video_id"], "original_video_id": row["original_video_id"],
        "type": row["type"], "question": row["question"], "options": row["options"],
    } for row in train]
    write(BASE / "pilot_1k_reasoning_easyr1.jsonl", prepared)

    probe = random.Random(PROBE_SEED).sample(reasoning_qualification, 32)
    assert len({row["original_video_id"] for row in probe}) == 32
    write(BASE / "reasoning_probe_32.jsonl", probe)
    write(BASE / "reasoning_probe_32_easyr1.jsonl", [{
        "prompt": reasoning_prompt(row), "answer": row["answer"],
        "videos": [row["frame_paths"]], "sample_id": row["sample_id"],
        "video_id": row["video_id"], "original_video_id": row["original_video_id"],
        "type": row["type"], "question": row["question"], "options": row["options"],
    } for row in probe])

    processor = AutoProcessor.from_pretrained(STUDENT, local_files_only=True)
    text_token_counts = []
    for row in prepared:
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": row["prompt"].split("<video>\n", 1)[1]},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        text_token_counts.append(len(processor.tokenizer.encode(prompt, add_special_tokens=False)))
    # The production Qwen processor adds video frame/timestamp tokens to this
    # text-only count; the 32-sample full processor preflight checks that margin.
    report = {
        "source_manifest": str(BASE / "pilot_1k_natural.jsonl"),
        "samples": 1000, "unique_original_videos": len(train_videos),
        "qualification_500_original_video_overlap": 0,
        "reasoning_qualification_128_original_video_overlap": 0,
        "source_visual_token_counts": dict(visual),
        "max_text_template_tokens_before_video_expansion": max(text_token_counts),
        "max_response_length": 1024,
        "global_batch": 16, "drop_last": True, "actually_consumed": 992,
        "optimizer_steps": 62, "checkpoint_steps": [16, 31, 47, 62],
        "dropped_tail_sample_ids_with_sequential_sampler": [row["sample_id"] for row in train[-8:]],
        "probe_seed": PROBE_SEED, "probe_size": 32,
    }
    (BASE / "pilot_1k_reasoning_prelaunch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
