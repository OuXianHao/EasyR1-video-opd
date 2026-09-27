"""Check every reasoning qualification prompt at the production High setting."""

import json
import os
import random
import sys
from pathlib import Path

from transformers import AutoProcessor

from verl.utils.dataset import process_video

BASE = Path(__file__).resolve().parent
os.environ["EASYR1_VIDEO_OPD_PREPROCESS"] = "1"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "reasoning_qualification_128"
    rows = [json.loads(x) for x in (BASE / f"{name}_easyr1.jsonl").read_text().splitlines()]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if limit is not None:
        rows = random.Random(20260927).sample(rows, limit)
    processor = AutoProcessor.from_pretrained(
        "/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct", local_files_only=True
    )
    checks = []
    for i, row in enumerate(rows):
        video, metadata = process_video(row["videos"][0], 4096, 147456, 2.0, return_metadata=True)
        assert video.shape[0] == 32
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": row["prompt"].split("<video>\n", 1)[1]},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(videos=[video], video_metadata=[metadata], text=[prompt],
                           do_resize=False, add_special_tokens=False, return_tensors="pt")
        result = {
            "sample_id": row["sample_id"], "frame_count": 32,
            "video_grid_thw": inputs["video_grid_thw"][0].tolist(),
            "processed_resolution_hw": list(video.shape[-2:]),
            "visual_tokens": int((inputs["input_ids"] == processor.video_token_id).sum()),
            "input_tokens": int(inputs["input_ids"].shape[-1]),
        }
        assert result["visual_tokens"] >= 2288, result
        assert result["input_tokens"] <= 3584, result
        checks.append(result)
        if (i + 1) % 32 == 0:
            print(f"preflight {i + 1}/{len(rows)}", flush=True)
    suffix = f"_{limit}" if limit is not None else ""
    (BASE / f"{name}_high_preflight{suffix}.json").write_text(
        json.dumps(checks, indent=2) + "\n"
    )
    print("max_prompt_tokens", max(row["input_tokens"] for row in checks))
    print("visual_token_values", sorted({row["visual_tokens"] for row in checks}))


if __name__ == "__main__":
    main()
