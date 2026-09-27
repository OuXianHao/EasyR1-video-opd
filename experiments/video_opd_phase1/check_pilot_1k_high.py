"""Check eight frozen 32-frame samples through the production High path."""

import json
import os
import random
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor

from verl.utils.dataset import process_video

BASE = Path(__file__).resolve().parent
os.environ["EASYR1_VIDEO_OPD_PREPROCESS"] = "1"


def main():
    rows = [json.loads(line) for line in (BASE / "pilot_1k_natural_easyr1.jsonl").read_text().splitlines()]
    picked = random.Random(20260926 + 1).sample(rows, 8)
    processor = AutoProcessor.from_pretrained(
        "/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct", local_files_only=True
    )
    results = []
    for row in picked:
        frames = row["videos"][0]
        assert len(frames) == 32
        with Image.open(frames[0]) as image:
            source_wh = image.size
        (video, metadata), fps = process_video(
            frames, 4096, 147456, 2.0, return_fps=True, return_metadata=True
        )
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": row["prompt"].split("<video>\n", 1)[1]}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(
            videos=[video], text=[prompt], video_metadata=[metadata],
            add_special_tokens=False, do_resize=False, return_tensors="pt"
        )
        grid = inputs["video_grid_thw"][0].tolist()
        tokens = int((inputs["input_ids"] == processor.video_token_id).sum())
        input_tokens = int(inputs["input_ids"].shape[-1])
        result = {
            "sample_id": row["sample_id"],
            "source_resolution_wh": source_wh,
            "processed_resolution_hw": list(video.shape[-2:]),
            "frame_count": int(video.shape[0]),
            "video_grid_thw": grid,
            "visual_tokens": tokens,
            "input_tokens": input_tokens,
            "fps": fps,
        }
        assert result["frame_count"] == 32
        assert grid[0] == 16 and tokens >= 1000
        assert input_tokens <= 3072
        assert list(video.shape[-2:]) == [grid[1] * 16, grid[2] * 16]
        results.append(result)
        print(json.dumps(result), flush=True)
    (BASE / "pilot_1k_natural_high_preflight.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
