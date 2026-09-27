"""Freeze outcome-blind, video-disjoint reasoning qualification data."""

import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from reasoning_format import reasoning_prompt

BASE = Path(__file__).resolve().parent
SEED = 20260927
HIGH_MAX_PIXELS = 147456


def expected_visual_tokens(row):
    """Mirror the production 32-pixel canvas choice without decoding 32 frames."""
    with Image.open(row["frame_paths"][0]) as frame:
        source_w, source_h = frame.size
    candidates = []
    for target_h in range(32, source_h + 1, 32):
        for target_w in range(32, source_w + 1, 32):
            area = target_h * target_w
            if area > HIGH_MAX_PIXELS:
                continue
            scale = min(target_w / source_w, target_h / source_h, 1.0)
            content_w = max(1, round(source_w * scale))
            content_h = max(1, round(source_h * scale))
            candidates.append((area, -(area - content_w * content_h), target_h, target_w))
    if not candidates:
        return 0
    return max(candidates)[0] // 64


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    qualification = read(BASE / "qualification_500.jsonl")
    train_pool = read(BASE / "train_pool_video_disjoint.jsonl")
    assert len(qualification) == 500
    assert not ({row["original_video_id"] for row in qualification} &
                {row["original_video_id"] for row in train_pool})
    eligible = [row for row in qualification if expected_visual_tokens(row) >= 2288]
    selected = random.Random(SEED).sample(eligible, 128)
    assert len({row["original_video_id"] for row in selected}) == 128
    assert all(len(row["frame_paths"]) == 32 for row in selected)
    prepared = []
    for row in selected:
        prepared.append({
            "prompt": reasoning_prompt(row),
            "answer": row["answer"], "videos": [row["frame_paths"]],
            "sample_id": row["sample_id"], "video_id": row["video_id"],
            "original_video_id": row["original_video_id"], "type": row["type"],
        })
    write(BASE / "reasoning_qualification_128.jsonl", selected)
    write(BASE / "reasoning_qualification_128_easyr1.jsonl", prepared)
    report = {
        "seed": SEED, "sample_count": 128,
        "unique_original_video_count": 128,
        "train_pool_original_video_overlap": 0,
        "question_type_counts": dict(sorted(Counter(row["type"] for row in selected).items())),
        "selection_uses_model_outcomes": False,
        "selection_criterion": "Production High canvas yields at least 2288 visual tokens; no model outcomes used",
        "high_eligible_pool_count": len(eligible),
        "low_resolution_excluded_count": len(qualification) - len(eligible),
        "frame_count_per_sample": 32,
    }
    (BASE / "reasoning_qualification_128_split.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
