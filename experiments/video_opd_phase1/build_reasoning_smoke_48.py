"""Freeze a small, outcome-blind High-video train subset for three OPD updates."""

import json
import random
from pathlib import Path

from build_reasoning_qualification_128 import expected_visual_tokens, write
from reasoning_format import reasoning_prompt

BASE = Path(__file__).resolve().parent
SEED = 20260928


def main():
    pool = [json.loads(line) for line in (BASE / "pilot_1k_natural.jsonl").read_text().splitlines()]
    eligible = [row for row in pool if expected_visual_tokens(row) >= 2288]
    selected = random.Random(SEED).sample(eligible, 48)
    qualification = [json.loads(line) for line in
                     (BASE / "reasoning_qualification_128.jsonl").read_text().splitlines()]
    assert not ({row["original_video_id"] for row in selected} &
                {row["original_video_id"] for row in qualification})
    assert all(len(row["frame_paths"]) == 32 for row in selected)
    prepared = [{
        "prompt": reasoning_prompt(row), "answer": row["answer"],
        "videos": [row["frame_paths"]], "sample_id": row["sample_id"],
        "video_id": row["video_id"], "original_video_id": row["original_video_id"],
        "type": row["type"],
    } for row in selected]
    write(BASE / "reasoning_smoke_48.jsonl", selected)
    write(BASE / "reasoning_smoke_48_easyr1.jsonl", prepared)
    print(json.dumps({"seed": SEED, "n": len(selected), "eligible_in_pilot": len(eligible),
                      "qualification_original_video_overlap": 0, "model_outcome_selection": False}))


if __name__ == "__main__":
    main()
