"""Sample a reproducible natural 1K from the frozen video-disjoint pool."""

import json
import random
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEED = 20260926
TARGET = 1000
MAX_PER_ORIGINAL = 8


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    pool = read(BASE / "train_pool_video_disjoint.jsonl")
    qualification = read(BASE / "qualification_500.jsonl")
    qualification_videos = {row["original_video_id"] for row in qualification}
    rng = random.Random(SEED)
    order = list(range(len(pool)))
    rng.shuffle(order)
    selected = []
    video_counts = Counter()
    for index in order:
        row = pool[index]
        original = row["original_video_id"]
        if video_counts[original] >= MAX_PER_ORIGINAL:
            continue
        selected.append(row)
        video_counts[original] += 1
        if len(selected) == TARGET:
            break
    if len(selected) != TARGET:
        raise RuntimeError("Pool cannot satisfy 1K with the per-video cap")
    overlap = set(video_counts) & qualification_videos
    if overlap:
        raise RuntimeError(f"Qualification original-video overlap: {sorted(overlap)}")
    assert len({row["sample_id"] for row in selected}) == TARGET
    assert all(len(row["frame_paths"]) == 32 for row in selected)

    prepared = []
    for row in selected:
        letters = ", ".join(chr(ord("A") + i) for i in range(len(row["options"])))
        prepared.append({
            "prompt": "<video>\n" + row["question"] + "\n" + "\n".join(row["options"])
            + "\nAnswer with exactly one option letter: " + letters + ".",
            "answer": row["answer"],
            "videos": [row["frame_paths"]],
            "sample_id": row["sample_id"],
            "video_id": row["video_id"],
            "original_video_id": row["original_video_id"],
            "type": row["type"],
        })

    pool_types = Counter(row["type"] for row in pool)
    selected_types = Counter(row["type"] for row in selected)
    report = {
        "seed": SEED,
        "sampling": "uniform random row order, accept while original-video count < 8",
        "source_pool": str(BASE / "train_pool_video_disjoint.jsonl"),
        "pool_qa": len(pool),
        "pilot_qa": len(selected),
        "unique_original_videos": len(video_counts),
        "question_type_counts": dict(sorted(selected_types.items())),
        "pool_question_type_counts": dict(sorted(pool_types.items())),
        "question_type_total_variation_distance": 0.5 * sum(
            abs(selected_types[t] / TARGET - pool_types[t] / len(pool)) for t in pool_types
        ),
        "qa_per_original_video_distribution": dict(sorted(Counter(video_counts.values()).items())),
        "max_qa_per_original_video": max(video_counts.values()),
        "qualification_500_original_video_overlap": len(overlap),
        "qualification_500_overlap_ids": sorted(overlap),
        "frame_count_per_qa": 32,
        "uses_model_outcomes_or_key_frame_annotations": False,
    }
    write(BASE / "pilot_1k_natural.jsonl", selected)
    write(BASE / "pilot_1k_natural_easyr1.jsonl", prepared)
    (BASE / "pilot_1k_natural_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
