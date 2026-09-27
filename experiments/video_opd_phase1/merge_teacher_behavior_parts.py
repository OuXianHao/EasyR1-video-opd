"""Assemble nonoverlapping Teacher evaluation shards in fixed 500-row order."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUN = BASE / "opd_eval_runs"


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    rows = load(BASE / "qualification_500.jsonl")
    ids = [r["sample_id"] for r in rows]
    assert len(ids) == len(set(ids)) == 500
    for condition in ("video", "text"):
        if condition == "video":
            raw_path = RUN / "behavior_teacher_video.jsonl"
            first = load(raw_path)
            # This shard was intentionally stopped after index 249; a few
            # already flushed rows beyond 249 are retained in the raw file.
            assert len(first) >= 250
            assert [r["sample_id"] for r in first[:250]] == ids[:250]
            raw_archive = RUN / "behavior_teacher_video_part1_raw.jsonl"
            if not raw_archive.exists():
                raw_path.rename(raw_archive)
            first = first[:250]
        else:
            first = load(RUN / "behavior_teacher_text_part1.jsonl")
            assert len(first) == 250 and [r["sample_id"] for r in first] == ids[:250]
        second = load(RUN / f"behavior_teacher_{condition}_part2.jsonl")
        assert len(second) == 250 and [r["sample_id"] for r in second] == ids[250:]
        merged = first + second
        assert len(merged) == len({r["sample_id"] for r in merged}) == 500
        (RUN / f"behavior_teacher_{condition}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
        print(condition, "merged", len(merged), "correct", sum(r["correct"] for r in merged))


if __name__ == "__main__":
    main()
