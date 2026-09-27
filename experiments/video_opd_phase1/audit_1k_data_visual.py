"""Read-only reconstruction of the completed run's sampler and video inputs."""

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from verl.utils.dataset import process_video

BASE = Path(__file__).resolve().parent
ROOT = Path("/home/xianhao/projects/EasyR1/outputs/video_opd_1k_natural")
os.environ["EASYR1_VIDEO_OPD_PREPROCESS"] = "1"


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text())
    rows = [json.loads(line) for line in (BASE / "pilot_1k_natural_easyr1.jsonl").read_text().splitlines()]
    assert len(rows) == 1000 and config["data"]["seed"] == 20260926
    generator = torch.Generator().manual_seed(config["data"]["seed"])
    order = torch.randperm(len(rows), generator=generator).tolist()
    batch_size = config["data"]["rollout_batch_size"]
    consumed = (len(rows) // batch_size) * batch_size
    consumed_order = order[:consumed]
    consumed_rows = [rows[index] for index in consumed_order]
    consumed_ids = [row["sample_id"] for row in consumed_rows]
    assert len(consumed_ids) == len(set(consumed_ids)) == 992
    snapshots = {}
    for step in (8, 16, 24, 31):
        state = torch.load(ROOT / f"global_step_{step}" / "dataloader.pt",
                           map_location="cpu", weights_only=False)
        main = state["_snapshot"]["_main_snapshot"]
        snapshots[step] = {
            "snapshot_step": state["_snapshot"]["_snapshot_step"],
            "samples_yielded": main["_sampler_iter_state"]["samples_yielded"],
            "batches_yielded": main["_sampler_iter_yielded"],
        }
        assert snapshots[step] == {"snapshot_step": step,
                                   "samples_yielded": step * batch_size,
                                   "batches_yielded": step}

    # Same fixed seed and actual RandomSampler order; sample without replacement.
    picked = random.Random(20260927).sample(consumed_rows, 16)
    processor = AutoProcessor.from_pretrained(config["worker"]["actor"]["model"]["model_path"],
                                              local_files_only=True)
    visual = []
    seen_videos = set()
    video_fingerprints = []
    for row in picked:
        frames = row["videos"][0]
        assert len(frames) == 32
        with Image.open(frames[0]) as image:
            source_wh = image.size
        (video, metadata), fps = process_video(
            frames, config["data"]["min_pixels"], config["data"]["max_pixels"],
            config["data"]["video_fps"], return_fps=True, return_metadata=True,
        )
        question_text = row["prompt"].split("<video>\n", 1)[1]
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": question_text}
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(videos=[video], text=[prompt], video_metadata=[metadata],
                           add_special_tokens=False, do_resize=False, return_tensors="pt")
        grid = inputs["video_grid_thw"][0].tolist()
        tokens = int((inputs["input_ids"] == processor.video_token_id).sum())
        record = {
            "sample_id": row["sample_id"], "original_video_id": row["original_video_id"],
            "source_path": frames[0], "source_resolution_wh": list(source_wh),
            "frame_count": len(frames), "processed_resolution_hw": list(video.shape[-2:]),
            "video_grid_thw": grid, "visual_tokens": tokens,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        }
        assert len(frames) == 32 and tokens >= 1000
        assert record["prompt_tokens"] <= config["data"]["max_prompt_length"]
        assert list(video.shape[-2:]) == [grid[1] * 16, grid[2] * 16]
        visual.append(record)
        if row["original_video_id"] not in seen_videos and len(video_fingerprints) < 10:
            seen_videos.add(row["original_video_id"])
            video_fingerprints.append({
                "original_video_id": row["original_video_id"],
                "source_path": frames[0],
                "pixel_tensor_sha256": hashlib.sha256(video.numpy().tobytes()).hexdigest(),
                "pixel_mean": float(video.float().mean()),
                "pixel_std": float(video.float().std()),
            })
    assert len(video_fingerprints) == 10
    assert len({item["pixel_tensor_sha256"] for item in video_fingerprints}) == 10
    values = sorted(item["visual_tokens"] for item in visual)
    result = {
        "sampler_reconstruction": "PyTorch RandomSampler, manual_seed(20260926), first 992 of randperm(1000)",
        "snapshot_verification": snapshots,
        "manifest_samples": len(rows), "consumed_samples": consumed,
        "unique_consumed_sample_ids": len(set(consumed_ids)),
        "duplicate_sample_ids": [item for item, count in Counter(consumed_ids).items() if count > 1],
        "missing_sample_ids": [rows[index]["sample_id"] for index in order[consumed:]],
        "first_20_sample_ids_in_loader_order": consumed_ids[:20],
        "last_20_sample_ids_in_loader_order": consumed_ids[-20:],
        "visual_sample_size": len(visual),
        "visual_tokens_min": values[0], "visual_tokens_median": (values[7] + values[8]) / 2,
        "visual_tokens_p95_nearest_rank": values[15], "visual_tokens_max": values[-1],
        "visual_checks": visual, "distinct_video_fingerprints": video_fingerprints,
    }
    (BASE / "training_integrity_data_visual_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("visual_checks", "distinct_video_fingerprints")}, indent=2))


if __name__ == "__main__":
    main()
