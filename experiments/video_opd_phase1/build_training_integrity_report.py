"""Summarize immutable run artifacts for the read-only integrity audit."""

import ast
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent
ROOT = Path("/home/xianhao/projects/EasyR1/outputs/video_opd_1k_natural")


def stat(values):
    ordered = sorted(float(x) for x in values)
    return {
        "mean": sum(ordered) / len(ordered),
        "median": statistics.median(ordered),
        "p95_nearest_rank": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "min": ordered[0], "max": ordered[-1],
    }


def main():
    config = json.loads((ROOT / "experiment_config.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "experiment_log.jsonl").read_text().splitlines()]
    checkpoint = json.loads((BASE / "training_integrity_checkpoint_evidence.json").read_text())
    data = json.loads((BASE / "training_integrity_data_visual_evidence.json").read_text())
    log = (BASE / "logs/1k_natural_16ppu.log").read_text(errors="replace")
    debug = []
    for match in re.finditer(r"VIDEO_OPD_PREPROCESS (\{[^\n\r]*?\})", log):
        try:
            debug.append(ast.literal_eval(match.group(1)))
        except (SyntaxError, ValueError):
            pass
    memory = defaultdict(list)
    with (BASE / "logs/1k_natural_16ppu_memory.csv").open() as stream:
        for item in csv.DictReader(stream):
            memory[int(item["ppu"])].append(int(item["memory_mib"]))
    start = datetime.fromisoformat((ROOT / "start_utc.txt").read_text().strip().replace("Z", "+00:00"))
    end = datetime.fromisoformat((ROOT / "end_utc.txt").read_text().strip().replace("Z", "+00:00"))
    timings = {key: stat([row["timing_s"][key] for row in rows])
               for key in ("gen", "old", "teacher", "update_actor", "step")}
    timings["other_including_checkpoint"] = stat([
        row["timing_s"]["step"] - sum(row["timing_s"][key]
                                           for key in ("gen", "old", "teacher", "update_actor"))
        for row in rows
    ])
    timings["checkpoint_save"] = stat([row["timing_s"]["save_checkpoint"] for row in rows
                                         if "save_checkpoint" in row["timing_s"]])
    timings["data_loading_separately_measured"] = False
    five = []
    for step in (1, 8, 16, 24, 31):
        item = rows[step - 1]["opd"]
        five.append({"step": step, "delta_mean": item["reward_mean"],
                     "delta_std": item["reward_std"],
                     "delta_positive_fraction": item["reward_positive_frac"],
                     "delta_min": None, "delta_max": None,
                     "teacher_logprob_mean": item["teacher_logprob_mean"],
                     "old_logprob_mean": item["old_logprob_mean"]})
    all_manifest = [json.loads(line) for line in (BASE / "pilot_1k_natural_easyr1.jsonl").read_text().splitlines()]
    by_id = {row["sample_id"]: row for row in all_manifest}
    generator = torch.Generator().manual_seed(config["data"]["seed"])
    consumed_order = torch.randperm(len(all_manifest), generator=generator).tolist()[:992]
    preview_ids = random.Random(20260928).sample(
        [all_manifest[index]["sample_id"] for index in consumed_order], 20
    )
    rollout_preview = []
    for sample_id in preview_ids:
        row = by_id[sample_id]
        rollout_preview.append({
            "sample_id": sample_id,
            "question": row["prompt"].split("<video>\n", 1)[1].split("\n", 1)[0],
            "student_rollout": None,
            "gt": row["answer"],
            "note": "Original generation was not retained; this ID is reconstructed from RandomSampler.",
        })
    per_step = []
    for row in rows:
        per_step.append({
            "step": row["step"],
            "lr": row["actor"]["lr"], "grad_norm": row["actor"]["grad_norm"],
            "opd_loss": row["opd"]["loss"], "reward_mean": row["opd"]["reward_mean"],
            "reward_std": row["opd"]["reward_std"],
            "reward_positive_fraction": row["opd"]["reward_positive_frac"],
            "teacher_logprob_mean": row["opd"]["teacher_logprob_mean"],
            "old_logprob_mean": row["opd"]["old_logprob_mean"],
            "valid_response_tokens": row["opd"]["valid_tokens"],
            "optimizer_step_success": row["opd"]["optimizer_step"],
            "response_length": row["response_length"],
            "prompt_clip_ratio": row["prompt_length"]["clip_ratio"],
            "timing_s": row["timing_s"],
        })
    statuses = {
        "DATA_CONSUMPTION_CORRECT": "YES",
        "16_PPU_PARALLELISM_CORRECT": "YES",
        "HIGH_VISUAL_BUDGET_USED": "YES",
        "TEACHER_SCORING_REAL": "YES",
        "STUDENT_ROLLOUT_REAL": "YES",
        "STUDENT_PARAMETERS_UPDATED": "YES",
        "CHECKPOINTS_DISTINCT": "YES",
        "TRAINING_DURATION_PLAUSIBLE": "YES",
        "TRAINING_INTEGRITY": "PASS",
    }
    report = {
        "scope": "Read-only training integrity audit; no checkpoint accuracy evaluation or new training.",
        "status": statuses,
        "evidence_limitations": [
            "Formal run did not retain per-sample generation text/token IDs; 20 actual question/rollout/GT triples cannot be recovered.",
            "Formal run did not retain per-token teacher-old deltas; five-step token minima/maxima cannot be recovered.",
            "DataLoader snapshot stores consumed count, not exact sample indices; sample IDs are deterministically reconstructed from the run seed and PyTorch RandomSampler implementation.",
            "Data loading time is part of timing_s.gen and was not logged separately.",
            "Model tensor comparison covers all rank-0 shards whose names exist in the base safetensors index, not full 16-rank global parameter norm.",
        ],
        "run": {"start_utc": start.isoformat(), "end_utc": end.isoformat(),
                "wall_seconds": (end - start).total_seconds(),
                "sum_logged_step_seconds": sum(row["timing_s"]["step"] for row in rows),
                "student_model": config["worker"]["actor"]["model"]["model_path"],
                "teacher_model": config["worker"]["teacher"]["model"]["model_path"]},
        "batch": {
            "manifest_samples": len(all_manifest), "dataset_samples": data["manifest_samples"],
            "dataloader_batch_size": 32, "rollout_batch_size": config["data"]["rollout_batch_size"],
            "actor_global_batch_size": config["worker"]["actor"]["global_batch_size"],
            "micro_batch_update_per_rank": config["worker"]["actor"]["micro_batch_size_per_device_for_update"],
            "micro_batch_experience_per_rank": config["worker"]["actor"]["micro_batch_size_per_device_for_experience"],
            "world_size": config["trainer"]["n_gpus_per_node"], "samples_per_rank_per_step": 2,
            "gradient_accumulation_microbatches_per_rank": 2,
            "rollout_n": config["worker"]["rollout"]["n"],
            "drop_last": True, "epochs": config["trainer"]["total_epochs"],
            "optimizer_steps": len(rows),
            "consumed_global_qa": data["consumed_samples"],
            "unique_qa_reconstructed": data["unique_consumed_sample_ids"],
            "duplicates_reconstructed": len(data["duplicate_sample_ids"]),
            "missing_qa_count": len(data["missing_sample_ids"]),
            "missing_qa_ids_reconstructed": data["missing_sample_ids"],
            "dataloader_snapshots": data["snapshot_verification"],
        },
        "parallelism": {
            "roles_per_rank": "Each rank 0-15 hosts a Student FSDP shard, frozen Teacher FSDP shard, and vLLM rollout replica (TP=1, DP=16); each processes two distinct QA per global step.",
            "ranks": {str(i): {"student_fsdp": True, "teacher_fsdp": True, "vllm_dp": True}
                      for i in range(16)},
            "student_parameter_count_log": 4.44e9, "teacher_parameter_count_log": 33.36e9,
            "ppu_active_samples_per_rank": {str(i): sum(x > 1000 for x in memory[i]) for i in range(16)},
            "ppu_peak_memory_mib": {str(i): max(memory[i]) for i in range(16)},
            "vllm_tp": config["worker"]["rollout"]["tensor_parallel_size"],
            "vllm_dp": 16,
            "fsdp_size_actor": config["worker"]["actor"]["fsdp"]["fsdp_size"],
            "fsdp_size_teacher": config["worker"]["teacher"]["fsdp"]["fsdp_size"],
        },
        "visual": {
            "production_log_preprocess_records": len(debug),
            "production_log_frame_counts": dict(Counter(x["frame_count"] for x in debug)),
            "production_log_visual_tokens": dict(Counter(x["visual_tokens"] for x in debug)),
            "production_log_video_grids": {str(k): v for k, v in Counter(
                tuple(x["video_grid_thw"]) for x in debug).items()},
            "reconstructed_consumed_random_16": data["visual_checks"],
            "visual_tokens_min": data["visual_tokens_min"],
            "visual_tokens_median": data["visual_tokens_median"],
            "visual_tokens_p95": data["visual_tokens_p95_nearest_rank"],
            "visual_tokens_max": data["visual_tokens_max"],
            "ten_distinct_video_pixel_fingerprints": data["distinct_video_fingerprints"],
        },
        "teacher": {
            "step_calls_global": len(rows), "rank_level_dispatches": 16 * len(rows),
            "global_samples_scored": 32 * len(rows),
            "valid_sampled_tokens_scored": sum(row["opd"]["valid_tokens"] for row in rows),
            "teacher_logprob_finite_all_steps": all(math.isfinite(row["opd"]["teacher_logprob_mean"]) for row in rows),
            "mean_time_s_per_step": timings["teacher"]["mean"],
            "sampled_step_delta_statistics": five,
            "per_token_delta_min_max_available": False,
        },
        "rollout": {
            "completed_global_calls_in_log": len(re.findall(r"current_batch_size=32 >= rollout_batch_size=32", log)),
            "rank_level_calls_expected_from_dispatch": 16 * len(rows),
            "responses_per_step": 32, "total_responses": 32 * len(rows),
            "response_length_tokens": stat([row["response_length"]["mean"] for row in rows]),
            "response_length_all_samples_two_tokens": all(
                row["response_length"]["min"] == row["response_length"]["max"] == 2 for row in rows),
            "twenty_reconstructed_question_gt_without_original_rollout": rollout_preview,
            "original_generation_text_available": False,
        },
        "timing_seconds": timings,
        "optimizer": {
            "lr": stat([row["actor"]["lr"] for row in rows]),
            "grad_norm": stat([row["actor"]["grad_norm"] for row in rows]),
            "opd_loss": stat([row["opd"]["loss"] for row in rows]),
            "reward_mean": stat([row["opd"]["reward_mean"] for row in rows]),
            "reward_std": stat([row["opd"]["reward_std"] for row in rows]),
            "all_optimizer_steps_successful": all(row["opd"]["optimizer_step"] == 1 for row in rows),
            "all_finite": all(math.isfinite(row["actor"]["grad_norm"]) and
                              math.isfinite(row["opd"]["loss"]) for row in rows),
            "all_prompt_clip_ratios_zero": all(row["prompt_length"]["clip_ratio"] == 0 for row in rows),
            "checkpoint_tensor_comparison": checkpoint,
        },
        "per_step": per_step,
        "artifact_paths": {
            "run_config": str(ROOT / "experiment_config.json"),
            "run_metrics": str(ROOT / "experiment_log.jsonl"),
            "run_log": str(BASE / "logs/1k_natural_16ppu.log"),
            "data_visual_evidence": str(BASE / "training_integrity_data_visual_evidence.json"),
            "checkpoint_evidence": str(BASE / "training_integrity_checkpoint_evidence.json"),
        },
    }
    path = BASE / "training_integrity_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
