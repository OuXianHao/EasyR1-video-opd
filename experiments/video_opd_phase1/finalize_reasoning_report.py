"""Build a reproducible qualification + three-step smoke report from run artifacts."""

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from analyze_reasoning_qualification import classify, partial_regions, token_spans
from reasoning_format import parse_answer, parse_regions

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def basic(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def field(row, path):
    section, name = path.split("/", 1)
    return row[section][name]


def memory(path):
    by_ppu = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            ppu = int(row["ppu"])
            by_ppu[ppu] = max(by_ppu.get(ppu, 0), int(row["memory_mib"]))
    return {"peak_mib_by_ppu": by_ppu, "peak_mib": max(by_ppu.values()),
            "ppus_over_10_gib": sorted(p for p, mib in by_ppu.items() if mib > 10240)}


def table_extremes(items):
    result = ["| sample | token | region | Δ | context |", "|---|---|---|---:|---|"]
    for x in items:
        token = x["token_text"].replace("|", "\\|").replace("\n", "\\n")
        context = x["context"].replace("|", "\\|").replace("\n", "\\n")[:140]
        result.append(f"| {x['sample_id']} | `{token}` | {x['region']} | {x['delta']:.3f} | {context} |")
    return "\n".join(result)


def main():
    q = json.loads((BASE / "reasoning_qualification_diagnostics.json").read_text())
    split = json.loads((BASE / "reasoning_qualification_128_split.json").read_text())
    q_preflight = json.loads((BASE / "reasoning_qualification_128_high_preflight.json").read_text())
    smoke_preflight = json.loads((BASE / "reasoning_smoke_48_high_preflight.json").read_text())
    q_log = read_jsonl(ROOT / "outputs/video_opd_reasoning_qualification_128/experiment_log.jsonl")
    s_log = read_jsonl(ROOT / "outputs/video_opd_reasoning_smoke_16ppu/experiment_log.jsonl")
    q_run = Path((BASE / "latest_reasoning_qual_run.txt").read_text().strip())
    s_run = Path((BASE / "latest_reasoning_smoke_run.txt").read_text().strip())
    smoke_files = sorted((s_run / "diagnostics").glob("step_*.json"),
                         key=lambda p: int(p.stem.split("_")[-1]))
    assert [x["step"] for x in q_log] == list(range(1, 9))
    assert [x["step"] for x in s_log] == [1, 2, 3]
    assert len(smoke_files) == 3
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct", local_files_only=True)
    smoke_source = {x["sample_id"]: x for x in read_jsonl(BASE / "reasoning_smoke_48.jsonl")}
    smoke_samples = []
    loss_coverage = []
    for file, metric in zip(smoke_files, s_log):
        d = json.loads(file.read_text())
        assert d["step"] == metric["step"] and len(d["samples"]) == 16
        valid_count = 0
        reasoning_count = 0
        for item in d["samples"]:
            ids = [x["token_id"] for x in item["tokens"]]
            valid_count += len(ids)
            generated = list(ids)
            while generated and generated[-1] in tokenizer.all_special_ids:
                generated.pop()
            response, spans = token_spans(tokenizer, generated)
            source = smoke_source[item["sample_id"]]
            letters = [chr(ord("A") + i) for i in range(len(source["options"]))]
            strict_regions = parse_regions(response, letters)
            regions = strict_regions or partial_regions(response)
            labels = [classify(span, regions) for span in spans]
            reasoning_count += labels.count("reasoning")
            smoke_samples.append({"step": d["step"], "sample_id": item["sample_id"],
                                  "response_tokens": len(ids),
                                  "reasoning_tokens": labels.count("reasoning"),
                                  "hit_1024": len(ids) >= 1024,
                                  "format_valid": strict_regions is not None,
                                  "answer_parse_valid": parse_answer(response, letters) is not None,
                                  "student_response": response})
        assert valid_count == int(field(metric, "opd/valid_tokens")), (valid_count, field(metric, "opd/valid_tokens"))
        loss_coverage.append({"step": d["step"], "valid_response_tokens": valid_count,
                              "opd_valid_tokens": field(metric, "opd/valid_tokens"),
                              "reasoning_tokens_with_response_mask_1": reasoning_count})
    assert len(smoke_samples) == 48 and len({x["sample_id"] for x in smoke_samples}) == 48
    assert all(x["reasoning_tokens"] > 0 for x in smoke_samples)
    assert all(math.isfinite(float(field(metric, "opd/loss"))) and
               math.isfinite(float(field(metric, "actor/grad_norm"))) and
               field(metric, "actor/grad_norm") > 0 and field(metric, "opd/optimizer_step") == 1
               for metric in s_log)
    q_timing = {name: basic([x["timing_s"][name] for x in q_log])
                for name in ("gen", "old", "teacher", "step")}
    s_timing = {name: basic([x["timing_s"][name] for x in s_log])
                for name in ("gen", "old", "teacher", "update_actor", "step")}
    visual_q = Counter(x["visual_tokens"] for x in q_preflight)
    visual_s = Counter(x["visual_tokens"] for x in smoke_preflight)
    smoke_memory = memory(s_run / "ppu_memory.csv")
    smoke_runtime = (BASE / "logs/reasoning_smoke_16ppu.log").read_text(errors="replace")
    runtime_visuals = Counter(map(int, re.findall(r"'visual_tokens': (\d+)", smoke_runtime)))
    assert set(runtime_visuals) <= {2288, 2304} and sum(runtime_visuals.values()) >= 48
    assert smoke_memory["ppus_over_10_gib"] == list(range(16))
    assert "contains 33.36B parameters" in smoke_runtime
    q_delta = q["region_delta"]
    result = {
        "qualification": {
            "split": split, "response": {key: q[key] for key in (
                "n", "total_response_tokens", "reasoning_content_tokens",
                "two_token_response_fraction", "empty_reasoning_fraction",
                "truncation_1024_fraction", "format_valid_rate", "answer_parse_valid_rate",
                "tag_rates")},
            "high_visual_tokens": dict(visual_q),
            "max_prompt_tokens": max(x["input_tokens"] for x in q_preflight),
            "teacher_forcing": {"calls": 8, "total_samples": 128,
                                "region_delta": q_delta,
                                "answer_tokens_mixed_with_format": q["answer_tokens_mixed_with_format"],
                                "note": q["answer_delta_note"]},
            "timing_seconds": q_timing,
            "memory": memory(q_run / "ppu_memory.csv"),
            "top_positive_delta_tokens": q["top_positive_delta_tokens"],
            "top_negative_delta_tokens": q["top_negative_delta_tokens"],
            "full_rollouts_path": str(BASE / "reasoning_qualification_rollouts_128.jsonl"),
            "examples_30_path": str(BASE / "reasoning_qualification_samples_30.md"),
        },
        "smoke": {
            "run_dir": str(s_run), "steps_completed": 3, "ppus": list(range(16)),
            "global_batch": 16, "max_response_length": 1024,
            "response_tokens": basic([x["response_tokens"] for x in smoke_samples]),
            "reasoning_tokens": basic([x["reasoning_tokens"] for x in smoke_samples]),
            "truncation_1024_count": sum(x["hit_1024"] for x in smoke_samples),
            "truncation_1024_fraction": sum(x["hit_1024"] for x in smoke_samples) / 48,
            "format_valid_rate": sum(x["format_valid"] for x in smoke_samples) / 48,
            "answer_parse_valid_rate": sum(x["answer_parse_valid"] for x in smoke_samples) / 48,
            "truncated_sample_ids": [x["sample_id"] for x in smoke_samples if x["hit_1024"]],
            "high_visual_tokens_preflight": dict(visual_s),
            "high_visual_tokens_runtime": dict(runtime_visuals),
            "max_prompt_tokens": max(x["input_tokens"] for x in smoke_preflight),
            "timing_seconds": s_timing, "memory": smoke_memory,
            "steps": [{"step": x["step"], "loss": field(x, "opd/loss"),
                       "reward_mean": field(x, "opd/reward_mean"), "reward_std": field(x, "opd/reward_std"),
                       "grad_norm": field(x, "actor/grad_norm"),
                       "optimizer_step": field(x, "opd/optimizer_step"),
                       "valid_tokens": field(x, "opd/valid_tokens")} for x in s_log],
            "loss_coverage": loss_coverage,
            "log_path": str(BASE / "logs/reasoning_smoke_16ppu.log"),
            "output_path": str(ROOT / "outputs/video_opd_reasoning_smoke_16ppu"),
        },
        "decisions": {
            "REASONING_OUTPUT_VALID": "YES",
            "TEACHER_REASONING_COMPATIBLE": "YES",
            "FORMAT_TOKEN_MASK_RECOMMENDED": "NO",
            "REASONING_TOKENS_INCLUDED_IN_OPD_LOSS": "YES",
            "READY_FOR_1K_REASONING_OPD": "YES",
        },
        "interpretation": [
            "Thirty full Student rollouts were manually read: they refer to video objects, actions, temporal and spatial relations. Some answers or details may be wrong; no reasoning accuracy judge was run.",
            "Two of 128 outputs omit </thinking>; one uses a plain Answer:D line. A bounded fallback parser recovers all 128 final options without changing OPD loss.",
            "Pure format-token Delta is near zero; the negative tail is mostly reasoning content or tokens jointly spanning content and a closing tag. No token mask was applied.",
            "The answer letter is usually joined with the preceding > in one Qwen token. Its Delta is the joint sampled-token value and cannot be separated into an independent character probability.",
            "The teacher scores the Student's own rollout. No Teacher-generated reasoning was used. Qualification did not update Student parameters; the three smoke steps did.",
            "One of 48 smoke responses hit the 1024-token cap and omitted the answer; it repeatedly reconsidered timestamp options. The qualification 128 had no capped responses.",
            "Smoke gradient norms are pre-clipping values 57.5, 102.5 and 294.0; standard max_grad_norm is 1.0. The three finite updates establish startup feasibility, not long-run dynamics.",
        ],
    }
    (BASE / "reasoning_qualification_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    response = result["qualification"]["response"]
    smoke = result["smoke"]
    lines = [
        "# Reasoning Video OPD qualification and 16-PPU smoke",
        "",
        "## Data and prompt",
        "",
        f"Fixed seed {split['seed']}; 128 qualification QA on 128 unique original videos; training-pool video overlap 0. The High source-resolution criterion excluded {split['low_resolution_excluded_count']} of 500 candidates without consulting model outcomes. All 128 selected clips passed 32-frame production preprocessing with visual tokens {dict(visual_q)} and max prompt {result['qualification']['max_prompt_tokens']} tokens.",
        "",
        "The Student prompt requests concise video-based reasoning inside `<thinking>` followed by `<answer>LETTER</answer>`. The response cap is 1024 tokens; GT answers and key-frame annotations are absent from the prompt. The Student uses temperature 1, top-p 1, top-k -1, n=1. Teacher forcing scores these same Student samples.",
        "",
        "## Response statistics (N=128)",
        "",
        f"Total tokens: min {response['total_response_tokens']['min']:.0f}, P5 {response['total_response_tokens']['p5']:.2f}, P25 {response['total_response_tokens']['p25']:.2f}, median {response['total_response_tokens']['median']:.1f}, mean {response['total_response_tokens']['mean']:.2f}, P75 {response['total_response_tokens']['p75']:.2f}, P95 {response['total_response_tokens']['p95']:.2f}, max {response['total_response_tokens']['max']:.0f}.",
        "",
        f"Reasoning content tokens: min {response['reasoning_content_tokens']['min']:.0f}, P5 {response['reasoning_content_tokens']['p5']:.2f}, P25 {response['reasoning_content_tokens']['p25']:.2f}, median {response['reasoning_content_tokens']['median']:.1f}, mean {response['reasoning_content_tokens']['mean']:.2f}, P75 {response['reasoning_content_tokens']['p75']:.2f}, P95 {response['reasoning_content_tokens']['p95']:.2f}, max {response['reasoning_content_tokens']['max']:.0f}.",
        "",
        f"2-token response {response['two_token_response_fraction']:.2%}; empty reasoning {response['empty_reasoning_fraction']:.2%}; 1024-token truncation {response['truncation_1024_fraction']:.2%}; strict format valid {response['format_valid_rate']:.2%}; final answer parse valid {response['answer_parse_valid_rate']:.2%}.",
        "",
        f"[Thirty complete, manually readable examples]({(BASE / 'reasoning_qualification_samples_30.md').name}); all 128 responses are in `{(BASE / 'reasoning_qualification_rollouts_128.jsonl').name}`. The first 30 include people, objects, actions, temporal order, location, OCR and counts. They are authentic Student outputs; incorrect choices remain in the file.",
        "",
        "## Teacher forcing Δ = logP(32B) − logP(4B old)",
        "",
        f"Eight Teacher-scoring calls covered 128 Student rollouts. The 32B checkpoint path was loaded (runtime reports 33.36B parameters), and both scorer grids and pixels matched. Scoring time mean {q_timing['teacher']['mean']:.2f} s/step, median {q_timing['teacher']['median']:.2f} s/step.",
        "",
        "| Region | Tokens | Mean | Std | Positive | Negative | P1 | P5 | P25 | P50 | P75 | P95 | P99 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("reasoning", "answer", "format", "eos"):
        x = q_delta[name]
        lines.append(f"| {name} | {x['count']} | {x['mean']:.3f} | {x['std']:.3f} | {x['positive_fraction']:.1%} | {x['negative_fraction']:.1%} | {x['p1']:.3f} | {x['p5']:.3f} | {x['p25']:.3f} | {x['median']:.3f} | {x['p75']:.3f} | {x['p95']:.3f} | {x['p99']:.3f} | {x['min']:.3f} | {x['max']:.3f} |")
    lines += [
        "",
        f"All {q['answer_tokens_mixed_with_format']} tagged answer tokens merge the preceding `>` and the answer letter, so their Δ is joint. The two malformed outputs contribute to partial reasoning diagnostics; one plain `Answer:D` ending has no separable tagged answer token. Pure format tokens have near-zero mean Δ. EOS is zero except one negative outlier. Reasoning median is near zero, with both positive and negative values; there is no broad extreme rejection of Student reasoning.",
        "",
        "### Top 30 positive Δ tokens",
        "",
        table_extremes(q["top_positive_delta_tokens"]),
        "",
        "### Top 30 negative Δ tokens",
        "",
        table_extremes(q["top_negative_delta_tokens"]),
        "",
        "The top 30 positive values are all reasoning tokens. The negative extremes are predominantly reasoning tokens, with one joint answer/format token and one malformed-output token. Some negative reasoning tokens straddle a closing tag or punctuation boundary. This does not justify masking all XML tags; a token-level tag mask could also suppress the joined answer letter.",
        "",
        "## Three-step 16-PPU OPD smoke",
        "",
        f"Separate 48-QA train subset, global batch 16, one generation per QA, micro batch 1 per device, three optimizer steps. 32-frame High preflight visual tokens {dict(visual_s)}; runtime High records {dict(runtime_visuals)}; max prompt {smoke['max_prompt_tokens']}. All 16 PPUs exceeded 10 GiB, with peak {smoke_memory['peak_mib']} MiB ({smoke_memory['peak_mib']/1024:.2f} GiB).",
        "",
        f"Smoke response tokens median {smoke['response_tokens']['median']:.1f}, mean {smoke['response_tokens']['mean']:.1f}, P95 {smoke['response_tokens']['p95']:.1f}, max {smoke['response_tokens']['max']:.0f}; reasoning median {smoke['reasoning_tokens']['median']:.1f}. {smoke['truncation_1024_count']}/48 hit the 1024-token cap; strict format valid {smoke['format_valid_rate']:.1%}, final answer parse valid {smoke['answer_parse_valid_rate']:.1%}. Teacher scoring mean {s_timing['teacher']['mean']:.2f} s, rollout mean {s_timing['gen']['mean']:.2f} s, actor update mean {s_timing['update_actor']['mean']:.2f} s, total step mean {s_timing['step']['mean']:.2f} s (includes cold first step).",
        "",
        f"The sole capped sample is {', '.join(smoke['truncated_sample_ids'])}. It repeatedly reconsidered conflicting timestamp options and ended before producing an answer. This is a 2.1% sampled long tail in the smoke, while the disjoint 128-sample qualification had zero capped outputs. It warrants monitoring in a later run, but is not a widespread truncation failure in this qualification.",
        "",
        "| Step | Loss | Reward mean | Reward std | Grad norm | Valid OPD tokens | Reasoning tokens in mask |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, coverage in zip(s_log, loss_coverage):
        lines.append(f"| {metric['step']} | {field(metric, 'opd/loss'):.5f} | {field(metric, 'opd/reward_mean'):.5f} | {field(metric, 'opd/reward_std'):.5f} | {field(metric, 'actor/grad_norm'):.5f} | {coverage['opd_valid_tokens']} | {coverage['reasoning_tokens_with_response_mask_1']} |")
    lines += [
        "",
        "Every diagnostic response token has response_mask=1; each step's diagnostic valid-token count exactly equals the OPD loss's valid-token count. Reasoning content tokens therefore participate in Teacher scoring and the unchanged Vanilla OPD loss. Optimizer steps and finite nonzero pre-clipping gradient norms were recorded for steps 1–3. Standard gradient-norm clipping is 1.0; no PPO ratio clipping, RL reward, extra weighting or format mask was applied. Raw gradient norms rose from 57.5 to 294 over three steps, so this smoke establishes finite updates and resource feasibility, not long-run dynamics.",
        "",
        f"The earlier answer-only pilot had ~2 response tokens, ~5.3 s Teacher scoring and ~12–13 s/step with global batch 32. This smoke uses global batch 16, so each of 16 ranks handles one rather than two QA per step. Despite longer responses, Teacher scoring averages {s_timing['teacher']['mean']:.2f} s because the ~2600-token multimodal prompt and per-rank batch size dominate. After the cold first step and one 1024-token runaway, steps 2–3 took {np.mean([x['timing_s']['step'] for x in s_log[1:]]):.2f} s on average. Both runs include an actor backward pass.",
        "",
        "## Decisions",
        "",
    ]
    lines.extend(f"- `{key} = {value}`" for key, value in result["decisions"].items())
    lines += [
        "",
        "The two malformed tags are a format limitation to track in a future 1K run; all final choices parse with the bounded fallback. No 1K reasoning training or accuracy evaluation was started.",
        "",
        f"Smoke log: `{smoke['log_path']}`. Smoke output: `{smoke['output_path']}`. Qualification log: `{BASE / 'logs/reasoning_qualification_128_16ppu.log'}`.",
        "",
    ]
    (BASE / "reasoning_qualification_report.md").write_text("\n".join(lines))
    print(json.dumps({"decisions": result["decisions"], "smoke": smoke["steps"],
                      "peak_mib": smoke_memory["peak_mib"]}, indent=2))


if __name__ == "__main__":
    main()
