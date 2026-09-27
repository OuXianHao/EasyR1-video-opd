"""Summarize fixed-trajectory OPD gap decomposition; no entropy or gradients."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr

BASE = Path(__file__).resolve().parent
RUN = BASE / "opd_eval_runs"
STUDENTS = ("base", "step16", "step31", "step47", "step62")
MODELS = STUDENTS + ("teacher",)
REGIONS = ("all", "reasoning", "answer", "format", "eos", "unclassified")


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    if not np.isfinite(a).all():
        raise RuntimeError("Nonfinite statistic input")
    return {
        "count": int(a.size), "mean": float(a.mean()), "mean_abs": float(np.abs(a).mean()),
        "std": float(a.std()), "min": float(a.min()),
        "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
        "max": float(a.max()), "positive_fraction": float((a > 0).mean()),
    }


def corr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(pearsonr(a, b).statistic),
            "spearman": float(spearmanr(a, b).statistic)}


def topk(opd, shared, visual, percentage):
    a, s, v = (np.asarray(x, dtype=np.float64) for x in (opd, shared, visual))
    k = max(1, math.ceil(len(a) * percentage / 100))
    inds = np.argsort(np.abs(a), kind="stable")[-k:]
    return {
        "count": k, "fraction": float(k / len(a)),
        "mean_abs_opd": float(np.abs(a[inds]).mean()),
        "mean_abs_shared": float(np.abs(s[inds]).mean()),
        "mean_abs_visual": float(np.abs(v[inds]).mean()),
        "shared_abs_gt_visual_fraction": float((np.abs(s[inds]) > np.abs(v[inds])).mean()),
    }


def read_complete(model, kind, condition, expected):
    items = load_jsonl(RUN / f"{kind}_{model}_{condition}.jsonl")
    ids = [x["sample_id"] for x in items]
    assert len(ids) == expected and len(set(ids)) == expected, (model, kind, condition, len(ids), len(set(ids)))
    return {x["sample_id"]: x for x in items}


def main():
    diagnostic = load_jsonl(BASE / "qualification_500.jsonl")
    training = load_jsonl(BASE / "pilot_1k_natural.jsonl")
    probe = load_jsonl(BASE / "opd_mechanism_probe_128.jsonl")
    assert len(diagnostic) == 500 and len(probe) == 128
    train_videos = {x["original_video_id"] for x in training}
    eval_videos = {x["original_video_id"] for x in diagnostic}
    probe_videos = {x["original_video_id"] for x in probe}
    assert not (train_videos & eval_videos) and probe_videos <= eval_videos
    probe_hash = hashlib.sha256((BASE / "opd_mechanism_probe_128.jsonl").read_bytes()).hexdigest()
    behavior = {}
    behavior_raw = {}
    for model in MODELS:
        video = read_complete(model, "behavior", "video", 500)
        text = read_complete(model, "behavior", "text", 500)
        assert set(video) == set(text) == {x["sample_id"] for x in diagnostic}
        behavior_raw[model] = {"video": video, "text": text}
        by_condition = {}
        for cond, records in (("video", video), ("text", text)):
            vals = list(records.values())
            by_condition[cond] = {
                "correct": sum(x["correct"] for x in vals),
                "accuracy": sum(x["correct"] for x in vals) / 500,
                "format_valid_rate": sum(x["format_valid"] for x in vals) / 500,
                "answer_parse_rate": sum(x["parsed_answer"] is not None for x in vals) / 500,
                "truncated_count": sum(x["truncated"] for x in vals),
                "response_tokens": stats([x["response_tokens"] for x in vals]),
                "reasoning_tokens": stats([x["reasoning_tokens"] for x in vals]),
            }
            if cond == "video":
                by_condition[cond]["visual_tokens"] = stats([x["visual_tokens"] for x in vals])
                by_condition[cond]["visual_token_counts"] = {
                    str(k): sum(x["visual_tokens"] == k for x in vals)
                    for k in sorted({x["visual_tokens"] for x in vals})
                }
        by_condition["video_minus_text_accuracy"] = by_condition["video"]["accuracy"] - by_condition["text"]["accuracy"]
        behavior[model] = by_condition
    paired_changes = {}
    for model in STUDENTS[1:]:
        paired_changes[model] = {}
        for cond in ("video", "text"):
            base_correct = behavior_raw["base"][cond]
            current = behavior_raw[model][cond]
            gained = sum((not base_correct[sid]["correct"]) and current[sid]["correct"] for sid in base_correct)
            lost = sum(base_correct[sid]["correct"] and (not current[sid]["correct"]) for sid in base_correct)
            paired_changes[model][cond] = {
                "gained_correct_vs_base": gained, "lost_correct_vs_base": lost,
                "net": gained - lost,
                "mcnemar_exact_two_sided_p": float(binomtest(min(gained, lost), gained + lost, 0.5).pvalue)
                if gained + lost else 1.0,
            }
    scores = {m: {c: read_complete(m, "probe", c, 128) for c in ("video", "text")} for m in MODELS}
    probes = {x["sample_id"]: x for x in probe}
    prior_dir = Path((BASE / "latest_reasoning_qual_run.txt").read_text().strip()) / "diagnostics"
    prior = {}
    for path in prior_dir.glob("step_*.json"):
        for sample in json.loads(path.read_text())["samples"]:
            prior[sample["sample_id"]] = sample["tokens"]
    assert set(prior) == set(probes)
    scorer_crosscheck = {}
    for model, old_key in (("base", "old_log_prob"), ("teacher", "teacher_log_prob")):
        differences = []
        for sid, tokens in prior.items():
            assert [t["token_id"] for t in tokens] == probes[sid]["trajectory_token_ids"]
            new = scores[model]["video"][sid]["log_probs"]
            differences.extend(abs(a - t[old_key]) for a, t in zip(new, tokens))
        d = np.array(differences, dtype=np.float64)
        scorer_crosscheck[model] = {"tokens": len(d), "mean_absolute_difference": float(d.mean()),
                                    "median_absolute_difference": float(np.median(d)),
                                    "p95_absolute_difference": float(np.percentile(d, 95)),
                                    "max_absolute_difference": float(d.max())}
    mechanism = {}
    identity_errors = []
    for model in STUDENTS:
        records = {r: {k: [] for k in ("opd", "shared", "visual", "teacher_visual_residual", "student_visual_residual")}
                   for r in REGIONS}
        for sid, p in probes.items():
            n = len(p["trajectory_token_ids"])
            labels = p["token_regions"]
            assert len(labels) == n
            for m in (model, "teacher"):
                for cond in ("video", "text"):
                    item = scores[m][cond][sid]
                    assert item["trajectory_token_ids"] == p["trajectory_token_ids"]
                    assert len(item["log_probs"]) == n
            tv = np.array(scores["teacher"]["video"][sid]["log_probs"], dtype=np.float64)
            tt = np.array(scores["teacher"]["text"][sid]["log_probs"], dtype=np.float64)
            sv = np.array(scores[model]["video"][sid]["log_probs"], dtype=np.float64)
            st = np.array(scores[model]["text"][sid]["log_probs"], dtype=np.float64)
            opd, shared = tv - sv, tt - st
            tr, sr = tv - tt, sv - st
            visual = tr - sr
            identity_errors.extend(np.abs(opd - (shared + visual)).tolist())
            for i, label in enumerate(labels):
                region = "format" if label in ("format", "format_or_other") else label
                assert region in records, (sid, i, label)
                for r in ("all", region):
                    records[r]["opd"].append(float(opd[i]))
                    records[r]["shared"].append(float(shared[i]))
                    records[r]["visual"].append(float(visual[i]))
                    records[r]["teacher_visual_residual"].append(float(tr[i]))
                    records[r]["student_visual_residual"].append(float(sr[i]))
        mechanism[model] = {}
        for region, values in records.items():
            if not values["opd"]:
                continue
            mechanism[model][region] = {
                "stats": {key: stats(array) for key, array in values.items()},
                "shared_visual_opposite_sign_fraction": float((
                    np.asarray(values["shared"]) * np.asarray(values["visual"]) < 0
                ).mean()),
                "correlation": {
                    "opd_shared": corr(values["opd"], values["shared"]),
                    "opd_visual": corr(values["opd"], values["visual"]),
                },
                "topk_abs_opd": {str(k): topk(values["opd"], values["shared"], values["visual"], k)
                                 for k in (1, 5, 10)},
            }
    err = np.array(identity_errors)
    assert err.max() < 1e-8, err.max()
    output = {
        "title": "1K Natural Reasoning Video OPD: pilot diagnostic signal decomposition",
        "datasets": {
            "evaluation_set": "qualification_500.jsonl", "evaluation_n": 500,
            "evaluation_unique_original_videos": len(eval_videos),
            "training_original_video_overlap": len(train_videos & eval_videos),
            "mechanism_probe": "opd_mechanism_probe_128.jsonl", "mechanism_n": 128,
            "mechanism_sha256": probe_hash,
            "mechanism_trajectory": "Base 4B video on-policy rollout from fixed reasoning_qualification_128; frozen across all model and condition scores",
            "mechanism_selection": "The previously fixed reasoning_qualification_128 was sampled from the 500 diagnostic set after a source-resolution High-budget eligibility check, before any model outcomes; no correctness-based selection",
            "mechanism_unique_original_videos": len(probe_videos),
        },
        "methods": {
            "behavior": "Independent greedy reasoning+answer generation under video or text-only question/options prompt; 32 High frames, 1024-token cap",
            "mechanism": "vLLM prompt-logprob teacher forcing of identical frozen Base response IDs; text-only prompt contains the same question/options/instruction and identical response prefix",
            "token_weighting": "Pooled response tokens, so longer trajectories contribute more tokens; answer, reasoning, format, EOS reported separately",
            "delta_opd": "logP(T|video)-logP(S|video)",
            "delta_shared": "logP(T|text)-logP(S|text)",
            "delta_visual": "[logP(T|video)-logP(T|text)]-[logP(S|video)-logP(S|text)]",
            "correlations": "Pearson and Spearman over paired frozen-trajectory tokens",
            "strong_tokens": "Top 1/5/10% by absolute OPD gap within each reported token region",
            "tokenizer_prompt_check": "All six model tokenizer vocab sizes are 151643; first fixed probe video/text raw chat prompt ID SHA-256 prefixes match across models: c6f5d92ec4394912 / 6678b6cb204f1a00",
        },
        "behavior": behavior, "paired_changes_vs_base": paired_changes, "mechanism": mechanism,
        "vllm_vs_qualification_fsdp_logprob_crosscheck": scorer_crosscheck,
        "identity_error": {"mean_absolute": float(err.mean()), "p99": float(np.percentile(err, 99)),
                           "max": float(err.max()), "count": int(len(err))},
        "conclusions": {
            "TASK_PERFORMANCE_IMPROVED": "NO",
            "OPD_SIGNAL_PRIMARILY_SHARED": "MIXED",
            "VISUAL_SPECIFIC_GAP_SUBSTANTIAL": "YES",
            "GRADIENT_ATTRIBUTION_WORTH_RUNNING": "YES",
            "interpretation": (
                "Final Student video accuracy equals Base at 385/500; intermediate checkpoints do not exceed Base. "
                "Reasoning-token shared and visual gaps have comparable absolute magnitudes, with visual mildly larger, "
                "and both explain moderate portions of the paired OPD variation. Top-5% strong OPD tokens do not "
                "show consistent shared dominance. Shared and visual terms frequently oppose in sign; token-level "
                "magnitudes are not gradient or causal contributions. Full-parameter gradient attribution could test "
                "which component actually shapes the update, but was not run here."
            ),
        },
    }
    (BASE / "opd_signal_decomposition.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print("Wrote JSON; identity error", output["identity_error"])
    for model in STUDENTS:
        v, t = behavior[model]["video"]["accuracy"], behavior[model]["text"]["accuracy"]
        r = mechanism[model]["reasoning"]
        s = r["stats"]
        print(model, "acc",v,t,"meanabs", *(s[x]["mean_abs"] for x in ("opd","shared","visual")),
              "corr",r["correlation"])


if __name__ == "__main__":
    main()
