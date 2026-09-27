"""Summarize on-policy reasoning and token-level 32B teacher forcing."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from reasoning_format import parse_answer, parse_regions

BASE = Path(__file__).resolve().parent
STUDENT = "/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct"


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    return {
        "count": int(len(a)), "mean": float(a.mean()), "std": float(a.std()),
        "min": float(a.min()), "p1": float(np.percentile(a, 1)),
        "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def delta_stats(items):
    values = [item["delta"] for item in items]
    result = describe(values)
    if values:
        result["positive_fraction"] = sum(v > 0 for v in values) / len(values)
        result["negative_fraction"] = sum(v < 0 for v in values) / len(values)
    return result


def token_spans(tokenizer, ids):
    full = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    pieces = [tokenizer.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False) for i in ids]
    if "".join(pieces) != full:
        # Byte-fragment tokens need context to decode correctly.
        prefixes = [""] + [tokenizer.decode(ids[:i], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False) for i in range(1, len(ids) + 1)]
        return full, [(len(prefixes[i]), len(prefixes[i + 1])) for i in range(len(ids))]
    starts = np.cumsum([0] + [len(piece) for piece in pieces]).tolist()
    return full, list(zip(starts[:-1], starts[1:]))


def classify(span, regions):
    if not regions:
        return "unclassified"
    a, b = span
    overlaps = {key: max(0, min(b, y) - max(a, x)) for key, (x, y) in regions.items()}
    # Qwen tokenizes the final letter with the preceding '>' (e.g. '>C').
    # Attribute that inseparable joint token to the answer and report it as mixed.
    if overlaps.get("answer", 0):
        return "answer"
    if overlaps.get("reasoning", 0):
        return "reasoning"
    return "format" if any(overlaps.values()) else "unclassified"


def partial_regions(response):
    """Locate the reasoning content even when a closing tag was omitted."""
    start = response.find("<thinking>")
    if start < 0:
        return None
    content_start = start + len("<thinking>")
    stops = [pos for marker in ("</thinking>", "<answer>", "\nAnswer:")
             if (pos := response.find(marker, content_start)) >= 0]
    content_end = min(stops) if stops else len(response)
    result = {"thinking_open": (start, content_start),
              "reasoning": (content_start, content_end)}
    for name, marker in (("thinking_close", "</thinking>"),
                         ("answer_open", "<answer>"), ("answer_close", "</answer>")):
        pos = response.find(marker)
        if pos >= 0:
            result[name] = (pos, pos + len(marker))
    answer_match = re.search(r"<answer>\s*([A-Z])", response)
    if answer_match:
        result["answer"] = answer_match.span(1)
    return result


def main():
    run_dir = Path((BASE / "latest_reasoning_qual_run.txt").read_text().strip())
    files = sorted((run_dir / "diagnostics").glob("step_*.json"),
                   key=lambda path: int(path.stem.split("_")[-1]))
    if len(files) != 8:
        raise RuntimeError(f"Expected 8 qualification steps, found {len(files)}: {files}")
    rows = {row["sample_id"]: row for row in map(json.loads,
            (BASE / "reasoning_qualification_128.jsonl").read_text().splitlines())}
    tokenizer = AutoTokenizer.from_pretrained(STUDENT, local_files_only=True)
    eos_ids = set(tokenizer.all_special_ids)
    per_sample = []
    region_items = defaultdict(list)
    extremes = []
    mixed_answer_tokens = 0
    step_stats = []
    for path in files:
        diagnostic = json.loads(path.read_text())
        step_stats.append({"step": diagnostic["step"], "delta": diagnostic["delta"],
                           "vllm_fsdp_abs_diff": diagnostic.get("vllm_fsdp_abs_diff")})
        for sample in diagnostic["samples"]:
            sample_id = sample["sample_id"]
            source = rows[sample_id]
            tokens = sample["tokens"]
            ids = [item["token_id"] for item in tokens]
            generated_ids = list(ids)
            while generated_ids and generated_ids[-1] in eos_ids:
                generated_ids.pop()
            response, spans = token_spans(tokenizer, generated_ids)
            allowed = [chr(ord("A") + i) for i in range(len(source["options"]))]
            strict_regions = parse_regions(response, allowed)
            regions = strict_regions or partial_regions(response)
            final = parse_answer(response, allowed)
            labels = [classify(span, regions) for span in spans]
            labels.extend("eos" if token_id in eos_ids else "unclassified"
                          for token_id in ids[len(generated_ids):])
            for i, (item, region) in enumerate(zip(tokens, labels)):
                if region == "answer" and regions:
                    a, b = spans[i]
                    if any(max(0, min(b, y) - max(a, x)) > 0 for key, (x, y) in
                           regions.items() if key not in ("answer", "reasoning")):
                        mixed_answer_tokens += 1
                enriched = dict(item)
                enriched.update({"sample_id": sample_id, "step": diagnostic["step"],
                                 "region": region,
                                 "context": tokenizer.decode(ids[max(0, i - 5):min(len(ids), i + 6)],
                                              skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)})
                region_items[region].append(enriched)
                extremes.append(enriched)
            per_sample.append({
                "sample_id": sample_id, "step": diagnostic["step"],
                "type": source["type"], "question": source["question"],
                "options": source["options"], "student_response": response,
                "student_final_answer": final, "gt_answer": source["answer"],
                "response_total_tokens": len(ids),
                "reasoning_content_tokens": labels.count("reasoning"),
                "answer_content_tokens": labels.count("answer"),
                "thinking_open": "<thinking>" in response,
                "thinking_close": "</thinking>" in response,
                "answer_open": "<answer>" in response,
                "answer_close": "</answer>" in response,
                "format_valid": strict_regions is not None,
                "answer_parse_valid": final is not None,
                "hit_1024": len(ids) >= 1024,
                "has_eos": len(generated_ids) < len(ids),
            })
    if len(per_sample) != 128 or len({s["sample_id"] for s in per_sample}) != 128:
        raise RuntimeError("Qualification did not consume exactly 128 unique samples")
    all_deltas = [item["delta"] for item in extremes]
    if not all(math.isfinite(x) for x in all_deltas):
        raise RuntimeError("Non-finite teacher-student gap")
    n = len(per_sample)
    result = {
        "run_dir": str(run_dir), "n": n,
        "total_response_tokens": describe([s["response_total_tokens"] for s in per_sample]),
        "reasoning_content_tokens": describe([s["reasoning_content_tokens"] for s in per_sample]),
        "two_token_response_fraction": sum(s["response_total_tokens"] <= 2 for s in per_sample) / n,
        "empty_reasoning_fraction": sum(s["reasoning_content_tokens"] == 0 for s in per_sample) / n,
        "truncation_1024_fraction": sum(s["hit_1024"] for s in per_sample) / n,
        "format_valid_rate": sum(s["format_valid"] for s in per_sample) / n,
        "answer_parse_valid_rate": sum(s["answer_parse_valid"] for s in per_sample) / n,
        "tag_rates": {key: sum(s[key] for s in per_sample) / n for key in
                      ("thinking_open", "thinking_close", "answer_open", "answer_close")},
        "region_delta": {key: delta_stats(region_items[key]) for key in
                         ("reasoning", "answer", "format", "eos", "unclassified")},
        "region_token_counts": dict(Counter(item["region"] for item in extremes)),
        "answer_tokens_mixed_with_format": mixed_answer_tokens,
        "answer_delta_note": "The Qwen tokenizer often merges '>' with the answer letter; answer Delta is the joint sampled-token gap, not a separable character probability.",
        "step_stats": step_stats,
        "top_positive_delta_tokens": sorted(extremes, key=lambda x: x["delta"], reverse=True)[:30],
        "top_negative_delta_tokens": sorted(extremes, key=lambda x: x["delta"])[:30],
    }
    (BASE / "reasoning_qualification_rollouts_128.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in per_sample))
    with (BASE / "reasoning_qualification_samples_30.md").open("w") as out:
        out.write("# Thirty complete on-policy Student reasoning samples\n\n")
        for i, sample in enumerate(per_sample[:30], start=1):
            out.write(f"## {i}. {sample['sample_id']} ({sample['type']})\n\n")
            out.write(f"Question: {sample['question']}\n\nOptions:\n\n")
            for option in sample["options"]:
                out.write(f"- {option}\n")
            out.write(f"\nStudent rollout:\n\n```text\n{sample['student_response']}\n```\n\n")
            out.write(f"Parsed final: {sample['student_final_answer']}  \nGT: {sample['gt_answer']}\n\n")
    (BASE / "reasoning_qualification_diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in
                      ("top_positive_delta_tokens", "top_negative_delta_tokens", "step_stats")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
