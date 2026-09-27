"""Score identical frozen Base response tokens under video and text prompts."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

from eval_reasoning_behavior import BASE, MODELS, make_input


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--condition", choices=("video", "text"), required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    os.environ["EASYR1_VIDEO_OPD_PREPROCESS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    rows = {r["sample_id"]: r for r in map(json.loads, (BASE / "reasoning_qualification_128.jsonl").read_text().splitlines())}
    probe = [json.loads(line) for line in (BASE / "opd_mechanism_probe_128.jsonl").read_text().splitlines()]
    assert len(probe) == 128 and set(rows) == {x["sample_id"] for x in probe}
    model_path = MODELS[args.model]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    base_tok = AutoTokenizer.from_pretrained(MODELS["base"], local_files_only=True)
    assert tokenizer.vocab_size == base_tok.vocab_size
    for p in probe:
        assert tokenizer.decode(p["trajectory_token_ids"], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) == base_tok.decode(
                                    p["trajectory_token_ids"], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
    engine = LLM(
        model=str(model_path), dtype="bfloat16", tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.72, max_model_len=4608,
        max_num_batched_tokens=4608, enforce_eager=True,
        disable_custom_all_reduce=True, enable_chunked_prefill=True,
        mm_processor_cache_gb=0, mm_processor_kwargs={"do_resize": False},
        trust_remote_code=False, disable_log_stats=True,
    )
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1,
                            detokenize=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["sample_id"] for line in args.output.read_text().splitlines() if line}
    with args.output.open("a") as f:
        for lo in range(args.start, args.end, args.batch):
            selected = [p for p in probe[lo:min(lo + args.batch, args.end)] if p["sample_id"] not in done]
            if not selected:
                continue
            inputs = []
            infos = []
            for p in selected:
                item, visual_tokens, grid = make_input(rows[p["sample_id"]], tokenizer, args.condition)
                prompt_len = len(item["prompt_token_ids"])
                item["prompt_token_ids"] = item["prompt_token_ids"] + p["trajectory_token_ids"]
                inputs.append(item)
                infos.append((prompt_len, visual_tokens, grid))
            tick = time.monotonic()
            outputs = engine.generate(inputs, params, use_tqdm=False)
            batch_sec = time.monotonic() - tick
            for p, result, (prompt_len, visual_tokens, grid) in zip(selected, outputs, infos):
                n = len(p["trajectory_token_ids"])
                lp_items = result.prompt_logprobs
                if lp_items is None:
                    raise RuntimeError("vLLM returned no prompt logprobs")
                # vLLM returns one entry per processed prompt token; response tokens are the suffix.
                suffix = lp_items[-n:]
                if len(suffix) != n:
                    raise RuntimeError((p["sample_id"], len(lp_items), n))
                lps = []
                for tid, options in zip(p["trajectory_token_ids"], suffix):
                    if options is None or tid not in options:
                        raise RuntimeError(f"Missing target prompt logprob {p['sample_id']} token {tid}: {options}")
                    lps.append(float(options[tid].logprob))
                if not np.isfinite(lps).all():
                    raise RuntimeError(f"Nonfinite logprob {p['sample_id']}")
                item = {
                    "sample_id": p["sample_id"], "model": args.model, "condition": args.condition,
                    "trajectory_token_ids": p["trajectory_token_ids"], "log_probs": lps,
                    "raw_prompt_tokens": prompt_len, "scored_tokens": n,
                    "vllm_prompt_logprobs_length": len(lp_items),
                    "frame_count": 32 if args.condition == "video" else 0,
                    "visual_tokens": visual_tokens, "video_grid_thw": grid,
                    "decode_seconds_per_batch": batch_sec,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                done.add(p["sample_id"])
            print(f"{args.model}/{args.condition} scored {len(done)}; last batch {batch_sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
