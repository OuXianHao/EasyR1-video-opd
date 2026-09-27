"""Deterministic paired MCQ evaluation on the frozen Video OPD diagnostic 64."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from inspect_original_200 import prompt_for
from probe_visual_budgets import BASE, ROWS, make_video_inputs


MODELS = {
    '4b': '/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct',
    '32b': '/home/xianhao/models/Qwen3-VL-32B-Instruct',
}
CONDITIONS = ('original', 'medium', 'high', 'text_only')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument('--limit', type=int, default=64)
    args = parser.parse_args()
    path = BASE / f'visual_budget_{args.model}.jsonl'
    done = {(r['sample_id'], r['condition']) for r in map(json.loads, path.open())} if path.exists() else set()
    processor = AutoProcessor.from_pretrained(MODELS[args.model], local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODELS[args.model], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda'
    ).eval()
    with path.open('a') as output, torch.inference_mode():
        for row in ROWS[:args.limit]:
            letters = ''.join(chr(ord('A') + i) for i in range(len(row['options'])))
            letter_ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in letters}
            if not all(len(ids) == 1 for ids in letter_ids.values()):
                raise RuntimeError(f'Option is not one token: {letter_ids}')
            for condition in args.conditions:
                if (row['sample_id'], condition) in done:
                    continue
                t0 = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                if condition == 'text_only':
                    prompt = prompt_for(row, processor, video=False)
                    inputs = processor(text=[prompt], add_special_tokens=False, return_tensors='pt')
                    info = {
                        'frame_count': 0, 'visual_tokens': 0, 'video_grid_thw': None,
                        'qwen_stage_resolution_hw': None, 'processor_resolution_hw': None,
                        'input_tokens': int(inputs['input_ids'].shape[-1]),
                    }
                else:
                    inputs, info = make_video_inputs(row, processor, condition)
                inputs = inputs.to('cuda')
                output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                response = tokenizer.decode(
                    output_ids[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True
                )
                answer = response.strip()
                valid = re.fullmatch(f'[{re.escape(letters)}]', answer) is not None
                del output_ids
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                log_probs = logits.log_softmax(dim=-1)
                option_log_probs = {letter: float(log_probs[ids[0]]) for letter, ids in letter_ids.items()}
                gt_log_prob = option_log_probs[row['answer']]
                result = {
                    'sample_id': row['sample_id'], 'video_id': row['video_id'],
                    'type': row.get('type'), 'condition': condition,
                    'ground_truth': row['answer'], 'response': response,
                    'parsed_answer': answer if valid else None,
                    'format_valid': valid,
                    'correct': bool(valid and answer == row['answer']),
                    'gt_option_log_prob_full_vocab': gt_log_prob,
                    'gt_option_probability_full_vocab': float(torch.exp(log_probs[letter_ids[row['answer']][0]])),
                    'gt_nll': -gt_log_prob,
                    'gt_vs_best_wrong_log_prob_margin': gt_log_prob - max(
                        value for letter, value in option_log_probs.items() if letter != row['answer']
                    ),
                    'option_log_probs_full_vocab': option_log_probs,
                    'seconds': time.monotonic() - t0,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'truncation': False, 'max_prompt_length': None,
                    'max_new_tokens': 8, 'do_sample': False,
                    'temperature': None, 'top_p': None, 'top_k': None,
                    **info,
                }
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                output.flush()
                done.add((row['sample_id'], condition))
                print(f"{args.model} {len(done)} {row['sample_id']} {condition} "
                      f"{answer!r} GT={row['answer']} visual={info['visual_tokens']} "
                      f"time={result['seconds']:.1f}s", flush=True)
                del logits, log_probs, inputs


if __name__ == '__main__':
    main()
