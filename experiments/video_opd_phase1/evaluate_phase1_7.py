"""Deterministic 500-question, video-disjoint Qwen3-VL qualification."""

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from verl.utils.dataset import process_video

from evaluate_visual_budget import MODELS
from inspect_original_200 import prompt_for

BASE = Path(__file__).resolve().parent
BUDGETS = {'medium': 86016, 'high': 147456}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--budget', choices=BUDGETS, required=True)
    parser.add_argument('--condition', choices=('video', 'text_only'), required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=500)
    parser.add_argument('--suffix', default='')
    args = parser.parse_args()
    if args.condition == 'video':
        os.environ['EASYR1_VIDEO_OPD_PREPROCESS'] = '1'
    path = BASE / f'phase1_7_{args.model}_{args.budget}_{args.condition}{args.suffix}.jsonl'
    done = {x['sample_id'] for x in map(json.loads, path.open())} if path.exists() else set()
    rows = [json.loads(x) for x in (BASE / 'qualification_500.jsonl').read_text().splitlines()]
    assert len(rows) == 500
    processor = AutoProcessor.from_pretrained(MODELS[args.model], local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        MODELS[args.model], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda'
    ).eval()
    with path.open('a') as output, torch.inference_mode():
        for index in range(args.start, args.end):
            row = rows[index]
            if row['sample_id'] in done:
                continue
            start = time.monotonic()
            letters = ''.join(chr(ord('A') + i) for i in range(len(row['options'])))
            option_ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in letters}
            assert all(len(ids) == 1 for ids in option_ids.values())
            if args.condition == 'video':
                with Image.open(row['frame_paths'][0]) as frame:
                    source_resolution_wh = list(frame.size)
                video, metadata = process_video(
                    row['frame_paths'], 4096, BUDGETS[args.budget], 2.0,
                    return_metadata=True,
                )
                prompt = prompt_for(row, processor, video=True)
                inputs = processor(
                    text=[prompt], videos=[video], video_metadata=[metadata],
                    do_resize=False, add_special_tokens=False, return_tensors='pt'
                )
                grid = inputs['video_grid_thw'][0].tolist()
                assert video.shape[0] == 32 and grid[0] == 16
                assert grid[1] * 16 == video.shape[-2] and grid[2] * 16 == video.shape[-1]
                info = {
                    'source_resolution_wh': source_resolution_wh,
                    'processed_resolution_hw': list(video.shape[-2:]),
                    'frame_count': 32, 'video_grid_thw': grid,
                    'visual_tokens': int((inputs['input_ids'] == processor.video_token_id).sum()),
                }
            else:
                prompt = prompt_for(row, processor, video=False)
                inputs = processor(text=[prompt], add_special_tokens=False, return_tensors='pt')
                info = {
                    'source_resolution_wh': None, 'processed_resolution_hw': None,
                    'frame_count': 0, 'video_grid_thw': None, 'visual_tokens': 0,
                }
            info['input_tokens'] = int(inputs['input_ids'].shape[-1])
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
            option_log_probs = {letter: float(log_probs[ids[0]]) for letter, ids in option_ids.items()}
            gt_log_prob = option_log_probs[row['answer']]
            result = {
                'index': index, 'sample_id': row['sample_id'],
                'original_video_id': row['original_video_id'], 'video_id': row['video_id'],
                'question_type': row['type'], 'category': row['category'],
                'source_split': row['source_split'], 'model': args.model,
                'visual_budget': args.budget, 'condition': args.condition,
                'ground_truth': row['answer'], 'response': response,
                'parsed_answer': answer if valid else None,
                'format_valid': valid, 'correct': bool(valid and answer == row['answer']),
                'gt_log_prob_full_vocab': gt_log_prob,
                'gt_probability_full_vocab': float(log_probs[option_ids[row['answer']][0]].exp()),
                'gt_nll': -gt_log_prob,
                'gt_vs_best_wrong_margin': gt_log_prob - max(
                    value for letter, value in option_log_probs.items() if letter != row['answer']
                ),
                'option_log_probs_full_vocab': option_log_probs,
                'seconds': time.monotonic() - start,
                'max_new_tokens': 8, 'do_sample': False,
                'truncation': False, **info,
            }
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            done.add(row['sample_id'])
            if (index - args.start + 1) % 25 == 0:
                print(f"{args.model}/{args.condition}/{args.budget} {index + 1}/{args.end} "
                      f"correct={sum(1 for _ in done)}", flush=True)
            if args.condition == 'video':
                del video
            del logits, log_probs, inputs


if __name__ == '__main__':
    main()
