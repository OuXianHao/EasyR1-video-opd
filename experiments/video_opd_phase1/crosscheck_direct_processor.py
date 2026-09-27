"""Small direct raw-JPEG Transformers/Qwen processor cross-check."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluate_visual_budget import MODELS
from inspect_original_200 import prompt_for
from probe_visual_budgets import BASE, ROWS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--limit', type=int, default=16)
    args = p.parse_args()
    output_path = BASE / f'direct_processor_{args.model}.jsonl'
    done = {x['sample_id'] for x in map(json.loads, output_path.open())} if output_path.exists() else set()
    processor = AutoProcessor.from_pretrained(MODELS[args.model], local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODELS[args.model], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda'
    ).eval()
    with output_path.open('a') as f, torch.inference_mode():
        for row in ROWS[:args.limit]:
            if row['sample_id'] in done:
                continue
            frames = []
            for path in row['frame_paths']:
                with Image.open(path) as im:
                    frames.append(np.asarray(im.convert('RGB')))
            raw_video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            assert raw_video.shape[0] == 32
            metadata = {'fps': 2.0, 'frames_indices': list(range(32)), 'total_num_frames': 32}
            prompt = prompt_for(row, processor)
            # The processor's longest_edge applies to the full video, not one frame.
            inputs = processor(
                text=[prompt], videos=[raw_video], video_metadata=[metadata],
                size={'shortest_edge': 4096, 'longest_edge': 32 * 16384},
                add_special_tokens=False, return_tensors='pt'
            )
            grid = inputs['video_grid_thw'][0].tolist()
            assert grid[0] == 16
            inputs = inputs.to('cuda')
            ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            response = processor.tokenizer.decode(
                ids[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True
            )
            letters = ''.join(chr(ord('A') + i) for i in range(len(row['options'])))
            answer = response.strip()
            valid = re.fullmatch(f'[{re.escape(letters)}]', answer) is not None
            record = {
                'sample_id': row['sample_id'], 'condition': 'raw_jpg_direct_hf_processor',
                'source_resolution_hw': list(raw_video.shape[-2:]), 'frame_count': 32,
                'video_grid_thw': grid,
                'processor_resolution_hw': [grid[1] * 16, grid[2] * 16],
                'visual_tokens': int((inputs['input_ids'] == processor.video_token_id).sum()),
                'input_tokens': int(inputs['input_ids'].shape[-1]),
                'prompt': prompt, 'max_new_tokens': 8, 'do_sample': False,
                'response': response, 'parsed_answer': answer if valid else None,
                'ground_truth': row['answer'], 'correct': bool(valid and answer == row['answer']),
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
            print(f"{args.model} {len(done)+1}/{args.limit} {row['sample_id']} {grid} {answer}", flush=True)
            done.add(row['sample_id'])
            del frames, raw_video, inputs, ids


if __name__ == '__main__':
    main()
