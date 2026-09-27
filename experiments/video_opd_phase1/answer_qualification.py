"""Independent deterministic 32-frame multiple-choice answers for a local model."""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from verl.utils.dataset import process_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', default='experiments/video_opd_phase1/qualification_32.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', device_map='cuda'
    ).eval()
    rows = [json.loads(s) for s in Path(args.manifest).read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f, torch.inference_mode():
        for row in rows:
            t0 = time.monotonic()
            (video, metadata), fps = process_video(
                row['frame_paths'], 4096, 16384, 2.0, return_fps=True, return_metadata=True
            )
            assert video.shape[0] == 32
            letters = ''.join(chr(ord('A') + i) for i in range(len(row['options'])))
            prompt = (
                row['question'] + '\n' + '\n'.join(row['options'])
                + '\nAnswer with exactly one option letter: ' + ', '.join(letters) + '.'
            )
            messages = [{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text': '\n' + prompt}]}]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(
                text=[prompt_text], videos=[video], video_metadata=[metadata],
                add_special_tokens=False, return_tensors='pt'
            ).to('cuda')
            assert inputs['video_grid_thw'].tolist()[0][0] == 16
            output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            response = processor.tokenizer.decode(output_ids[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
            answer = response.strip()
            valid = re.fullmatch(f'[{re.escape(letters)}]', answer) is not None
            result = {
                'sample_id': row['sample_id'], 'video_id': row['video_id'],
                'answer': row['answer'], 'response': response, 'parsed_answer': answer if valid else None,
                'format_valid': valid, 'correct': valid and answer == row['answer'],
                'frame_count': 32, 'video_grid_thw': inputs['video_grid_thw'].tolist(),
                'visual_tokens': int((inputs['input_ids'] == processor.video_token_id).sum().item()),
                'seconds': round(time.monotonic() - t0, 3),
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
