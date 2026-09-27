"""Trace the exact original HF qualification preprocessing for frozen 200 rows."""

import json
import re
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, GenerationConfig

from verl.utils.dataset import process_video


BASE = Path(__file__).resolve().parent
MODEL = '/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct'
ROWS = [json.loads(line) for line in (BASE / 'qualification_200.jsonl').read_text().splitlines()]


def prompt_for(row, processor, video=True):
    letters = ''.join(chr(ord('A') + i) for i in range(len(row['options'])))
    body = (
        row['question'] + '\n' + '\n'.join(row['options'])
        + '\nAnswer with exactly one option letter: ' + ', '.join(letters) + '.'
    )
    content = ([{'type': 'video'}, {'type': 'text', 'text': '\n' + body}]
               if video else [{'type': 'text', 'text': '\n' + body}])
    return processor.apply_chat_template(
        [{'role': 'user', 'content': content}], add_generation_prompt=True, tokenize=False
    )


def main():
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    tokenizer = processor.tokenizer
    generation_config = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    prior_answers = {
        name: {x['sample_id']: x for x in map(json.loads, (BASE / f'{file}_answers_200.jsonl').open())}
        for name, file in (('student', 'student'), ('teacher', 'teacher'))
    }
    audit_path = BASE / 'original_200_input_audit.jsonl'
    with audit_path.open('w') as output:
        for i, row in enumerate(ROWS):
            sizes = []
            for frame in row['frame_paths']:
                with Image.open(frame) as img:
                    sizes.append(list(img.size))
            (video, metadata), sample_fps = process_video(
                row['frame_paths'], 4096, 16384, 2.0, return_fps=True, return_metadata=True
            )
            assert video.shape[0] == 32
            prompt = prompt_for(row, processor)
            inputs = processor(
                text=[prompt], videos=[video], video_metadata=[metadata],
                add_special_tokens=False, return_tensors='pt'
            )
            grid = inputs['video_grid_thw'][0].tolist()
            raw_ids = tokenizer.encode(prompt, add_special_tokens=False)
            result = {
                'sample_id': row['sample_id'], 'type': row['type'],
                'disk_frame_sizes_wh': sizes,
                'qwen_vl_utils_video_tensor_shape': list(video.shape),
                'qwen_vl_utils_frame_resolution_hw': list(video.shape[-2:]),
                'qwen_vl_utils_fps': sample_fps,
                'processor_video_grid_thw': grid,
                'processor_effective_frame_resolution_hw': [grid[1] * 16, grid[2] * 16],
                'pixel_values_videos_shape': list(inputs['pixel_values_videos'].shape),
                'input_frame_count': int(video.shape[0]),
                'raw_prompt_token_count': len(raw_ids),
                'raw_text_template_token_count': len(raw_ids) - raw_ids.count(processor.video_token_id),
                'nonvisual_expanded_token_count': int(inputs['input_ids'].shape[-1])
                    - int((inputs['input_ids'] == processor.video_token_id).sum()),
                'visual_token_count': int((inputs['input_ids'] == processor.video_token_id).sum()),
                'total_input_token_count': int(inputs['input_ids'].shape[-1]),
                'truncation': False, 'max_prompt_length': None,
                'temperature': None, 'top_p': None, 'top_k': None,
                'configured_sampling_defaults_inactive': {
                    k: getattr(generation_config, k) for k in ('temperature', 'top_p', 'top_k')
                },
                'max_new_tokens': 8, 'do_sample': False,
                'student_parsed_answer': prior_answers['student'][row['sample_id']]['parsed_answer'],
                'teacher_parsed_answer': prior_answers['teacher'][row['sample_id']]['parsed_answer'],
                'student_format_valid': prior_answers['student'][row['sample_id']]['format_valid'],
                'teacher_format_valid': prior_answers['teacher'][row['sample_id']]['format_valid'],
                'student_correct': prior_answers['student'][row['sample_id']]['correct'],
                'teacher_correct': prior_answers['teacher'][row['sample_id']]['correct'],
            }
            output.write(json.dumps(result) + '\n')
            output.flush()
            if (i + 1) % 25 == 0:
                print(f'inspected {i + 1}/{len(ROWS)}', flush=True)
    prompt_path = BASE / 'prompt_audit_20.txt'
    with prompt_path.open('w') as f:
        for row in ROWS[:20]:
            prompt = prompt_for(row, processor)
            # The expected answer is intentionally kept outside this file.
            f.write(f"===== {row['sample_id']} =====\n{prompt}\n\n")
    runtime = {
        'script': str(BASE / 'answer_qualification.py'),
        'manifest': str(BASE / 'qualification_200.jsonl'),
        'models': [MODEL, '/home/xianhao/models/Qwen3-VL-32B-Instruct'],
        'actual_cli_arguments': [
            '--model MODEL_PATH', '--manifest experiments/video_opd_phase1/qualification_200.jsonl',
            '--output experiments/video_opd_phase1/{student,teacher}_answers_200.jsonl',
        ],
        'qwen_vl_utils_min_pixels': 4096, 'qwen_vl_utils_max_pixels': 16384,
        'video_sample_fps': 2.0, 'video_metadata_passed': True,
        'generation_call': {'max_new_tokens': 8, 'do_sample': False},
        'generation_config_defaults_inactive_when_do_sample_false': {
            k: getattr(generation_config, k) for k in ('temperature', 'top_p', 'top_k')
        },
        'no_explicit_max_prompt_length_or_truncation': True,
        'prompt_template': 'video content block, newline, question, options, direct answer instruction',
        'processor_video_config': processor.video_processor.to_dict(),
        'processor_image_config': processor.image_processor.to_dict(),
    }
    (BASE / 'original_200_runtime.json').write_text(json.dumps(runtime, indent=2, default=str) + '\n')
    print(f'wrote {audit_path}', flush=True)


if __name__ == '__main__':
    main()
