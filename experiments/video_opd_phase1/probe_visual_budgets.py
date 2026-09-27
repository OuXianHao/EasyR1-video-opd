"""Measure candidate grids before model runs, capping at real JPEG pixels."""

import json
from pathlib import Path

from transformers import AutoProcessor

from verl.utils.dataset import process_video

from inspect_original_200 import prompt_for

BASE = Path(__file__).resolve().parent
ROWS = [json.loads(line) for line in (BASE / 'diagnostic_64.jsonl').read_text().splitlines()]
ORIGINAL = {x['sample_id']: x for x in map(json.loads, (BASE / 'original_200_input_audit.jsonl').open())}
BUDGETS = {'original': 16384, 'medium': 98304, 'high': 163840}


def make_video_inputs(row, processor, condition):
    source = ORIGINAL[row['sample_id']]
    source_min_pixels = min(w * h for w, h in source['disk_frame_sizes_wh'])
    source_min_w = min(w for w, h in source['disk_frame_sizes_wh'])
    source_min_h = min(h for w, h in source['disk_frame_sizes_wh'])
    budget = BUDGETS[condition]
    effective_max = budget if condition == 'original' else min(budget, source_min_pixels)
    for _ in range(8):
        (video, metadata), fps = process_video(
            row['frame_paths'], 4096, effective_max, 2.0,
            return_fps=True, return_metadata=True,
        )
        prompt = prompt_for(row, processor)
        inputs = processor(text=[prompt], videos=[video], video_metadata=[metadata],
                           add_special_tokens=False, return_tensors='pt')
        grid = inputs['video_grid_thw'][0].tolist()
        final_h, final_w = grid[1] * 16, grid[2] * 16
        if condition == 'original' or (final_h <= source_min_h and final_w <= source_min_w):
            break
        effective_max = max(4096, int(effective_max * 0.9))
    else:
        raise RuntimeError(f"Could not avoid upscaling for {row['sample_id']}")
    assert video.shape[0] == 32 and grid[0] == 16
    info = {
        'configured_max_pixels': budget, 'effective_max_pixels': effective_max,
        'source_min_pixels': source_min_pixels,
        'qwen_stage_resolution_hw': list(video.shape[-2:]),
        'processor_resolution_hw': [final_h, final_w],
        'video_grid_thw': grid,
        'visual_tokens': int((inputs['input_ids'] == processor.video_token_id).sum()),
        'input_tokens': int(inputs['input_ids'].shape[-1]),
        'frame_count': int(video.shape[0]),
    }
    return inputs, info


def main():
    processor = AutoProcessor.from_pretrained(
        '/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct', local_files_only=True
    )
    with (BASE / 'budget_grid_probe.jsonl').open('w') as f:
        for i, row in enumerate(ROWS):
            for condition, budget in BUDGETS.items():
                _, info = make_video_inputs(row, processor, condition)
                result = {'sample_id': row['sample_id'], 'condition': condition, **info}
                f.write(json.dumps(result) + '\n')
            if (i + 1) % 16 == 0:
                print(f'probed {i + 1}/{len(ROWS)}', flush=True)


if __name__ == '__main__':
    main()
