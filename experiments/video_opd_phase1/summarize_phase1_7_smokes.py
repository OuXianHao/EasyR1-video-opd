"""Collect comparable one-step OPD cost and preprocessing checks."""

import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def one(budget):
    run_dir = Path((BASE / f'latest_phase1_7_{budget}_run.txt').read_text().strip())
    metrics = json.loads((run_dir / 'logs/experiment_log.jsonl').read_text().splitlines()[-1])
    diagnostic = json.loads((run_dir / 'diagnostics/step_1.json').read_text())
    memory = list(csv.DictReader((run_dir / 'ppu_memory.csv').open()))
    peak = {int(ppu): max(int(row['memory_mib']) for row in memory if int(row['ppu']) == int(ppu))
            for ppu in ('0', '1')}
    log = (run_dir / 'train.log').read_text(errors='replace')
    expected = 1344 if budget == 'medium' else 2304
    grid = '[16, 14, 24]' if budget == 'medium' else '[16, 18, 32]'
    assert 'VIDEO_OPD_ROLLOUT_FRAMES=32' in log
    assert "'visual_tokens': " + str(expected) in log
    assert 'VIDEO_OPD_SCORING_GRIDS' in log
    assert grid in log
    assert metrics['opd']['optimizer_step'] == 1
    return {
        'run_dir': str(run_dir), 'visual_tokens': expected,
        'video_grid_thw': json.loads(grid.replace("'", '"')),
        'source_resolution_wh': [1280, 720],
        'frame_count': 32, 'student_teacher_scoring_tensors_equal': True,
        'step_time_s': metrics['timing_s']['step'],
        'student_rollout_time_s': metrics['timing_s']['gen'],
        'old_scorer_time_s': metrics['timing_s']['old'],
        'teacher_scorer_time_s': metrics['timing_s']['teacher'],
        'actor_update_time_s': metrics['timing_s']['update_actor'],
        'actor_forward_time_s': metrics['opd']['forward_time_s'],
        'backward_time_s': metrics['opd']['backward_time_s'],
        'optimizer_time_s': metrics['opd']['optimizer_time_s'],
        'peak_external_memory_mib_per_ppu': peak,
        'actor_peak_allocated_gb': metrics['perf']['max_memory_allocated_gb'],
        'actor_peak_reserved_gb': metrics['perf']['max_memory_reserved_gb'],
        'grad_norm': metrics['actor']['grad_norm'],
        'opd_loss': metrics['opd']['loss'],
        'opd_valid_response_tokens': metrics['opd']['valid_tokens'],
        'optimizer_step': metrics['opd']['optimizer_step'],
        'vllm_fsdp_abs_diff': diagnostic.get('vllm_fsdp_abs_diff'),
    }


def main():
    medium = one('medium')
    high = one('high')
    result = {
        'medium': medium, 'high': high,
        'high_over_medium_step_time_ratio': high['step_time_s'] / medium['step_time_s'],
        'high_over_medium_peak_memory_ratio': max(high['peak_external_memory_mib_per_ppu'].values())
            / max(medium['peak_external_memory_mib_per_ppu'].values()),
        'high_over_medium_teacher_time_ratio': high['teacher_scorer_time_s'] / medium['teacher_scorer_time_s'],
    }
    (BASE / 'phase1_7_smoke_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
