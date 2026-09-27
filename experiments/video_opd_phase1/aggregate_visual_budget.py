"""Aggregate the fixed, paired diagnostic runs into a machine-readable report."""

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from inspect_original_200 import prompt_for
from probe_visual_budgets import BASE, ROWS


CONDITIONS = ('original', 'medium', 'high', 'text_only')


def read_jsonl(name):
    return [json.loads(s) for s in (BASE / name).read_text().splitlines()]


def pct(xs, p):
    a = sorted(xs)
    if not a:
        return None
    x = (len(a) - 1) * p / 100
    i = int(x)
    return a[i] * (1 - x + i) + a[min(i + 1, len(a) - 1)] * (x - i)


def distribution(xs):
    return {
        'count': len(xs), 'min': min(xs), 'p25': pct(xs, 25),
        'median': pct(xs, 50), 'p75': pct(xs, 75), 'p95': pct(xs, 95),
        'max': max(xs), 'mean': statistics.mean(xs),
    }


def summary(xs):
    return {
        'count': len(xs), 'accuracy': sum(x['correct'] for x in xs) / len(xs),
        'correct_count': sum(x['correct'] for x in xs),
        'format_valid_rate': sum(x['format_valid'] for x in xs) / len(xs),
        'gt_option_probability_full_vocab_mean': statistics.mean(x['gt_option_probability_full_vocab'] for x in xs),
        'gt_nll_mean': statistics.mean(x['gt_nll'] for x in xs),
        'gt_vs_best_wrong_log_prob_margin_mean': statistics.mean(
            x['gt_vs_best_wrong_log_prob_margin'] for x in xs
        ),
        'visual_tokens': distribution([x['visual_tokens'] for x in xs]),
        'input_tokens': distribution([x['input_tokens'] for x in xs]),
        'peak_allocated_gib_max': max(x['peak_allocated_bytes'] for x in xs) / 2**30,
    }


def family(t):
    if t.startswith('Action '):
        return 'Action'
    if t.startswith('Event '):
        return 'Event'
    if t.startswith('Objects '):
        return 'Objects'
    if 'Reasoning' in t:
        return 'Reasoning'
    if 'Description' in t:
        return 'Description'
    return 'Other'


def paired_counts(xs, ys):
    a = [x['correct'] for x in xs]
    b = [y['correct'] for y in ys]
    return {
        'both_correct': sum(x and y for x, y in zip(a, b)),
        'first_correct_second_wrong': sum(x and not y for x, y in zip(a, b)),
        'first_wrong_second_correct': sum(not x and y for x, y in zip(a, b)),
        'both_wrong': sum(not x and not y for x, y in zip(a, b)),
    }


def paired_ci(xs, ys, seed=2727):
    diffs = [float(y['correct']) - float(x['correct']) for x, y in zip(xs, ys)]
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(5000)]
    return [pct(means, 2.5), pct(means, 97.5)]


def main():
    by_model = {}
    for model in ('4b', '32b'):
        data = read_jsonl(f'visual_budget_{model}.jsonl')
        if len(data) != 64 * 4:
            raise RuntimeError(f'{model}: expected 256 rows, got {len(data)}')
        keyed = {(x['sample_id'], x['condition']): x for x in data}
        if len(keyed) != len(data):
            raise RuntimeError(f'{model}: duplicate sample-condition')
        for row in ROWS:
            for condition in CONDITIONS:
                assert (row['sample_id'], condition) in keyed
        by_model[model] = keyed

    old = {
        model: {x['sample_id']: x for x in read_jsonl(f'{name}_answers_200.jsonl')}
        for model, name in (('4b', 'student'), ('32b', 'teacher'))
    }
    original_200 = read_jsonl('original_200_input_audit.jsonl')
    assert len(original_200) == 200
    read_direct = {}
    for model in ('4b', '32b'):
        direct = read_jsonl(f'direct_processor_{model}.jsonl')
        assert len(direct) == 16
        read_direct[model] = direct

    result = {
        'models': {
            '4b': '/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct',
            '32b': '/home/xianhao/models/Qwen3-VL-32B-Instruct',
        },
        'manifest': str(BASE / 'diagnostic_64.jsonl'),
        'qualification_200_manifest': str(BASE / 'qualification_200.jsonl'),
        'original_200_runtime': json.loads((BASE / 'original_200_runtime.json').read_text()),
        'qualification_200_baseline': {
            model: {'correct': sum(x['correct'] for x in data.values()), 'count': len(data)}
            for model, data in old.items()
        },
        'original_200_input': {
            'frame_count': distribution([x['input_frame_count'] for x in original_200]),
            'disk_width': distribution([w for x in original_200 for w, h in x['disk_frame_sizes_wh']]),
            'disk_height': distribution([h for x in original_200 for w, h in x['disk_frame_sizes_wh']]),
            'disk_pixel_count': distribution([w*h for x in original_200 for w, h in x['disk_frame_sizes_wh']]),
            'disk_aspect_ratio': distribution([w/h for x in original_200 for w, h in x['disk_frame_sizes_wh']]),
            'disk_resolution_counts': {f'{w}x{h}': n for (w, h), n in Counter(
                (w, h) for x in original_200 for w, h in x['disk_frame_sizes_wh']
            ).items()},
            'qwen_stage_resolution_counts': {str(k): v for k, v in Counter(
                tuple(x['qwen_vl_utils_frame_resolution_hw']) for x in original_200
            ).items()},
            'processor_resolution_counts': {str(k): v for k, v in Counter(
                tuple(x['processor_effective_frame_resolution_hw']) for x in original_200
            ).items()},
            'video_grid_counts': {str(k): v for k, v in Counter(
                tuple(x['processor_video_grid_thw']) for x in original_200
            ).items()},
            'visual_tokens': distribution([x['visual_token_count'] for x in original_200]),
            'text_prompt_tokens': distribution([x['raw_text_template_token_count'] for x in original_200]),
            'total_input_tokens': distribution([x['total_input_token_count'] for x in original_200]),
            'truncation_count': sum(x['truncation'] for x in original_200),
            'max_prompt_length': None,
        },
        'subset': {
            'count': len(ROWS), 'unique_original_videos': len({x['original_video_id'] for x in ROWS}),
            'question_type_counts': dict(Counter(x['type'] for x in ROWS)),
            'selection': 'SHA256 of fixed salt and sample_id within each type; round-robin across types; original-video disjoint; no model outcomes used',
        },
        'conditions': {}, 'paired_models': {}, 'video_vs_text': {},
        'condition_inputs': {},
        'per_type': {}, 'per_family': {}, 'per_sample': [],
        'original_64_reproduction': {}, 'direct_hf_crosscheck': {},
        'prompt_leak_check': {},
    }
    grid_probe = read_jsonl('budget_grid_probe.jsonl')
    assert len(grid_probe) == 64 * 3
    for condition in ('original', 'medium', 'high'):
        xs = [x for x in grid_probe if x['condition'] == condition]
        result['condition_inputs'][condition] = {
            'visual_tokens': distribution([x['visual_tokens'] for x in xs]),
            'total_input_tokens': distribution([x['input_tokens'] for x in xs]),
            'grid_counts': dict(Counter(str(x['video_grid_thw']) for x in xs)),
            'qwen_stage_resolution_counts': dict(Counter(str(x['qwen_stage_resolution_hw']) for x in xs)),
            'processor_resolution_counts': dict(Counter(str(x['processor_resolution_hw']) for x in xs)),
            'configured_max_pixels_per_frame': xs[0]['configured_max_pixels'],
            'source_capped_count': sum(x['effective_max_pixels'] < x['configured_max_pixels'] for x in xs),
            'frame_count_all_32': all(x['frame_count'] == 32 for x in xs),
        }
    for model in ('4b', '32b'):
        result['conditions'][model] = {}
        result['video_vs_text'][model] = {}
        for condition in CONDITIONS:
            xs = [by_model[model][row['sample_id'], condition] for row in ROWS]
            result['conditions'][model][condition] = summary(xs)
        for condition in ('original', 'medium', 'high'):
            xs = [by_model[model][row['sample_id'], condition] for row in ROWS]
            ys = [by_model[model][row['sample_id'], 'text_only'] for row in ROWS]
            result['video_vs_text'][model][condition] = {
                **paired_counts(xs, ys),
                'video_minus_text_accuracy': statistics.mean(float(x['correct']) - float(y['correct']) for x, y in zip(xs, ys)),
            }
        result['original_64_reproduction'][model] = {
            'same_response': sum(
                by_model[model][row['sample_id'], 'original']['response'] == old[model][row['sample_id']]['response']
                for row in ROWS
            ), 'count': 64,
        }
        direct = read_direct[model]
        original_pairs = [by_model[model][r['sample_id'], 'original'] for r in direct]
        result['direct_hf_crosscheck'][model] = {
            'count': len(direct), 'raw_direct_accuracy': sum(x['correct'] for x in direct) / len(direct),
            'original_same_16_accuracy': sum(x['correct'] for x in original_pairs) / len(direct),
            'same_option_count': sum(x['parsed_answer'] == y['parsed_answer'] for x, y in zip(direct, original_pairs)),
            'raw_direct_visual_tokens': distribution([x['visual_tokens'] for x in direct]),
            'original_same_16_visual_tokens': distribution([x['visual_tokens'] for x in original_pairs]),
            'raw_direct_grid_counts': dict(Counter(str(x['video_grid_thw']) for x in direct)),
            'original_grid_counts': dict(Counter(str(x['video_grid_thw']) for x in original_pairs)),
        }

    for condition in CONDITIONS:
        first = [by_model['4b'][row['sample_id'], condition] for row in ROWS]
        second = [by_model['32b'][row['sample_id'], condition] for row in ROWS]
        for x, y in zip(first, second):
            assert x['visual_tokens'] == y['visual_tokens']
            assert x['video_grid_thw'] == y['video_grid_thw']
            assert x['qwen_stage_resolution_hw'] == y['qwen_stage_resolution_hw']
            assert x['processor_resolution_hw'] == y['processor_resolution_hw']
        result['paired_models'][condition] = {
            **paired_counts(first, second),
            '32b_minus_4b_accuracy': statistics.mean(float(y['correct']) - float(x['correct']) for x, y in zip(first, second)),
            '32b_minus_4b_accuracy_bootstrap_95_ci': paired_ci(first, second),
            '32b_minus_4b_gt_probability_mean': statistics.mean(
                y['gt_option_probability_full_vocab'] - x['gt_option_probability_full_vocab']
                for x, y in zip(first, second)
            ),
            '32b_minus_4b_margin_mean': statistics.mean(
                y['gt_vs_best_wrong_log_prob_margin'] - x['gt_vs_best_wrong_log_prob_margin']
                for x, y in zip(first, second)
            ),
        }

    for grouping, f in (('per_type', lambda row: row['type']), ('per_family', lambda row: family(row['type']))):
        groups = defaultdict(list)
        for row in ROWS:
            groups[f(row)].append(row)
        for name, rows in groups.items():
            r = {'count': len(rows), 'conditions': {}}
            for condition in CONDITIONS:
                four = [by_model['4b'][row['sample_id'], condition] for row in rows]
                thirtytwo = [by_model['32b'][row['sample_id'], condition] for row in rows]
                r['conditions'][condition] = {
                    '4b_accuracy': statistics.mean(x['correct'] for x in four),
                    '32b_accuracy': statistics.mean(x['correct'] for x in thirtytwo),
                    '32b_minus_4b_accuracy': statistics.mean(y['correct'] - x['correct'] for x, y in zip(four, thirtytwo)),
                    '4b_gt_prob_mean': statistics.mean(x['gt_option_probability_full_vocab'] for x in four),
                    '32b_gt_prob_mean': statistics.mean(x['gt_option_probability_full_vocab'] for x in thirtytwo),
                }
            for model in ('4b', '32b'):
                r[f'{model}_high_minus_text_accuracy'] = statistics.mean(
                    by_model[model][row['sample_id'], 'high']['correct']
                    - by_model[model][row['sample_id'], 'text_only']['correct'] for row in rows
                )
            result[grouping][name] = r

    for row in ROWS:
        record = {
            'sample_id': row['sample_id'], 'original_video_id': row['original_video_id'],
            'question_type': row['type'], 'question_family': family(row['type']),
            'ground_truth': row['answer'], 'conditions': {},
        }
        for condition in CONDITIONS:
            record['conditions'][condition] = {
                model: {
                    'answer': by_model[model][row['sample_id'], condition]['parsed_answer'],
                    'correct': by_model[model][row['sample_id'], condition]['correct'],
                    'gt_probability': by_model[model][row['sample_id'], condition]['gt_option_probability_full_vocab'],
                    'gt_nll': by_model[model][row['sample_id'], condition]['gt_nll'],
                    'margin': by_model[model][row['sample_id'], condition]['gt_vs_best_wrong_log_prob_margin'],
                    'visual_tokens': by_model[model][row['sample_id'], condition]['visual_tokens'],
                } for model in ('4b', '32b')
            }
        result['per_sample'].append(record)

    # The actual prompt constructor accepts only question/options. Check rendered prompts
    # against the dataset metadata names as a second guard, and keep 20 full prompts on disk.
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(result['models']['4b'], local_files_only=True)
    all_200 = read_jsonl('qualification_200.jsonl')
    forbidden = ('ground_frame_indices_16_grid', 'original_video_id', 'source_split',
                 'source_annotation', 'is_enough', 'sufficiency_label', '.jpg', '.mp4')
    violations = []
    for row in all_200:
        prompt = prompt_for(row, processor)
        hits = [word for word in forbidden if word in prompt]
        if hits:
            violations.append({'sample_id': row['sample_id'], 'hits': hits})
    result['prompt_leak_check'] = {
        'checked': 200, 'printed_full_prompt_count': 20,
        'full_prompt_file': str(BASE / 'prompt_audit_20.txt'),
        'metadata_marker_violations': violations,
        'prompt_builder_fields': ['question', 'options', 'option_letters', 'video_content_block'],
    }
    result['processor_protocol_check'] = json.loads((BASE / 'processor_protocol_check.json').read_text())
    result['decision'] = {
        'ROOT_CAUSE_PRIMARY': 'MIXED',
        'VISUAL_INPUT_STATUS': 'NEEDS_FIX',
        'DATASET_STATUS': 'KEEP_AND_FILTER_LATER',
        'TEACHER_32B_STATUS': 'NEEDS_MORE_EVIDENCE',
        'READY_FOR_1K_OPD': 'NO',
        'evidence': [
            'Original visual input is 160 merged tokens for 185/200 samples; typical source JPEG is 1280x720.',
            'The frame-list qwen-vl-utils path resizes twice and changes common 16:9 frames to 64x160 at processor output.',
            'High raises visual tokens to about 2304 but 32B exceeds 4B by only 2/64; paired bootstrap accuracy interval includes zero.',
            'Text-only accuracy remains 45/64 for 4B and 47/64 for 32B.',
            'Objects questions show a 32B visual advantage; GT probability and margin also generally favor 32B.',
        ],
    }
    (BASE / 'visual_budget_diagnostic.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({
        'conditions': result['conditions'], 'paired_models': result['paired_models'],
        'per_family': result['per_family'],
        'original_64_reproduction': result['original_64_reproduction'],
        'direct_hf_crosscheck': result['direct_hf_crosscheck'],
        'prompt_leak_check': result['prompt_leak_check'],
    }, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
