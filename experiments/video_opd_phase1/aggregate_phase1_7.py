"""Aggregate held-out 500-row Video OPD Teacher qualification and tag analysis."""

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
N = 500


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(x)
    return values[lo] * (1 - (x - lo)) + values[min(lo + 1, len(values) - 1)] * (x - lo)


def dist(values):
    return {
        'mean': statistics.mean(values), 'median': quantile(values, .5),
        'p25': quantile(values, .25), 'p75': quantile(values, .75),
        'min': min(values), 'max': max(values),
    }


def family(type_name):
    if type_name.startswith('Objects '): return 'Objects'
    if type_name.startswith('Action '): return 'Action'
    if type_name.startswith('Event '): return 'Event'
    if 'Reasoning' in type_name: return 'Reasoning'
    if 'Description' in type_name: return 'Description'
    return 'Other'


def load_outputs(model, condition):
    if condition == 'text_only':
        files = [BASE / f'phase1_7_{model}_high_text_only.jsonl']
    else:
        files = sorted(BASE.glob(f'phase1_7_{model}_high_video_p*.jsonl'))
    rows = [r for file in files for r in read_jsonl(file)]
    if len(rows) != N or len({r['sample_id'] for r in rows}) != N:
        raise RuntimeError(f'{model}/{condition} incomplete or duplicate: {len(rows)} rows in {files}')
    return {r['sample_id']: r for r in rows}, [str(f) for f in files]


def model_summary(rows):
    return {
        'n': len(rows), 'correct': sum(r['correct'] for r in rows),
        'accuracy': statistics.mean(r['correct'] for r in rows),
        'format_valid_rate': statistics.mean(r['format_valid'] for r in rows),
        'mean_gt_logprob': statistics.mean(r['gt_log_prob_full_vocab'] for r in rows),
        'mean_gt_probability': statistics.mean(r['gt_probability_full_vocab'] for r in rows),
        'mean_gt_nll': statistics.mean(r['gt_nll'] for r in rows),
        'mean_margin': statistics.mean(r['gt_vs_best_wrong_margin'] for r in rows),
    }


def paired(a, b):
    return {
        'both_correct': sum(x and y for x, y in zip(a, b)),
        'first_only': sum(x and not y for x, y in zip(a, b)),
        'second_only': sum(not x and y for x, y in zip(a, b)),
        'both_wrong': sum(not x and not y for x, y in zip(a, b)),
    }


def bootstrap_gap(a, b):
    differences = [int(y) - int(x) for x, y in zip(a, b)]
    rng = random.Random(1707)
    means = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000)]
    return [quantile(means, .025), quantile(means, .975)]


def group_summary(rows):
    n = len(rows)
    video_4b = [r['models']['4b']['video'] for r in rows]
    video_32b = [r['models']['32b']['video'] for r in rows]
    text_4b = [r['models']['4b']['text_only'] for r in rows]
    text_32b = [r['models']['32b']['text_only'] for r in rows]
    return {
        'n': n,
        '4b_video_acc': statistics.mean(x['correct'] for x in video_4b),
        '32b_video_acc': statistics.mean(x['correct'] for x in video_32b),
        '4b_text_acc': statistics.mean(x['correct'] for x in text_4b),
        '32b_text_acc': statistics.mean(x['correct'] for x in text_32b),
        '32b_minus_4b_video_acc': statistics.mean(
            int(y['correct']) - int(x['correct']) for x, y in zip(video_4b, video_32b)
        ),
        '4b_video_minus_text_acc': statistics.mean(
            int(x['correct']) - int(y['correct']) for x, y in zip(video_4b, text_4b)
        ),
        '32b_video_minus_text_acc': statistics.mean(
            int(x['correct']) - int(y['correct']) for x, y in zip(video_32b, text_32b)
        ),
        'mean_visual_gain_4b': statistics.mean(r['visual_gain_4b'] for r in rows),
        'mean_visual_gain_32b': statistics.mean(r['visual_gain_32b'] for r in rows),
        '4b_gt_nll': statistics.mean(x['gt_nll'] for x in video_4b),
        '32b_gt_nll': statistics.mean(x['gt_nll'] for x in video_32b),
        '4b_margin': statistics.mean(x['gt_vs_best_wrong_margin'] for x in video_4b),
        '32b_margin': statistics.mean(x['gt_vs_best_wrong_margin'] for x in video_32b),
        'analysis_tag_counts': dict(Counter(tag for r in rows for tag in r['analysis_tags'])),
        'video_paired': paired([x['correct'] for x in video_4b], [x['correct'] for x in video_32b]),
    }


def main():
    manifest = read_jsonl(BASE / 'qualification_500.jsonl')
    train_pool = read_jsonl(BASE / 'train_pool_video_disjoint.jsonl')
    assert len(manifest) == N
    data = {}
    paths = {}
    for model in ('4b', '32b'):
        data[model] = {}
        for condition in ('video', 'text_only'):
            data[model][condition], paths[f'{model}_{condition}'] = load_outputs(model, condition)
    ids = [r['sample_id'] for r in manifest]
    assert all(set(d.keys()) == set(ids) for model in data.values() for d in model.values())
    tagged = []
    for row in manifest:
        sample_id = row['sample_id']
        a = data['4b']['video'][sample_id]
        b = data['32b']['video'][sample_id]
        at = data['4b']['text_only'][sample_id]
        bt = data['32b']['text_only'][sample_id]
        assert a['visual_tokens'] == b['visual_tokens']
        assert a['video_grid_thw'] == b['video_grid_thw']
        assert a['processed_resolution_hw'] == b['processed_resolution_hw']
        gain_a = a['gt_log_prob_full_vocab'] - at['gt_log_prob_full_vocab']
        gain_b = b['gt_log_prob_full_vocab'] - bt['gt_log_prob_full_vocab']
        tags = []
        if all(x['correct'] for x in (a, b, at, bt)) and abs(gain_a) <= .5 and abs(gain_b) <= .5:
            tags.append('easy_text_prior_heavy')
        if ((a['correct'] and not at['correct'] and gain_a > .25)
                or (b['correct'] and not bt['correct'] and gain_b > .25)
                or (gain_a > .5 and gain_b > .5)):
            tags.append('video_dependent')
        if not a['correct'] and b['correct'] and b['gt_probability_full_vocab'] > a['gt_probability_full_vocab']:
            tags.append('teacher_resolvable')
        if not a['correct'] and not b['correct'] and max(
            a['gt_probability_full_vocab'], b['gt_probability_full_vocab']
        ) < .5:
            tags.append('both_hard')
        tagged.append({
            'sample_id': sample_id, 'original_video_id': row['original_video_id'],
            'video_id': row['video_id'], 'question_type': row['type'],
            'category': row['category'], 'source_split': row['source_split'],
            'ground_truth': row['answer'], 'frame_count': 32,
            'video_grid_thw': a['video_grid_thw'],
            'visual_tokens': a['visual_tokens'],
            'source_resolution_wh': a['source_resolution_wh'],
            'processed_resolution_hw': a['processed_resolution_hw'],
            'visual_gain_4b': gain_a, 'visual_gain_32b': gain_b,
            'analysis_tags': tags or ['unclassified'],
            'models': {
                '4b': {'video': a, 'text_only': at},
                '32b': {'video': b, 'text_only': bt},
            },
        })
    (BASE / 'phase1_7_qualification_tagged.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in tagged)
    )
    type_groups = defaultdict(list)
    family_groups = defaultdict(list)
    for row in tagged:
        type_groups[row['question_type']].append(row)
        family_groups[family(row['question_type'])].append(row)
    result = {
        'models': {
            '4b': '/mnt/data/zhzhu/models/Qwen3-VL-4B-Instruct',
            '32b': '/home/xianhao/models/Qwen3-VL-32B-Instruct',
        },
        'qualification_manifest': str(BASE / 'qualification_500.jsonl'),
        'train_pool_manifest': str(BASE / 'train_pool_video_disjoint.jsonl'),
        'per_sample_tagged_file': str(BASE / 'phase1_7_qualification_tagged.jsonl'),
        'raw_result_files': paths,
        'split_audit': json.loads((BASE / 'qualification_500_split_audit.json').read_text()),
        'smokes': json.loads((BASE / 'phase1_7_smoke_summary.json').read_text()),
        'visual_budget': 'HIGH',
        'visual_tokens': dist([r['visual_tokens'] for r in tagged]),
        'source_resolution_counts': dict(Counter(str(r['source_resolution_wh']) for r in tagged)),
        'processed_resolution_counts': dict(Counter(str(r['processed_resolution_hw']) for r in tagged)),
        'video_grid_counts': dict(Counter(str(r['video_grid_thw']) for r in tagged)),
        'input_token_max': max(r['models'][m]['video']['input_tokens'] for r in tagged for m in ('4b','32b')),
        'model_metrics': {
            m: {c: model_summary([data[m][c][sample_id] for sample_id in ids])
                for c in ('video','text_only')}
            for m in ('4b','32b')
        },
        'visual_gain': {m: dist([r[f'visual_gain_{m}'] for r in tagged]) for m in ('4b','32b')},
        'video_paired_4b_32b': paired(
            [r['models']['4b']['video']['correct'] for r in tagged],
            [r['models']['32b']['video']['correct'] for r in tagged],
        ),
        'video_accuracy_gap_bootstrap_95_ci': bootstrap_gap(
            [r['models']['4b']['video']['correct'] for r in tagged],
            [r['models']['32b']['video']['correct'] for r in tagged],
        ),
        'video_vs_text_paired': {
            m: paired(
                [r['models'][m]['video']['correct'] for r in tagged],
                [r['models'][m]['text_only']['correct'] for r in tagged],
            ) for m in ('4b','32b')
        },
        'type_metrics': {t: group_summary(rows) for t, rows in type_groups.items()},
        'family_metrics': {t: group_summary(rows) for t, rows in family_groups.items()},
        'tag_counts': dict(Counter(tag for r in tagged for tag in r['analysis_tags'])),
        'tag_rules': {
            'easy_text_prior_heavy': 'both models correct with video and text, and |visual_gain| <= 0.5 for each',
            'video_dependent': 'either model video correct/text wrong with gain > 0.25, or both gains > 0.5',
            'teacher_resolvable': '4B video wrong, 32B video correct, 32B GT probability greater than 4B',
            'both_hard': 'both video wrong and both GT probabilities < 0.5',
        },
        'train_pool_distribution': {
            'qa_count': len(train_pool),
            'original_video_count': len({r['original_video_id'] for r in train_pool}),
            'type_counts': dict(Counter(r['type'] for r in train_pool)),
            'family_counts': dict(Counter(family(r['type']) for r in train_pool)),
        },
    }
    prior_diagnostic_videos = {
        row['original_video_id'] for row in read_jsonl(BASE / 'diagnostic_64.jsonl')
    }
    fresh = [row for row in tagged if row['original_video_id'] not in prior_diagnostic_videos]
    result['prior_diagnostic_overlap'] = {
        'overlap_count': len(tagged) - len(fresh), 'fresh_count': len(fresh),
        'selection_used_prior_outcomes': False,
        'fresh_only_metrics': group_summary(fresh),
        'fresh_only_video_accuracy_gap_bootstrap_95_ci': bootstrap_gap(
            [r['models']['4b']['video']['correct'] for r in fresh],
            [r['models']['32b']['video']['correct'] for r in fresh],
        ),
    }
    result['decision'] = {
        'PREPROCESSING_FIXED': 'YES',
        'RECOMMENDED_VISUAL_BUDGET': 'HIGH',
        'TEACHER_32B': 'PARTIALLY_QUALIFIED',
        'DATASET': 'KEEP_WITH_STRATIFIED_SAMPLING',
        'RECOMMENDED_1K_PILOT_COMPOSITION': {
            'pool': 'train_pool_video_disjoint.jsonl (4145 QA; 260 original videos)',
            'unique_qa_target': 1000,
            'max_qa_per_original_video': 8,
            'type_proportional_base': 400,
            'action_count': 130,
            'objects_spatial_relation': 60,
            'objects_spatial_location': 50,
            'objects_existence': 70,
            'event_sequence': 50,
            'other_type_coverage': 240,
            'feasibility_dry_run': '1000 distinct QA, 241 original videos, max 8 QA/video; no pilot manifest saved',
            'selection_note': 'Proposed quotas only; draw without replacement in the next turn and keep the qualification_500 evaluation fixed and outcome-independent.',
        },
        'READY_FOR_1K_OPD': 'YES',
        'interpretation': 'Engineering and held-out evaluation gates are met for a bounded research pilot; Teacher advantage is type-specific and uncertain overall.',
    }
    result['pilot_quota_feasibility'] = json.loads((BASE / 'pilot_quota_feasibility.json').read_text())
    result['train_pool_frame_integrity'] = json.loads((BASE / 'train_pool_frame_integrity.json').read_text())
    (BASE / 'phase1_7_report.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in (
        'model_metrics', 'visual_gain', 'video_paired_4b_32b',
        'video_accuracy_gap_bootstrap_95_ci', 'family_metrics', 'tag_counts',
    )}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
