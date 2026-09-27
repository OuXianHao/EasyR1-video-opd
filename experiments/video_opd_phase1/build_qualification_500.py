"""Freeze a type-stratified, original-video-disjoint evaluation set and train pool."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/xianhao/datasets/rlsd_v_frames32_dataset/rlsd_v_frames32_dataset')
OUT = Path(__file__).resolve().parent
SOURCES = {
    'legacy_train': ROOT / 'rlsd_v_frames32_train_14311.jsonl',
    'legacy_val': ROOT / 'rlsd_v_frames32_val200.jsonl',
}
SALT = 'video-opd-phase1-7-qualification500-v1'
TARGET = 500


def digest(value):
    return hashlib.sha256(f'{SALT}:{value}'.encode()).hexdigest()


def emit(row, source_split):
    stem = row['video_stem']
    original = stem.split('.', 1)[0]
    result = dict(row)
    result.update({
        'sample_id': row['question_id'], 'video_id': stem,
        'original_video_id': original, 'source_split': source_split,
        'source_frame_paths': row['frame_paths'],
        'frame_paths': [str(ROOT / 'frames_32' / stem / Path(p).name) for p in row['frame_paths']],
    })
    return result


def write_jsonl(name, rows):
    (OUT / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    rows = [emit(json.loads(line), split) for split, path in SOURCES.items()
            for line in path.read_text().splitlines()]
    assert len(rows) == 14511
    originals = {r['original_video_id'] for r in rows}
    assert len(originals) == 760
    by_type = defaultdict(lambda: defaultdict(list))
    counts = Counter(r['type'] for r in rows)
    for row in rows:
        by_type[row['type']][row['original_video_id']].append(row)
    exact = {type_name: TARGET * count / len(rows) for type_name, count in counts.items()}
    quotas = {type_name: math.floor(value) for type_name, value in exact.items()}
    for type_name in sorted(counts, key=lambda t: (-(exact[t] - quotas[t]), t))[:TARGET - sum(quotas.values())]:
        quotas[type_name] += 1
    assert sum(quotas.values()) == TARGET and min(quotas.values()) >= 1
    selected = []
    used = set()
    # Reserve rare types first; choices within a type are hash ordered and never
    # depend on Student/Teacher outcomes, confidence, or any prior evaluation.
    for type_name in sorted(counts, key=lambda t: (len(by_type[t]), t)):
        candidates = sorted(by_type[type_name], key=lambda original: digest(original))
        for original in candidates:
            if original in used:
                continue
            options = by_type[type_name][original]
            chosen = min(options, key=lambda row: digest(row['sample_id']))
            selected.append(chosen)
            used.add(original)
            if sum(row['type'] == type_name for row in selected) == quotas[type_name]:
                break
    if len(selected) < TARGET:
        remaining = sorted(originals - used, key=digest)
        by_original = defaultdict(list)
        for row in rows:
            by_original[row['original_video_id']].append(row)
        for original in remaining[:TARGET - len(selected)]:
            chosen = min(by_original[original], key=lambda row: digest(row['sample_id']))
            selected.append(chosen)
            used.add(original)
    assert len(selected) == TARGET and len(used) == TARGET
    selected.sort(key=lambda row: digest(row['original_video_id']))
    train_pool = [row for row in rows if row['original_video_id'] not in used]
    assert not used & {row['original_video_id'] for row in train_pool}
    assert len(rows) == len(train_pool) + sum(row['original_video_id'] in used for row in rows)
    write_jsonl('qualification_500.jsonl', selected)
    write_jsonl('train_pool_video_disjoint.jsonl', train_pool)
    prepared = []
    for row in selected:
        letters = ', '.join(chr(ord('A') + i) for i in range(len(row['options'])))
        prepared.append({
            'prompt': '<video>\n' + row['question'] + '\n' + '\n'.join(row['options'])
            + '\nAnswer with exactly one option letter: ' + letters + '.',
            'answer': row['answer'], 'videos': [row['frame_paths']],
            'sample_id': row['sample_id'], 'video_id': row['video_id'],
            'original_video_id': row['original_video_id'], 'type': row['type'],
        })
    write_jsonl('qualification_500_easyr1.jsonl', prepared)
    result = {
        'selection_salt': SALT, 'selection_uses_model_outcomes': False,
        'source_count': len(rows), 'source_original_videos': len(originals),
        'qualification_count': len(selected), 'qualification_original_videos': len(used),
        'qualification_type_quota': quotas,
        'qualification_type_actual': dict(Counter(row['type'] for row in selected)),
        'qualification_source_split': dict(Counter(row['source_split'] for row in selected)),
        'train_pool_count': len(train_pool),
        'train_pool_original_videos': len({row['original_video_id'] for row in train_pool}),
        'train_pool_type_counts': dict(Counter(row['type'] for row in train_pool)),
        'full_type_counts': dict(counts),
        'disjoint_original_video_count': len(used & {row['original_video_id'] for row in train_pool}),
    }
    (OUT / 'qualification_500_split_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
