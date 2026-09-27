"""Audit the local, already sampled 32-frame VideoQA package and freeze a subset."""

import collections
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/xianhao/datasets/rlsd_v_frames32_dataset/rlsd_v_frames32_dataset')
OUT = Path(__file__).resolve().parent
SPLITS = {
    'train': ROOT / 'rlsd_v_frames32_train_14311.jsonl',
    'val': ROOT / 'rlsd_v_frames32_val200.jsonl',
}


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def main():
    data = {name: read_jsonl(path) for name, path in SPLITS.items()}
    records = [(split, row) for split, rows in data.items() for row in rows]
    counts = collections.Counter()
    videos = collections.defaultdict(set)
    originals = collections.defaultdict(set)
    qa = collections.defaultdict(list)
    clips = collections.defaultdict(list)
    key_agreement = collections.Counter()
    for split, row in records:
        vid = row['video_stem']
        original = vid.split('.', 1)[0]
        videos[vid].add(split)
        originals[original].add(split)
        qa[(original, row['question'].strip().casefold(), tuple(row['options']))].append((split, row['question_id']))
        clips[tuple(Path(p).name for p in row['frame_paths']) + (vid,)].append((split, row['question_id']))
        paths = [ROOT / 'frames_32' / vid / Path(p).name for p in row['frame_paths']]
        if len(paths) != 32:
            counts['not_32_frames'] += 1
        if not (ROOT / 'frames_32' / vid).is_dir():
            counts['missing_frame_dir'] += 1
        counts['missing_frames'] += sum(not p.is_file() for p in paths)
        counts['missing_annotation'] += not (ROOT / 'ground_jsons_trans_improve_trans' / f"{row['question_id']}.json").is_file()
        counts['legacy_paths_missing'] += sum(not Path(p).exists() for p in row['frame_paths'])
        source = ROOT / 'ground_jsons_trans_improve_trans' / f"{row['question_id']}.json"
        if source.is_file():
            old = json.loads(source.read_text())
            key_agreement['source_has_ground_frames'] += old.get('ground_frames') is not None
            key_agreement['source_is_enough_true'] += old.get('is_enough_trans') is True
            key_agreement['key_matches_improved'] += row['ground_frame_indices'] == old.get('ground_frames_trans_improve')
            key_agreement['key_matches_trans'] += row['ground_frame_indices'] == old.get('ground_frames_trans')
            key_agreement['key_set_matches_improved'] += set(row['ground_frame_indices']) == set(old.get('ground_frames_trans_improve', []))
    train = data['train']
    # Stable hash ordering; choose one question for each original source video.
    ordered = sorted(train, key=lambda r: hashlib.sha256(r['question_id'].encode()).hexdigest())
    chosen, used = [], set()
    for row in ordered:
        original = row['video_stem'].split('.', 1)[0]
        if original in used:
            continue
        used.add(original)
        chosen.append(row)
        if len(chosen) == 200:
            break
    def manifest(row):
        vid = row['video_stem']
        return {
            'sample_id': row['question_id'], 'video_id': vid,
            'original_video_id': vid.split('.', 1)[0],
            'question': row['question'], 'options': row['options'], 'answer': row['answer'],
            'frame_paths': [str(ROOT / 'frames_32' / vid / Path(p).name) for p in row['frame_paths']],
            'ground_frame_indices_16_grid': row['ground_frame_indices'],
            'type': row['type'], 'category': row['category'], 'source_split': 'train',
        }
    for n in (32, 200):
        path = OUT / f'qualification_{n}.jsonl'
        path.write_text(''.join(json.dumps(manifest(r), ensure_ascii=False) + '\n' for r in chosen[:n]))
    report = {
        'dataset_identity': 'RLSD-V frames32; BSPO provenance not found in local files',
        'paths': {k: str(v) for k, v in SPLITS.items()},
        'split_records': {k: len(v) for k, v in data.items()},
        'option_count_distribution': dict(collections.Counter(len(r['options']) for _, r in records)),
        'total_records': len(records),
        'unique_clip_videos': len(videos),
        'unique_original_videos_by_prefix': len(originals),
        'split_overlap_clip_videos': sum(len(x) > 1 for x in videos.values()),
        'split_overlap_original_videos': sum(len(x) > 1 for x in originals.values()),
        'duplicate_qa_extra': sum(len(x)-1 for x in qa.values()),
        'duplicate_qa_cross_split': sum(len({s for s, _ in x}) > 1 for x in qa.values()),
        'duplicate_32_frame_clip_extra': sum(len(x)-1 for x in clips.values()),
        'duplicate_32_frame_clip_cross_split': sum(len({s for s, _ in x}) > 1 for x in clips.values()),
        'counts': dict(counts), 'key_annotation_source_checks': dict(key_agreement),
        'qualification_32_unique_originals': len({r['video_stem'].split('.',1)[0] for r in chosen[:32]}),
        'qualification_200_unique_originals': len({r['video_stem'].split('.',1)[0] for r in chosen}),
        'example_redacted': {
            'video_id': train[0]['video_stem'], 'question': train[0]['question'],
            'options': train[0]['options'], 'answer': train[0]['answer'],
            'frame_count': len(train[0]['frame_paths']),
            'frame_name_first_last': [Path(train[0]['frame_paths'][0]).name, Path(train[0]['frame_paths'][-1]).name],
            'key_frame_indices': train[0]['ground_frame_indices'],
            'metadata': {k: train[0][k] for k in ('question_id', 'type', 'category')},
        },
    }
    (OUT / 'data_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
