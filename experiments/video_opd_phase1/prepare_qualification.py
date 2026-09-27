"""Validate frozen frames and make EasyR1 input without answer leakage."""

import json
import sys
from pathlib import Path

from PIL import Image

OUT = Path(__file__).resolve().parent


def main(n):
    manifest = OUT / f'qualification_{n}.jsonl'
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    assert len(rows) == n
    assert len({r['original_video_id'] for r in rows}) == n
    prepared = []
    for row in rows:
        frames = row['frame_paths']
        assert len(frames) == 32, row['sample_id']
        assert [int(Path(p).name.split('_', 1)[0]) for p in frames] == list(range(32)), row['sample_id']
        for path in frames:
            with Image.open(path) as image:
                image.verify()
        prompt = (
            '<video>\n' + row['question'] + '\n' + '\n'.join(row['options'])
            + '\nAnswer with exactly one option letter: '
            + ', '.join(chr(ord('A') + i) for i in range(len(row['options']))) + '.'
        )
        prepared.append({
            'prompt': prompt, 'answer': row['answer'], 'videos': [frames],
            'sample_id': row['sample_id'], 'video_id': row['video_id'],
            'original_video_id': row['original_video_id'],
        })
    path = OUT / f'qualification_{n}_easyr1.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in prepared))
    print(f'{n} samples, {32*n} decoded frames, {n} unique original videos: {path}')


if __name__ == '__main__':
    main(int(sys.argv[1]))
