"""Reproduce diagnostic_64 from frozen qualification_200 without model outcomes."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (BASE / 'qualification_200.jsonl').read_text().splitlines()]
by_type = defaultdict(list)
for row in rows:
    by_type[row['type']].append(row)
for group in by_type.values():
    group.sort(key=lambda row: hashlib.sha256(
        f"visual-budget-v1:{row['sample_id']}".encode()).hexdigest())

selected = []
seen_videos = set()
rank = 0
while len(selected) < 64:
    progress = False
    for type_name in sorted(by_type):
        group = by_type[type_name]
        if rank >= len(group):
            continue
        row = group[rank]
        if row['original_video_id'] in seen_videos:
            continue
        selected.append(row)
        seen_videos.add(row['original_video_id'])
        progress = True
        if len(selected) == 64:
            break
    if not progress:
        raise RuntimeError('Fewer than 64 distinct original videos')
    rank += 1

output = BASE / 'diagnostic_64.jsonl'
rendered = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
if output.exists() and output.read_text() != rendered:
    raise RuntimeError('Existing fixed manifest differs from reproducible selection')
output.write_text(rendered)
print(f'{len(selected)} rows, {len(seen_videos)} distinct original videos, {len(by_type)} question types')
