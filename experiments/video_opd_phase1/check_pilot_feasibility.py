"""Check proposed 1k quotas against the frozen train pool without saving a selection."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (BASE / 'train_pool_video_disjoint.jsonl').read_text().splitlines()]
type_counts = Counter(x['type'] for x in rows)
by_type = defaultdict(list)
for row in rows:
    by_type[row['type']].append(row)
for group in by_type.values():
    group.sort(key=lambda row: hashlib.sha256(
        f"pilot-feasibility:{row['sample_id']}".encode()).hexdigest())

CAP = 8
BASE_QUOTA = 400
BOOSTS = {
    'Action Count': 130,
    'Objects Spatial Relation': 60,
    'Objects Spatial Location': 50,
    'Objects Existence': 70,
    'Event Sequence': 50,
}
quotas = {type_name: math.floor(BASE_QUOTA * n / len(rows)) for type_name, n in type_counts.items()}
remainder = BASE_QUOTA - sum(quotas.values())
for type_name in sorted(type_counts, key=lambda t: (
    -(BASE_QUOTA * type_counts[t] / len(rows) - quotas[t]), t
))[:remainder]:
    quotas[type_name] += 1

used = set()
per_video = Counter()
per_type = Counter()


def take(type_name, amount):
    taken = 0
    for row in by_type[type_name]:
        if row['sample_id'] in used or per_video[row['original_video_id']] >= CAP:
            continue
        used.add(row['sample_id'])
        per_video[row['original_video_id']] += 1
        per_type[type_name] += 1
        taken += 1
        if taken == amount:
            break
    return taken


base_actual = {t: take(t, quotas[t]) for t in sorted(type_counts, key=lambda t: (type_counts[t], t))}
boost_actual = {t: take(t, n) for t, n in BOOSTS.items()}
coverage = 0
while len(used) < 1000:
    # Favor remaining types with least representation relative to their pool size.
    for type_name in sorted(type_counts, key=lambda t: (per_type[t] / type_counts[t], per_type[t], t)):
        if take(type_name, 1):
            coverage += 1
            break
    else:
        raise RuntimeError('Unable to fill 1,000 rows under proposed cap')

assert sum(base_actual.values()) == BASE_QUOTA
assert boost_actual == BOOSTS
assert coverage == 240
assert len(used) == 1000 and max(per_video.values()) <= CAP
report = {
    'proposed_only_no_manifest_written': True,
    'target_qa': 1000,
    'base_actual': sum(base_actual.values()),
    'boost_actual': boost_actual,
    'coverage_actual': coverage,
    'unique_original_videos': len(per_video),
    'max_qa_per_original_video': max(per_video.values()),
    'type_counts_after_dry_run': dict(per_type),
}
(BASE / 'pilot_quota_feasibility.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'type_counts_after_dry_run'}, indent=2))
