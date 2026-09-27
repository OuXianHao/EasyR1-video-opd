"""Summarize the actual vLLM/FSDP/Teacher sampled-token diagnostic."""

import json
import statistics
import sys
from pathlib import Path


def stats(values):
    if not values:
        return None
    values = sorted(values)
    def quantile(p):
        x = p * (len(values) - 1)
        lo = int(x)
        return values[lo] * (1 - (x - lo)) + values[min(lo + 1, len(values) - 1)] * (x - lo)
    return {
        'count': len(values), 'mean': statistics.fmean(values),
        'std': statistics.pstdev(values), 'min': values[0], 'max': values[-1],
        'positive_fraction': sum(v > 0 for v in values) / len(values),
        'negative_fraction': sum(v < 0 for v in values) / len(values),
        **{f'p{int(p*100)}': quantile(p) for p in (.01, .05, .25, .5, .75, .95, .99)},
    }


def main(run_dir):
    path = Path(run_dir)
    diagnostic = json.loads((path / 'diagnostics/step_1.json').read_text())
    manifest = {json.loads(s)['sample_id']: json.loads(s) for s in
                (Path(__file__).parent / 'qualification_32.jsonl').read_text().splitlines()}
    all_tokens = []
    letters, specials = [], []
    rollout_results = []
    for sample in diagnostic['samples']:
        item = manifest[sample['sample_id']]
        allowed = {chr(ord('A') + i) for i in range(len(item['options']))}
        tokens = sample['tokens']
        response = ''.join(t['token_text'] for t in tokens if not t['token_text'].startswith('<|')).strip()
        rollout_results.append({
            'sample_id': sample['sample_id'], 'response': response, 'answer': item['answer'],
            'format_valid': response in allowed, 'correct': response == item['answer'],
        })
        for token in tokens:
            enriched = {'sample_id': sample['sample_id'], **token}
            all_tokens.append(enriched)
            if token['token_text'].startswith('<|') and token['token_text'].endswith('|>'):
                specials.append(token['delta'])
            elif token['token_text'].strip() in allowed and len(token['token_text'].strip()) == 1:
                letters.append(token['delta'])
    all_tokens.sort(key=lambda t: t['delta'])
    summary = {
        'run_dir': str(path), 'sample_count': len(diagnostic['samples']),
        'delta': stats([t['delta'] for t in all_tokens]),
        'option_letter_delta': stats(letters), 'special_token_delta': stats(specials),
        'reasoning_token_delta': None,
        'vllm_fsdp_abs_diff': diagnostic.get('vllm_fsdp_abs_diff'),
        'largest_negative_tokens': all_tokens[:10],
        'largest_positive_tokens': all_tokens[-10:][::-1],
        'rollout_results': rollout_results,
        'rollout_format_valid': sum(r['format_valid'] for r in rollout_results),
        'rollout_correct': sum(r['correct'] for r in rollout_results),
    }
    output = Path(__file__).parent / 'teacher_forcing_qualification_summary.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ('largest_negative_tokens', 'largest_positive_tokens', 'rollout_results')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
