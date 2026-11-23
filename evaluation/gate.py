"""Nonzero exit on missing/failed metrics. Usage: python -m evaluation.gate [file]"""
import json
import math
import sys
from pathlib import Path

THRESHOLDS = {'recall_at_10': 0.80, 'mrr_at_10': 0.60, 'ndcg_at_10': 0.65}

def check(results, min_questions=20):
    failures = []
    if results.get('num_questions', 0) < min_questions:
        failures.append(f'Need at least {min_questions} labeled questions')
    metrics = results.get('hybrid_reranked', {})
    for key, threshold in THRESHOLDS.items():
        value = metrics.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < threshold:
            failures.append(f'{key}: {value} < {threshold}')
    return failures

if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('evaluation/results.json')
    if not path.exists():
        sys.exit(f'FAIL: missing evaluation results: {path}')
    failures = check(json.loads(path.read_text()))
    if failures:
        sys.exit('FAIL:\n' + '\n'.join(failures))
    print('PASS: retrieval quality gate')
