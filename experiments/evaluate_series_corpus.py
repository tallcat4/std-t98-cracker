"""Random-code blind regression on a pre-split ThumbDV NPZ corpus.

NPZ fields: bits=(frames,7) packed ThumbDV bits; metadata=JSON with split and
natural_frames. Appended silence is excluded. No corpus audio is published.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from std_t98_cracker.burst import unpack49
from std_t98_cracker.enhanced_sequence import CandidateSequence
from std_t98_cracker.series_solver import SeriesSolver
from evaluate_series import solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('--split', choices=('train', 'eval'), default='eval')
    parser.add_argument('--method', choices=('linear', 'guided-ai'), default='linear')
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--max-bursts', type=int, default=37, help='37 bursts = 2.96 s of natural speech')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--margin', type=float)
    parser.add_argument('--json', required=True, type=Path)
    args = parser.parse_args()
    if args.max_bursts < 1 or args.json.exists():
        raise ValueError('Need positive max-bursts and a new output path')
    solver, verifier = SeriesSolver(), None
    threshold, margin = 40., 20.
    if args.method == 'guided-ai':
        from std_t98_cracker.enhanced import HeavyScorer
        from std_t98_cracker.guided_ai import GuidedAISolver
        scorer = HeavyScorer(device='auto')
        verifier = scorer.describe()
        solver = GuidedAISolver(solver, scorer)
        threshold, margin = 20., 6.
    threshold = threshold if args.threshold is None else args.threshold
    margin = margin if args.margin is None else args.margin
    rng, rows = np.random.default_rng(args.seed), []
    for path in sorted(args.corpus.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['split'] != args.split:
                continue
            n = min(int(metadata['natural_frames']) // 4, args.max_bursts)
            if not n:
                continue
            plain = unpack49(data['bits'][:n * 4].reshape(-1, 4, 7))
        series, key = int(rng.integers(0, 16)), int(rng.integers(1, 32768))
        if len(rows) % 17 == 0:
            series, key = 0, 0
        encrypted = plain ^ CandidateSequence(series).batch(np.array([key]))[0]
        # The solver sees only encrypted bits and FEC validity, not these labels.
        result = solve(encrypted, np.ones((n, 4), dtype=bool), solver, threshold, margin)
        def outcome(decision):
            if decision is None or decision['key'] is None:
                return 'unresolved'
            return 'correct' if (decision['series'], decision['key']) == (series, key) else 'wrong'
        rows.append(dict(input=path.name, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         expected_series=series, expected_key=key, **result,
                         final_outcome=outcome(result['final']), first_outcome=outcome(result['first_confirmation']),
                         any_wrong_confirmation=any(outcome(d) == 'wrong' for d in result['accepted_pairs'])))
        if len(rows) % 10 == 0:
            print(len(rows), 'evaluated', flush=True)
    if not rows:
        raise ValueError('No matching corpus entries')
    summary = {phase: {status: sum(r[f'{phase}_outcome'] == status for r in rows)
                       for status in ('correct', 'unresolved', 'wrong')} for phase in ('final', 'first')}
    summary['any_wrong_confirmation'] = sum(r['any_wrong_confirmation'] for r in rows)
    summary['correct_first_within_seconds'] = {str(seconds): sum(
        r['first_outcome'] == 'correct' and r['first_confirmation']['traffic_seconds'] <= seconds
        for r in rows) for seconds in (1, 3)}
    root = Path(__file__).resolve().parents[1]
    report = dict(schema='blind-series-corpus-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  method=args.method, split=args.split, seed=args.seed, max_bursts=args.max_bursts,
                  threshold=threshold, margin=margin, labels_used_by_solver=False,
                  ai_used=verifier is not None, verifier=verifier, summary=summary, captures=rows,
                  source_sha256={name: hashlib.sha256((root/'std_t98_cracker'/name).read_bytes()).hexdigest()
                                 for name in ('series_solver.py', 'guided_ai.py', 'enhanced_sequence.py', 'enhanced.py')})
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
