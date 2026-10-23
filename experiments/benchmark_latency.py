"""Measure time to the first key with one more burst arriving every 80 ms."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from std_t98_cracker.linear import SilenceSolver
from std_t98_cracker.sequence import Sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'eval'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--solver-file', type=Path)
    parser.add_argument('--threshold', type=float, default=20)
    parser.add_argument('--margin', type=float, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Output exists')
    solver_file = Path(__file__).resolve().parents[1]/'std_t98_cracker/linear.py'
    cls = SilenceSolver
    if args.solver_file:
        solver_file = args.solver_file
        spec = importlib.util.spec_from_file_location('std_t98_cracker._latency_variant', solver_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = module.SilenceSolver
    solver, sequence = cls(), Sequence()
    plan = json.loads((args.corpus/'plan.json').read_text())
    rows = [row for row in plan if row['split'] == args.split]
    results = []
    for row in rows:
        with np.load(args.corpus/(row['id']+'.npz')) as data:
            meta = json.loads(str(data['metadata']))
            natural = meta['natural_frames']
            bits = np.unpackbits(data['bits'], axis=-1)[:natural, :49]
            pcm = data['pcm'][:natural*160].reshape(-1, 160).astype(np.float32)
        energy = np.mean(pcm*pcm, axis=1)
        sums = np.convolve(energy, np.ones(48), mode='valid')
        cases = [('start', 0, min(natural//4, 40)),
                 ('middle-0.96s', max(10, (natural-48)//2), 12),
                 ('active-0.96s', 10+int(np.argmax(sums[10:])), 12)]
        for category, offset, count in cases:
            plain = bits[offset:offset+count*4].reshape(count, 4, 49)
            seed = int.from_bytes(hashlib.sha256(f'98027:{row["id"]}:{category}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            for trial in range(args.rounds):
                key = 0 if trial == 0 else int(rng.integers(1, 32768))
                encrypted = plain ^ sequence.thumbdv(key)[None]
                found, at, cpu_seconds = None, None, 0.0
                session = solver.new_session(args.threshold, args.margin) if hasattr(solver, 'new_session') else None
                decision = None
                for end in range(1, count+1):
                    start = time.perf_counter()
                    if session is not None:
                        decision = session.push(encrypted[end-1])
                        found = decision['key']
                    else:
                        found = solver.recover(encrypted[:end])
                    cpu_seconds += time.perf_counter()-start
                    if found is not None:
                        at = end*.08
                        break  # First output matters, even if later evidence would reject it.
                results.append(dict(id=row['id'], corpus=row['corpus'], category=category,
                                    trial=trial, expected=key, recovered=found,
                                    first_seconds=at, max_seconds=count*.08,
                                    score=decision['score'] if decision else None,
                                    margin=decision['margin'] if decision else None,
                                    cpu_seconds=cpu_seconds,
                                    outcome='correct' if found == key else 'unresolved' if found is None else 'wrong'))
    groups = defaultdict(Counter)
    for row in results:
        group = groups[row['category']]
        group['trials'] += 1
        group[row['outcome']] += 1
        if row['outcome'] == 'correct' and row['first_seconds'] <= 1:
            group['correct_by_1s'] += 1
        group['cpu_seconds'] += row['cpu_seconds']
    report = dict(split=args.split, rounds=args.rounds, utterances=len(rows),
                  threshold=args.threshold, margin=args.margin,
                  solver_sha256=hashlib.sha256(solver_file.read_bytes()).hexdigest(),
                  groups=dict(groups), trials=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(dict(groups), indent=2))


if __name__ == '__main__':
    main()
