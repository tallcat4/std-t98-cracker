"""Reproducible chip-frame/known-random-key benchmark; no AI and no RF input."""
import argparse
from collections import defaultdict
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


def cases(path):
    z = np.load(path)
    metadata = json.loads(str(z['metadata']))
    bits = np.unpackbits(z['bits'], axis=-1)[:, :49]
    natural = metadata['natural_frames']
    # Exclude reset transients and rounding. Never join unrelated utterances.
    for length in (40, 160):
        if natural < length + 10:
            continue
        start = max(10, (natural-length)//2)
        yield f'natural-{length*.02:.1f}s', bits[start:start+length].reshape(-1, 4, 49)
        pcm = z['pcm'][:natural*160].reshape(-1, 160).astype(np.float32)
        energy = np.mean(pcm*pcm, axis=1)
        sums = np.convolve(energy, np.ones(length), mode='valid')
        start = 10 + int(np.argmax(sums[10:]))
        yield f'active-{length*.02:.1f}s', bits[start:start+length].reshape(-1, 4, 49)
    # A separately-labelled positive control, not part of natural speech accuracy.
    yield 'added-zero-0.8s', bits[-40:].reshape(10, 4, 49)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--corpus', type=Path, required=True)
    ap.add_argument('--split', choices=('train', 'eval'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=98024)
    ap.add_argument('--solver-file', type=Path, help='Optional frozen baseline module (development only)')
    ap.add_argument('--require-complete', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError('Output exists')
    solver_class = SilenceSolver
    solver_file = Path(__file__).resolve().parents[1]/'std_t98_cracker/linear.py'
    if args.solver_file:
        solver_file = args.solver_file
        spec = importlib.util.spec_from_file_location('std_t98_cracker._benchmark_baseline', solver_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solver_class = module.SilenceSolver
    solver, sequence = solver_class(), Sequence()
    stats = defaultdict(lambda: dict(trials=0, correct=0, wrong=0, unresolved=0, seconds=0.0))
    details = []
    plan = json.loads((args.corpus/'plan.json').read_text())
    planned = [row for row in plan if row['split'] == args.split]
    completed = [row for row in planned if (args.corpus/(row['id']+'.npz')).exists()]
    if args.require_complete and len(planned) != len(completed):
        raise RuntimeError(f'Capture incomplete: {len(completed)}/{len(planned)}')
    for row in plan:
        path = args.corpus/(row['id']+'.npz')
        if row['split'] != args.split or not path.exists():
            continue
        for category, plain in cases(path):
            seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{row["id"]}:{category}'.encode()).digest()[:8], 'little')
            rng = np.random.default_rng(seed)
            for trial in range(args.rounds):
                key = 0 if trial == 0 else int(rng.integers(1, 32768))
                encrypted = plain ^ sequence.thumbdv(key)[None]
                start = time.perf_counter()
                recovered = solver.recover(encrypted)
                duration = time.perf_counter()-start
                outcome = 'correct' if recovered == key else 'unresolved' if recovered is None else 'wrong'
                group = f'{row["corpus"]}/ecmode-{row["ecmode"]}/{category}'
                stats[group]['trials'] += 1
                stats[group][outcome] += 1
                stats[group]['seconds'] += duration
                details.append(dict(id=row['id'], category=category, trial=trial, expected=key,
                                    recovered=recovered, seconds=duration, outcome=outcome))
    report = dict(split=args.split, seed=args.seed, rounds=args.rounds,
                  planned_utterances=len(planned), completed_utterances=len(completed),
                  solver_sha256=hashlib.sha256(solver_file.read_bytes()).hexdigest(),
                  groups=dict(stats), trials=details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
    for group, s in sorted(stats.items()):
        print(group, s['trials'], 'correct', s['correct'], 'wrong', s['wrong'],
              'unresolved', s['unresolved'], 'mean ms', round(s['seconds']/s['trials']*1000, 3))
    print('Total trials:',len(details))


if __name__ == '__main__':
    main()
