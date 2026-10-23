"""Generate a planted random PN15 test; score a completed search separately."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from std_t98_cracker.burst import pack49, write_bursts
from std_t98_cracker.enhanced import write_json
from std_t98_cracker.enhanced_sequence import arbitrary_pn
from std_t98_cracker.sequence import THUMBDV_MAP


def maximal(mask):
    state = 1
    for n in range(1, 32768):
        state = (state >> 1) | (((state & mask).bit_count() & 1) << 14)
        if state == 1:
            return n == 32767
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    make = sub.add_parser('make')
    make.add_argument('--corpus', type=Path, required=True)
    make.add_argument('--seed', type=int, default=981027)
    make.add_argument('--seconds', type=float, default=1.2)
    make.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('check')
    check.add_argument('--truth', type=Path, required=True)
    check.add_argument('--result', type=Path, required=True)
    check.add_argument('--json', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'make':
        if not math.isfinite(args.seconds) or args.seconds <= 0:
            raise ValueError('seconds must be positive and finite')
        frame_count = math.ceil(args.seconds/0.08)*4
        rng = np.random.default_rng(args.seed)
        # Random primitive polynomial: avoid short-cycle ambiguities in the planted test.
        while True:
            mask = int(rng.integers(0, 32768)) | 1
            if maximal(mask):
                break
        offset = int(rng.integers(0, 32767))
        key = int(rng.integers(1, 32768))
        state = (key+offset-1) % 32767+1
        plan = json.loads((args.corpus/'plan.json').read_text(encoding='utf-8'))
        samples = [row for row in plan if row['split'] == 'eval' and row['duration'] >= max(3, 0.4+frame_count/50)]
        if not samples:
            raise ValueError('no evaluation utterance is long enough')
        row = samples[int(rng.integers(len(samples)))]
        with np.load(args.corpus/(row['id']+'.npz')) as data:
            plain = np.unpackbits(data['bits'], axis=-1)[20:20+frame_count, :49].reshape(-1, 4, 49)
        pn = arbitrary_pn(mask, state).reshape(4, 49)[:, THUMBDV_MAP]
        args.output.mkdir(parents=True, exist_ok=True)
        write_bursts(args.output/'encrypted.burst', pack49(plain ^ pn))
        write_json(args.output/'truth.json', dict(mask=mask, offset=offset, known_key=key, initial_state=state,
                   seed=args.seed, sample=row['id'], corpus=row['corpus'], first_frame=20, frame_count=frame_count,
                   selection='random maximal-period mask, random offset and key'))
        print(f'Input: {args.output/"encrypted.burst"}; known key: {key}; truth is stored separately.')
    else:
        truth = json.loads(args.truth.read_text(encoding='utf-8'))
        report = json.loads(args.result.read_text(encoding='utf-8'))
        best = report['top_candidates'][0] if report['top_candidates'] else None
        rank = next((i+1 for i, row in enumerate(report['top_candidates'])
                     if (row['mask'], row['offset']) == (truth['mask'], truth['offset'])), None)
        result = dict(correct=rank == 1, truth_rank=rank, truth=truth, best=best,
                      candidates=report['candidates'], retained=report['retained'],
                      elapsed_seconds=report['elapsed_seconds'], config=report['config'])
        write_json(args.json, result)
        print(json.dumps(result, indent=2))
        if rank != 1:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
