"""Optional CUDA integration checks using an existing local ThumbDV corpus."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from std_t98_cracker.burst import pack49, write_bursts
from std_t98_cracker.enhanced import HeavyScorer, compare, write_json
from std_t98_cracker.enhanced_sequence import CandidateSequence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all-keys', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'validation.json').exists():
        raise FileExistsError(args.output/'validation.json')
    plan = json.loads((args.corpus/'plan.json').read_text(encoding='utf-8'))
    chosen = [next(row for row in plan if row['split'] == 'eval' and row['corpus'] == corpus and row['duration'] >= 3)
              for corpus in ('jsut', 'libri')]
    scorer = HeavyScorer()
    results = []
    started = time.perf_counter()
    for row in chosen:
        with np.load(args.corpus/(row['id']+'.npz')) as data:
            # Use 1.2 s of natural speech starting 0.4 s into the utterance.
            plain = np.unpackbits(data['bits'], axis=-1)[20:80, :49].reshape(3, 5, 4, 49)
        for family in range(17):
            for key in (1, 12345, 32767):
                pn = CandidateSequence(family).batch(np.array([key]))[0]
                encrypted = plain ^ pn
                report = compare(encrypted, scorer, range(17), np.array([key]))
                best = report['top_candidates'][0]
                results.append(dict(corpus=row['corpus'], sample=row['id'], expected_family=family, key=key,
                                    best=best, correct=best['family'] == family and best['key'] == key))
        print(f'{row["corpus"]}: {sum(r["correct"] for r in results)}/{len(results)} ranked correctly', flush=True)
    if args.all_keys:
        family, key = 7, 12345
        encrypted = plain ^ CandidateSequence(family).batch(np.array([key]))[0]
        write_bursts(args.output/'synthetic.burst', pack49(encrypted.reshape(-1, 4, 49)))
        report = compare(encrypted, scorer, range(17), np.arange(1, 32768), progress=lambda line: print(line, flush=True))
        write_json(args.output/'all-keys.json', report)
    write_json(args.output/'validation.json', dict(model=scorer.describe(), results=results,
                correct=sum(r['correct'] for r in results), total=len(results), elapsed_seconds=time.perf_counter()-started,
                interpretation='Synthetic XOR over real chip frames; does not establish compatibility with a radio.'))


if __name__ == '__main__':
    main()
