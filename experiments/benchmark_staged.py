"""Same-input comparison of reference, staged eager, and staged CUDA replay."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from std_t98_cracker.burst import read_bursts, unpack49
from std_t98_cracker.enhanced import select_blocks, sha256, write_json
from std_t98_cracker.enhanced_benchmark import GpuMonitor
from std_t98_cracker.enhanced_sequence import coefficient_table
from std_t98_cracker.enhanced_staged import StagedSweep, spaced_blocks
from std_t98_cracker.enhanced_sweep import MaskSweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worst-only', action='store_true', help='force both screening blocks and all three final blocks for every candidate')
    args = p.parse_args()
    if args.output.exists() or not np.isfinite(args.seconds) or args.seconds < 5:
        raise ValueError('use a new output path and at least 5 seconds per mode')
    bits, payload = read_bursts(args.input)
    if bits != 49:
        raise ValueError('benchmark input must be 49-bit .burst')
    frames = unpack49(payload)
    valid = np.ones((len(frames), 4), dtype=bool)
    blocks, plan = spaced_blocks(frames, valid)
    legacy, legacy_starts = select_blocks(frames, valid)
    coefficients = coefficient_table(np.arange(32768))
    results = []
    modes = [('staged-forced-all', True, 0)] if args.worst_only else [
        ('reference', False, -10), ('staged-eager', False, 0),
        ('staged-graph', True, 0), ('staged-graph-conservative', True, -10)]
    for name, graphs, threshold in modes:
        engine = MaskSweep(legacy, prefilter_margin=threshold) if name == 'reference' else StagedSweep(blocks, light_margin=threshold, graphs=graphs)
        for i in range(3):
            engine.score_mask(coefficients[10000+i], 20)
        engine.heavy.synchronize()
        start, done, best, totals = time.perf_counter(), 0, [], {}
        with GpuMonitor() as monitor:
            while time.perf_counter()-start < args.seconds:
                if name == 'reference':
                    if done % 128 == 0:
                        best = []
                    cutoff = best[-1]['minimum_margin'] if len(best) >= 20 else float('-inf')
                    rows, count = engine.score_mask(coefficients[(10000+done) % 32768], 20, cutoff)
                    best.extend(rows)
                    best.sort(key=lambda r: (r['minimum_margin'], r['mean_margin']), reverse=True)
                    del best[20:]
                    stats = dict(candidates=32767, light_pass=count)
                elif name == 'staged-forced-all':
                    t = engine.t
                    with t.inference_mode():
                        pn = engine.pn(coefficients[(10000+done) % 32768])
                        for slot in (0, 1):
                            engine.margins(pn, slot, light=True)
                        for slot in (2, 3):
                            engine.margins(pn, slot)
                        scores = t.stack([engine.margins(pn, slot) for slot in (4, 5, 6)], dim=1)
                        medians, means = scores.median(dim=1).values, scores.mean(dim=1)
                        order = t.argsort(means, descending=True, stable=True)
                        order = order[t.argsort(medians[order], descending=True, stable=True)][:20]
                        scores[order].cpu()
                    stats = dict(candidates=32767, light_pass=32767, hybrid_pass=32767, final_pass=32767,
                                 light_evaluations=2*32767, hybrid_evaluations=2*32767, final_evaluations=3*32767)
                else:
                    _, stats = engine.score_mask(coefficients[(10000+done) % 32768], 20)
                for key, value in stats.items():
                    totals[key] = totals.get(key, 0)+value
                done += 1
            engine.heavy.synchronize()
            end = time.perf_counter()
        elapsed = end-start
        row = dict(mode=name, seconds=elapsed, counts=totals, candidates_per_second=totals['candidates']/elapsed,
                   projected_2pow30_minutes=2**30*elapsed/totals['candidates']/60,
                   forced_workload=name == 'staged-forced-all',
                   telemetry=[s for s in monitor.samples if start <= s['monotonic'] <= end])
        results.append(row)
        model = engine.heavy.describe()
        print(json.dumps({k:v for k,v in row.items() if k != 'telemetry'}), flush=True)
        del engine
    write_json(args.output, dict(input_sha256=sha256(args.input), plan=plan, legacy_starts=legacy_starts,
                                model=model, results=results,
                                limitation='Short steady-state measurements on the same input. Reference uses first 1.2 seconds; staged uses seven spaced blocks. '
                                'All FP32, no TF32. Setup, recording/FEC and checkpoint I/O excluded. '
                                'Forced-all disregards classifier outcomes to measure maximum stage work; it is not a recovery test. Other projections are input-dependent.'))


if __name__ == '__main__':
    main()
