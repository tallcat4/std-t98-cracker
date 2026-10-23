"""Measure the GPU PN/scoring path, including an all-survive prefilter worst case."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from std_t98_cracker.burst import read_bursts, unpack49
from std_t98_cracker.enhanced import select_blocks, sha256, write_json
from std_t98_cracker.enhanced_benchmark import GpuMonitor
from std_t98_cracker.enhanced_sequence import coefficient_table
from std_t98_cracker.enhanced_sweep import MaskSweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path, help='49-bit input used only as a fixed inference workload')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.seconds < 5:
        raise ValueError('use a new output path and at least 5 seconds per mode')
    bits, payload = read_bursts(args.input)
    if bits != 49:
        raise ValueError('benchmark input must be 49-bit .burst')
    blocks, _ = select_blocks(unpack49(payload), np.ones((len(payload), 4), dtype=bool))
    coefficients = coefficient_table(np.arange(32768))
    measurements = []
    for name, nblocks, threshold, rank_bound in (
        ('hybrid-only-1-block', 1, None, False),
        ('hybrid-only-3-blocks', 3, None, False),
        ('prefilter-all-survive-3-blocks', 3, -1e6, False),
        ('prefilter-minus10-rank-bound-3-blocks', 3, -10, True),
    ):
        engine = MaskSweep(blocks[:nblocks], batch_size=args.batch_size, prefilter_margin=threshold)
        for i in range(3):
            engine.score_mask(coefficients[10000+i], 20)
        begin, candidates, retained, done = time.perf_counter(), 0, 0, 0
        best = []
        with GpuMonitor() as monitor:
            while time.perf_counter()-begin < args.seconds:
                # Reset the ranking bound at each shard, matching the search path.
                if done % 128 == 0:
                    best = []
                cutoff = best[-1]['minimum_margin'] if rank_bound and len(best) >= 20 else float('-inf')
                rows, count = engine.score_mask(coefficients[(10000+done) % 32768], 20, cutoff)
                best.extend(rows)
                best.sort(key=lambda r: (r['minimum_margin'], r['mean_margin']), reverse=True)
                del best[20:]
                retained += count
                candidates += 32767
                done += 1
            engine.heavy.synchronize()
            end = time.perf_counter()
        elapsed = end-begin
        row = dict(mode=name, blocks=nblocks, threshold=threshold, rank_bound=rank_bound,
                   candidates=candidates, retained=retained, elapsed_seconds=elapsed,
                   candidates_per_second=candidates/elapsed, projected_2pow30_hours=2**30*elapsed/candidates/3600,
                   telemetry=[s for s in monitor.samples if begin <= s['monotonic'] <= end])
        if threshold == -1e6 and retained != candidates:
            raise RuntimeError('all-survive workload did not retain all candidates')
        measurements.append(row)
        model_description = engine.heavy.describe()
        print(json.dumps({k:v for k,v in row.items() if k != 'telemetry'}), flush=True)
        del engine
    write_json(args.output, dict(schema='enhanced-gpu-sweep-benchmark-v1', input_sha256=sha256(args.input),
                                model=model_description, batch_size=args.batch_size, measurements=measurements,
                                interpretation='Measured GPU PN generation, XOR, model inference and top candidate transfer. '
                                'FP32, no TF32. 2^30 times are extrapolations. '
                                'All-survive mode deliberately retains every candidate, approximating the filtering worst case. '
                                'Minus10/rank-bound time depends on input. Excludes recording/FEC/setup/checkpoint I/O.'))


if __name__ == '__main__':
    main()
