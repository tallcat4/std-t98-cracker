"""Sustained hybrid throughput measurement; no lightweight candidate filtering."""
import json
import subprocess
import threading
import time

import numpy as np


class GpuMonitor:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                for line in result.stdout.splitlines():
                    fields = line.split(',')
                    if len(fields) == 6:
                        self.samples.append(dict(monotonic=time.perf_counter(), gpu=int(fields[0]),
                                                 utilization=float(fields[1]), memory_mib=float(fields[2]),
                                                 power_watts=float(fields[3]), temperature_c=float(fields[4]),
                                                 clock_mhz=float(fields[5])))
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)


def benchmark(args):
    from .enhanced import HeavyScorer, write_json
    from .enhanced_sequence import CandidateSequence
    if args.json.exists():
        raise FileExistsError(args.json)
    if args.seconds < 5 or any(b < 1 for b in args.batches):
        raise ValueError('seconds must be >= 5; batch sizes must be positive')
    scorer = HeavyScorer(args.device)
    t = scorer.torch
    rng = np.random.default_rng(98)
    tuning = []
    with GpuMonitor() as monitor, t.inference_mode():
        for batch in args.batches:
            inputs = None
            try:
                inputs = t.as_tensor(rng.integers(0, 2, (batch, 20, 49), dtype=np.uint8),
                                     device=scorer.device, dtype=t.float32)
                for _ in range(3):
                    scorer.model(inputs)
                scorer.synchronize()
                begin, iterations = time.perf_counter(), 0
                while time.perf_counter()-begin < 3:
                    scorer.model(inputs)
                    scorer.synchronize()
                    iterations += 1
                elapsed = time.perf_counter()-begin
                row = dict(batch_size=batch, candidate_blocks_per_second=batch*iterations/elapsed,
                           elapsed_seconds=elapsed, iterations=iterations)
                tuning.append(row)
                print(json.dumps(row), flush=True)
            except t.cuda.OutOfMemoryError:
                tuning.append(dict(batch_size=batch, error='CUDA out of memory'))
                print(f'batch {batch}: CUDA out of memory', flush=True)
            finally:
                del inputs
                if scorer.device.type == 'cuda':
                    t.cuda.empty_cache()
        successes = [r for r in tuning if 'error' not in r]
        if not successes:
            raise RuntimeError('no batch size completed')
        scorer.batch_size = max(successes, key=lambda r: r['candidate_blocks_per_second'])['batch_size']
        # The complete comparison path includes PN generation, host/device transfer,
        # XOR, FP32 hybrid inference, and score transfer. No early exits or pruning.
        keys = np.arange(1, 32768)
        measurements = []
        for block_count in (1, 3):
            blocks = rng.integers(0, 2, (block_count, 5, 4, 49), dtype=np.uint8)
            scorer.score(blocks, CandidateSequence(0).batch(keys[:scorer.batch_size]))
            scorer.synchronize()
            begin, iterations, pn_seconds = time.perf_counter(), 0, 0.0
            while time.perf_counter()-begin < args.seconds:
                generation_start = time.perf_counter()
                pn = CandidateSequence(iterations % 17).batch(keys)
                pn_seconds += time.perf_counter()-generation_start
                scorer.score(blocks, pn)
                iterations += 1
            scorer.synchronize()
            end = time.perf_counter()
            elapsed = end-begin
            candidates = len(keys)*iterations
            rate = candidates/elapsed
            telemetry = [s for s in monitor.samples if begin <= s['monotonic'] <= end]
            row = dict(blocks_per_candidate=block_count, batch_size=scorer.batch_size,
                       candidates=candidates, elapsed_seconds=elapsed, candidates_per_second=rate,
                       candidate_blocks_per_second=rate*block_count, pn_generation_seconds=pn_seconds,
                       worst_case_candidates=2**30, projected_2pow30_hours=2**30/rate/3600,
                       projected_17_family_seconds=17*32767/rate,
                       telemetry=telemetry)
            measurements.append(row)
            print(f'{block_count} blocks: {rate:.1f} candidates/s; 2^30 = {row["projected_2pow30_hours"]:.2f} hours', flush=True)
    report = dict(schema='enhanced-benchmark-v1', model=scorer.describe(), tuning=tuning,
                  measurements=measurements,
                  interpretation='Measured full scoring path on synthetic bits, not accuracy. Extrapolation, not a completed 2^30 search. '
                  'Fixed mask per batch, 32767 states per mask. Excludes recording/FEC/file IO and final sorting. '
                  'Default burst reset and zero skip; additional hypotheses scale total work. No pruning or early stop.')
    write_json(args.json, report)
    print(args.json)
