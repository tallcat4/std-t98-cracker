"""Spaced, disjoint light-OR / hybrid-OR / hybrid-two-of-three search."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .burst import read_bursts, read_llr, unpack49
from .enhanced import integer_list, sha256, write_json
from .enhanced_sequence import coefficient_table
from .enhanced_sweep import MaskSweep


# Indices into seven chronologically sorted, nonoverlapping windows. Each stage
# sees separated time ranges; the last stage never reuses screening windows.
STAGES = ((1, 4), (2, 5), (0, 3, 6))
BURST_SECONDS = 0.08


def spaced_blocks(frames, valid, gap_seconds=2.0, starts=None):
    if not math.isfinite(gap_seconds) or gap_seconds < 0.4:
        raise ValueError('gap must be at least 0.4 seconds')
    gap = math.ceil(gap_seconds/BURST_SECONDS)
    if starts is None:
        good = valid.all(axis=1)
        available = np.flatnonzero(np.convolve(good.astype(int), np.ones(5, int), mode='valid') == 5) if len(good) >= 5 else np.array([], int)
        if len(available) < 7:
            raise ValueError('not enough FEC-valid windows for seven disjoint blocks')
        targets = np.linspace(available[0], available[-1], 7)
        starts = []
        for i, target in enumerate(targets):
            lower = starts[-1]+5 if starts else 0
            if i >= 3:
                lower = max(lower, starts[i-3]+gap)
            possible = available[(available >= lower) & (available <= available[-1]-(6-i)*5)]
            if not len(possible):
                raise ValueError('not enough separated valid windows; use a longer capture or explicit --block-starts')
            starts.append(int(possible[np.argmin(abs(possible-target))]))
    if len(starts) != 7 or any(b-a < 5 for a, b in zip(starts, starts[1:])):
        raise ValueError('provide seven chronological, nonoverlapping 5-burst starts')
    for start in starts:
        if start < 0 or start+5 > len(frames) or not valid[start:start+5].all():
            raise ValueError('all selected blocks must be in range and FEC-valid')
    for stage in STAGES:
        if any(starts[b]-starts[a] < gap for a, b in zip(stage, stage[1:])):
            raise ValueError('selected blocks are too close within a stage; use a longer capture')
    order = [i for stage in STAGES for i in stage]
    blocks = np.stack([frames[starts[i]:starts[i]+5] for i in order])
    plan = dict(chronological_starts=list(starts), light_starts=[starts[i] for i in STAGES[0]],
                hybrid_starts=[starts[i] for i in STAGES[1]], final_starts=[starts[i] for i in STAGES[2]],
                gap_seconds=gap_seconds, time_basis='stored burst index * 0.08 s; missing RF time is not reconstructed')
    return blocks, plan


class StagedSweep(MaskSweep):
    def __init__(self, blocks, device='cuda', batch_size=1024, light_margin=0.0, hybrid_margin=0.0, graphs=True):
        if np.shape(blocks) != (7, 5, 4, 49):
            raise ValueError('staged search needs exactly seven 5-burst blocks')
        if not np.isfinite([light_margin, hybrid_margin]).all():
            raise ValueError('thresholds must be finite')
        super().__init__(blocks, device, batch_size, light_margin)
        self.hybrid_margin = hybrid_margin
        self.runner = self.heavy.model
        self.graphs = graphs and self.heavy.device.type == 'cuda'
        if self.graphs:
            from .cuda_graph import GraphRunner
            self.runner = GraphRunner(self.heavy.model, self.heavy.device, batch_size)
            self.runner.warmup()

    def margins(self, pn, slot, light=False):
        t = self.t
        batch = 32768 if light else self.heavy.batch_size
        output = t.empty(len(pn), device=self.heavy.device)
        for start in range(0, len(pn), batch):
            part = pn[start:start+batch]
            inputs = (self.blocks[slot][None] ^ part[:, None]).reshape(-1, 20, 49).float()
            logits = self.light(inputs.flatten(1)) if light else self.runner(inputs)
            output[start:start+len(part)] = logits[:, 0]-logits[:, 1]
        return output

    def score_mask(self, coefficients, top=20):
        t = self.t
        with t.inference_mode():
            pn = self.pn(coefficients)
            indices = t.arange(len(pn), device=self.heavy.device)
            counts = dict(candidates=len(pn), light_pass=0, hybrid_pass=0, final_pass=0,
                          light_evaluations=0, hybrid_evaluations=0, final_evaluations=0)
            # Within each OR stage, a positive first result already qualifies;
            # evaluate the second window only for candidates still undecided.
            for slots, threshold, light, field in (((0, 1), self.prefilter_margin, True, 'light'),
                                                   ((2, 3), self.hybrid_margin, False, 'hybrid')):
                keep = t.zeros(len(indices), dtype=t.bool, device=self.heavy.device)
                for slot in slots:
                    undecided = (~keep).nonzero().flatten()
                    if not len(undecided):
                        break
                    scores = self.margins(pn[indices[undecided]], slot, light)
                    counts[field+'_evaluations'] += len(scores)
                    keep[undecided] = scores >= threshold
                indices = indices[keep]
                counts[field+'_pass'] = len(indices)
                if not len(indices):
                    return [], counts
            scores = t.stack([self.margins(pn[indices], slot) for slot in (4, 5, 6)], dim=1)
            counts['final_evaluations'] = scores.numel()
            votes = (scores > 0).sum(dim=1)
            eligible = votes >= 2
            scores, indices, votes = scores[eligible], indices[eligible], votes[eligible]
            counts['final_pass'] = len(indices)
            if not len(indices):
                return [], counts
            median, mean = scores.median(dim=1).values, scores.mean(dim=1)
            order = t.argsort(mean, descending=True, stable=True)
            order = order[t.argsort(median[order], descending=True, stable=True)][:top]
            values = scores[order].cpu().numpy()
            states = self.states[indices[order]].cpu().tolist()
            rows = [dict(initial_state=state, final_margins=value.tolist(), final_plain_votes=int((value > 0).sum()),
                         median_margin=float(np.median(value)), mean_margin=float(value.mean()))
                    for state, value in zip(states, values)]
            return rows, counts


def run_staged(blocks, output, known_key, plan, mask_start=0, mask_stop=32768, light_margin=0.0,
               hybrid_margin=0.0, device='cuda', batch_size=1024, graphs=True, resume=False, top=20):
    if not 1 <= known_key <= 32767 or not 0 <= mask_start < mask_stop <= 32768:
        raise ValueError('invalid key or mask range')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    engine = StagedSweep(blocks, device, batch_size, light_margin, hybrid_margin, graphs)
    config = dict(schema='enhanced-staged-v1', block_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
                  known_key=known_key, plan=plan, mask_start=mask_start, mask_stop=mask_stop, light_margin=light_margin,
                  hybrid_margin=hybrid_margin, graphs=engine.graphs, model=engine.heavy.describe(),
                  light_model_sha256=sha256(engine.light_path), top=top, final_rule='two of three > 0, rank by median then mean')
    path = output/'config.json'
    if path.exists():
        if not resume or json.loads(path.read_text(encoding='utf-8')) != config:
            raise ValueError('resume requires identical input and parameters')
    else:
        if any(output.glob('shard-*.json')):
            raise ValueError('shards without config; use a new output directory')
        write_json(path, config)
    coefficients = coefficient_table(np.arange(mask_start, mask_stop))
    best, totals, elapsed = [], {}, 0.0
    rank = lambda row: (row['median_margin'], row['mean_margin'])
    for start in range(mask_start, mask_stop, 128):
        stop = min(start+128, mask_stop)
        path = output/f'shard-{start:05d}-{stop:05d}.json'
        if path.exists():
            shard = json.loads(path.read_text(encoding='utf-8'))
        else:
            began, rows, counts = time.perf_counter(), [], {}
            for mask in range(start, stop):
                candidates, found = engine.score_mask(coefficients[mask-mask_start], top)
                for key, value in found.items():
                    counts[key] = counts.get(key, 0)+value
                for candidate in candidates:
                    candidate.update(mask=mask, polynomial=f'0x{mask | 0x8000:04X}',
                                     offset=(candidate['initial_state']-known_key) % 32767)
                rows.extend(candidates)
                rows.sort(key=rank, reverse=True)
                del rows[top:]
            shard = dict(start=start, stop=stop, counts=counts, elapsed_seconds=time.perf_counter()-began, top_candidates=rows)
            temp = path.with_suffix('.tmp')
            with temp.open('w', encoding='utf-8') as stream:
                json.dump(shard, stream, indent=2, allow_nan=False)
            temp.replace(path)
        for key, value in shard['counts'].items():
            totals[key] = totals.get(key, 0)+value
        elapsed += shard['elapsed_seconds']
        best.extend(shard['top_candidates'])
        best.sort(key=rank, reverse=True)
        del best[top:]
        leader = f'{best[0]["polynomial"]}/{best[0]["offset"]}, score={best[0]["median_margin"]:.3f}' if best else 'none'
        print(f'masks {stop}/{mask_stop}, stage passes {totals["light_pass"]}/{totals["hybrid_pass"]}/{totals["final_pass"]}, best {leader}, {elapsed:.1f}s', flush=True)
    result = dict(schema='enhanced-staged-result-v1', config=config, counts=totals, elapsed_seconds=elapsed,
                  candidates=totals['candidates'], retained=totals['light_pass'], top_candidates=best, confirmed=False,
                  projected_2pow30_hours=elapsed/totals['candidates']*2**30/3600,
                  scope='Input-dependent staged screening. Independent time windows, not a calibrated probability or a guarantee. All models FP32.')
    if not (output/'result.json').exists():
        write_json(output/'result.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--llr', type=Path)
    p.add_argument('--known-key', type=int, required=True)
    p.add_argument('--gap-seconds', type=float, default=2.0)
    p.add_argument('--block-starts', type=integer_list)
    p.add_argument('--light-margin', type=float, default=0.0, help='0 is the default plain decision; use -10 for more conservative screening')
    p.add_argument('--hybrid-margin', type=float, default=0.0)
    p.add_argument('--mask-start', type=int, default=0)
    p.add_argument('--mask-stop', type=int, default=32768)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-cuda-graphs', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        bits, payload = read_bursts(args.input)
        if bits == 72:
            from .fec import FecDecoder
            frames, valid, _ = FecDecoder().fec(payload, read_llr(args.llr, len(payload)) if args.llr else None)
        else:
            if args.llr:
                raise ValueError('LLR requires 72-bit input')
            frames, valid = unpack49(payload), np.ones((len(payload), 4), dtype=bool)
        blocks, plan = spaced_blocks(frames, valid, args.gap_seconds, args.block_starts)
        print(json.dumps(plan), flush=True)
        run_staged(blocks, args.output, args.known_key, plan, args.mask_start, args.mask_stop, args.light_margin,
                   args.hybrid_margin, args.device, args.batch_size, not args.no_cuda_graphs, args.resume)
    except (ValueError, OSError, RuntimeError) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
