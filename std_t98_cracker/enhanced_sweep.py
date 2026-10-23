"""Explicit 15-bit mask/state search; optional conservative FFNN prefilter."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .enhanced import HeavyScorer, select_blocks, sha256, write_json
from .enhanced_sequence import coefficient_table
from .burst import read_bursts, read_llr, unpack49


class MaskSweep:
    def __init__(self, blocks, device='cuda', batch_size=1024, prefilter_margin=None):
        self.heavy = HeavyScorer(device, batch_size)
        self.t = t = self.heavy.torch
        self.blocks = t.as_tensor(blocks, dtype=t.uint8, device=self.heavy.device)
        self.states = t.arange(1, 32768, device=self.heavy.device, dtype=t.int32)
        self.shifts = t.arange(15, device=self.heavy.device, dtype=t.int32)
        self.key_bits = ((self.states[:, None] >> self.shifts) & 1).float()
        self.prefilter_margin = prefilter_margin
        self.light = None
        if prefilter_margin is not None:
            from .ai import _load_checkpoint
            self.light_path = Path(__file__).parent/'models'/'ambe2_ffnn.safetensors'
            self.light, metadata = _load_checkpoint(self.light_path, self.heavy.device)
            if metadata.get('model_type') == 'hybrid':
                raise ValueError('prefilter must be the FFNN model')

    def pn(self, coefficients):
        t = self.t
        coef = t.as_tensor(coefficients.astype(np.int32), device=self.heavy.device)
        basis = ((coef[:, None] >> self.shifts) & 1).float().T
        # All sums are integers <= 15, exactly representable in FP32.
        return ((self.key_bits @ basis).to(t.uint8) & 1).reshape(-1, 4, 49)

    def score_mask(self, coefficients, top=10, rank_cutoff=float('-inf')):
        t = self.t
        with t.inference_mode():
            pn = self.pn(coefficients)
            if self.light is not None:
                keep = t.zeros(len(pn), dtype=t.bool, device=self.heavy.device)
                for block in self.blocks:
                    inputs = (block[None] ^ pn[:, None]).reshape(-1, 980).float()
                    logits = self.light(inputs)
                    # Reject only if EVERY independent block is below the threshold.
                    keep |= logits[:, 0] - logits[:, 1] >= self.prefilter_margin
                indices = keep.nonzero().flatten()
                selected = pn[indices]
            else:
                indices = t.arange(len(pn), device=self.heavy.device)
                selected = pn
            count = len(selected)
            if count == 0:
                return [], 0
            scores = t.empty((count, len(self.blocks)), device=self.heavy.device)
            active = t.arange(count, device=self.heavy.device)
            for j, block in enumerate(self.blocks):
                for start in range(0, len(active), self.heavy.batch_size):
                    positions = active[start:start+self.heavy.batch_size]
                    part = selected[positions]
                    inputs = (block[None] ^ part[:, None]).reshape(-1, 20, 49).float()
                    logits = self.heavy.model(inputs)
                    scores[positions, j] = logits[:, 0]-logits[:, 1]
                if rank_cutoff != float('-inf'):
                    # The final minimum cannot exceed a prefix minimum. Equality
                    # is retained because mean scores break ties. This is an exact
                    # ranking bound, not an additional learned rejection threshold.
                    active = active[scores[active, :j+1].min(dim=1).values >= rank_cutoff]
                    if len(active) == 0:
                        return [], count
            scores = scores[active]
            indices = indices[active]
            minimum, mean = scores.min(dim=1).values, scores.mean(dim=1)
            # Preserve exact minimum/mean ordering, including ties.
            order = t.argsort(mean, descending=True, stable=True)
            order = order[t.argsort(minimum[order], descending=True, stable=True)][:top]
            values = scores[order].cpu().numpy()
            states = self.states[indices[order]].cpu().tolist()
            rows = [dict(initial_state=s, margins=v.tolist(), minimum_margin=float(v.min()),
                         mean_margin=float(v.mean()), all_blocks_plain=bool((v > 0).all())) for s, v in zip(states, values)]
            return rows, count


def run_sweep(blocks, output, known_key, mask_start=0, mask_stop=32768, device='cuda', batch_size=1024,
              prefilter_margin=None, top=20, resume=False, shard_size=128, rank_bound=False):
    """Search all nonzero states for every requested mask; shard files permit resume."""
    if not 1 <= known_key <= 32767 or not 0 <= mask_start < mask_stop <= 32768:
        raise ValueError('key must be 1..32767; mask range must be within [0,32768)')
    if top < 1 or shard_size < 1 or not np.isfinite(prefilter_margin if prefilter_margin is not None else 0):
        raise ValueError('invalid search limits')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    engine = MaskSweep(blocks, device, batch_size, prefilter_margin)
    config = dict(schema='enhanced-sweep-v1', block_sha256=hashlib.sha256(blocks.tobytes()).hexdigest(),
                  known_key=known_key, mask_start=mask_start, mask_stop=mask_stop, blocks=len(blocks),
                  prefilter_margin=prefilter_margin, model=engine.heavy.describe(), top=top, shard_size=shard_size,
                  prefilter_sha256=sha256(engine.light_path) if engine.light else None)
    config_path = output/'config.json'
    if config_path.exists():
        if not resume or json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise ValueError('existing output requires --resume with identical input and parameters')
    else:
        if any(output.glob('shard-*.json')):
            raise ValueError('shards without config; use a new output directory')
        write_json(config_path, config)
    coefficients = coefficient_table(np.arange(mask_start, mask_stop))
    global_best, total_candidates, survivors, elapsed = [], 0, 0, 0.0
    for start in range(mask_start, mask_stop, shard_size):
        stop = min(start+shard_size, mask_stop)
        path = output/f'shard-{start:05d}-{stop:05d}.json'
        if path.exists():
            shard = json.loads(path.read_text(encoding='utf-8'))
        else:
            began = time.perf_counter()
            best, retained = [], 0
            for mask in range(start, stop):
                cutoff = best[-1]['minimum_margin'] if rank_bound and len(best) >= top else float('-inf')
                rows, count = engine.score_mask(coefficients[mask-mask_start], top, cutoff)
                retained += count
                for row in rows:
                    row.update(mask=mask, polynomial=f'0x{mask | 0x8000:04X}',
                               offset=(row['initial_state']-known_key) % 32767)
                best.extend(rows)
                best.sort(key=lambda r: (r['minimum_margin'], r['mean_margin']), reverse=True)
                del best[top:]
            shard = dict(start=start, stop=stop, candidates=(stop-start)*32767, retained=retained,
                         elapsed_seconds=time.perf_counter()-began, top_candidates=best, rank_bound=rank_bound)
            # Publish only a complete shard so interruptions can be resumed safely.
            temp = path.with_suffix('.tmp')
            with temp.open('w', encoding='utf-8') as f:
                json.dump(shard, f, indent=2, allow_nan=False)
            temp.replace(path)
        global_best.extend(shard['top_candidates'])
        global_best.sort(key=lambda r: (r['minimum_margin'], r['mean_margin']), reverse=True)
        del global_best[top:]
        total_candidates += shard['candidates']
        survivors += shard['retained']
        elapsed += shard['elapsed_seconds']
        if global_best:
            best = global_best[0]
            print(f'masks {stop}/{mask_stop}, retained {survivors}/{total_candidates}, '
                  f'best={best["polynomial"]}/{best["offset"]}, min={best["minimum_margin"]:.3f}, '
                  f'elapsed={elapsed:.1f}s', flush=True)
    report = dict(schema='enhanced-sweep-result-v1', config=config, candidates=total_candidates, retained=survivors,
                  elapsed_seconds=elapsed, candidates_per_second=total_candidates/elapsed,
                  projected_2pow30_hours=2**30*elapsed/total_candidates/3600,
                  top_candidates=global_best, confirmed=False,
                  scope='All 32767 nonzero states per mask. Offset is modulo 32767 for the known key. '
                  'State zero is a plaintext control, not a nonzero-key offset. '
                  'Prefilter can reject a true candidate; rates depend on input. '
                  'Some shards may use an exact minimum-score rank bound; see shard files. '
                  'The timing projection is input-dependent, not an upper bound on runtime.')
    result_path = output/'result.json'
    if not result_path.exists():
        write_json(result_path, report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--llr', type=Path)
    p.add_argument('--known-key', type=int, required=True)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--block-starts', type=lambda v: [int(x) for x in v.split(',')])
    p.add_argument('--mask-start', type=int, default=0)
    p.add_argument('--mask-stop', type=int, default=32768, help='exclusive upper bound')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--prefilter-margin', type=float, help='opt-in FFNN prefilter; e.g. -10, retain if ANY block passes')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--rank-bound', action='store_true', help='skip later blocks only when a prefix minimum cannot enter the shard top 20')
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
        blocks, _ = select_blocks(frames, valid, args.blocks, args.block_starts)
        run_sweep(blocks, args.output, args.known_key, args.mask_start, args.mask_stop, args.device,
                  args.batch_size, args.prefilter_margin, resume=args.resume, rank_bound=args.rank_bound)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
