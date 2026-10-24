"""Explicit experimental candidate comparison using only the heavy hybrid model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .burst import read_bursts, read_llr, unpack49
from .enhanced_sequence import CandidateSequence, MASKS, OFFSETS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def setting(value):
    try:
        series, key = map(int, value.split('-'))
        if not 1 <= series <= 15 or not 1 <= key <= 32767:
            raise ValueError()
        return series, key
    except ValueError:
        raise argparse.ArgumentTypeError('setting must be SERIES-KEY, e.g. 5-00001 (series 1..15, key 1..32767)') from None


def integer_list(value):
    try:
        result = [int(x) for x in value.split(',')]
        if not result or len(set(result)) != len(result):
            raise ValueError()
        return result
    except ValueError:
        raise argparse.ArgumentTypeError('expected distinct comma-separated integers') from None


def select_blocks(frames, valid, count=3, starts=None):
    """Keep the original burst boundaries; never join across failed FEC."""
    if count < 2:
        raise ValueError('use at least two independent 5-burst blocks')
    if starts is None:
        starts, cursor = [], 0
        while cursor + 5 <= len(frames) and len(starts) < count:
            if valid[cursor:cursor+5].all():
                starts.append(cursor)
                cursor += 5
            else:
                cursor += 1
    if len(starts) != count:
        raise ValueError(f'need {count} nonoverlapping valid blocks of 5 bursts; found {len(starts)}')
    used = set()
    for start in starts:
        indices = set(range(start, start+5))
        if start < 0 or start+5 > len(frames) or used & indices or not valid[start:start+5].all():
            raise ValueError('block starts must select nonoverlapping, in-range, FEC-valid blocks')
        used.update(indices)
    return np.stack([frames[s:s+5] for s in starts]), list(starts)


class HeavyScorer:
    def __init__(self, device='cuda', batch_size=512, model_path=None, threads=4):
        import torch
        from .ai import _load_checkpoint
        if batch_size < 1:
            raise ValueError('batch size must be positive')
        torch.set_num_threads(threads)
        self.torch = torch
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.batch_size = batch_size
        self.path = Path(model_path) if model_path else Path(__file__).parent / 'models' / 'ambe2_hybrid.safetensors'
        self.model, metadata = _load_checkpoint(self.path, self.device)
        if metadata.get('model_type') != 'hybrid':
            raise ValueError('this experiment requires the heavy hybrid checkpoint')
        # Keep FP32 inference consistent with the existing precision verifier.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)

    def describe(self):
        t = self.torch
        return dict(model='hybrid', model_sha256=sha256(self.path), dtype='float32',
                    device=str(self.device), gpu=t.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
                    torch=t.__version__, cuda=t.version.cuda, batch_size=self.batch_size,
                    tf32=False, lightweight_pruning=False)

    def score(self, blocks, pn):
        """Every candidate is evaluated on EVERY block; output is plain logit margin."""
        t = self.torch
        output = np.empty((len(pn), len(blocks)), dtype=np.float32)
        with t.inference_mode():
            encrypted = t.as_tensor(blocks, dtype=t.uint8, device=self.device)
            masks = t.as_tensor(pn, dtype=t.uint8, device=self.device)
            for start in range(0, len(pn), self.batch_size):
                mask = masks[start:start+self.batch_size]
                for j, block in enumerate(encrypted):
                    plain = (block[None] ^ mask[:, None]).reshape(-1, 20, 49).float()
                    logits = self.model(plain)
                    margin = logits[:, 0] - logits[:, 1]
                    output[start:start+len(mask), j] = margin.cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError('model returned a nonfinite score')
        return output


def rank_candidates(sequence, keys, scores, top=10):
    # Favor support across independent blocks, then the average margin.
    order = np.lexsort((-scores.mean(axis=1), -scores.min(axis=1)))[:top]
    return [dict(family=sequence.family, mask=f'0x{MASKS[sequence.family]:04X}',
                 key=int(keys[i]), initial_state=sequence.initial_state(int(keys[i])),
                 margins=[float(v) for v in scores[i]],
                 minimum_margin=float(scores[i].min()), mean_margin=float(scores[i].mean()),
                 all_blocks_plain=bool((scores[i] > 0).all())) for i in order]


def compare(blocks, scorer, families, keys, initialization='offset', skip=0, reset='burst', top=10, progress=None):
    rows, family_best = [], []
    for family in families:
        sequence = CandidateSequence(family, initialization, skip, reset)
        scores = scorer.score(blocks, sequence.batch(keys))
        ranked = rank_candidates(sequence, keys, scores, top)
        rows.extend(ranked)
        family_best.append(ranked[0])
        if progress:
            progress(f'family {family:02d}: best key {ranked[0]["key"]:05d}, minimum margin {ranked[0]["minimum_margin"]:.3f}')
    baseline = scorer.score(blocks, np.zeros((1, 4, 49), dtype=np.uint8))[0]
    rows.sort(key=lambda row: (row['minimum_margin'], row['mean_margin']), reverse=True)
    return dict(top_candidates=rows[:top], family_best=family_best,
                plaintext_control=dict(margins=baseline.tolist(), all_blocks_plain=bool((baseline > 0).all())),
                evaluated_candidates=len(families)*len(keys), evaluated_blocks=len(blocks),
                status='ranked-candidates', confirmed=False)


def probe(args):
    if args.json.exists():
        raise FileExistsError(args.json)
    bits, payload = read_bursts(args.input)
    if bits == 72:
        from .fec import FecDecoder
        llr = read_llr(args.llr, len(payload)) if args.llr else None
        frames, valid, corrected = FecDecoder().fec(payload, llr)
    else:
        if args.llr:
            raise ValueError('LLR is only applicable to 72-bit input')
        frames, valid = unpack49(payload), np.ones((len(payload), 4), dtype=bool)
        corrected = np.zeros_like(valid, dtype=np.int16)
    blocks, starts = select_blocks(frames, valid, args.blocks, args.block_starts)
    if any(f < 0 or f >= len(MASKS) for f in args.families):
        raise ValueError('families must be 0..16')
    known_key = args.setting[1] if args.setting else args.key
    if known_key is None and not args.all_keys:
        raise ValueError('specify --setting, --key, or --all-keys')
    if known_key is not None and not 1 <= known_key <= 32767:
        raise ValueError('key must be 1..32767; unencrypted input is tested separately')
    keys = np.arange(1, 32768) if args.all_keys else np.array([known_key])
    scorer = HeavyScorer(args.device, args.batch_size)
    started = time.perf_counter()
    report = compare(blocks, scorer, args.families, keys, args.initialization, args.skip, args.reset,
                     args.top, lambda line: print(line, flush=True))
    report.update(schema='enhanced-probe-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  input=dict(name=args.input.name, sha256=sha256(args.input), bits=bits, bursts=len(payload),
                             llr_sha256=sha256(args.llr) if args.llr else None,
                             invalid_frames=int((~valid).sum()), corrected_bits=int(corrected[valid].sum())),
                  radio_series=args.setting[0] if args.setting else None, radio_key=known_key,
                  parameters=dict(initialization=args.initialization, skip=args.skip, reset=args.reset,
                                  all_keys=args.all_keys, families=args.families, block_starts=starts),
                  model=scorer.describe(), elapsed_seconds=time.perf_counter()-started)
    write_json(args.json, report)
    best = report['top_candidates'][0]
    print(f'Best candidate: family {best["family"]}, key {best["key"]:05d}; scores={best["margins"]}')
    print(f'Ranked hypotheses only; report: {args.json}')


def plugin(args):
    if args.mask is not None:
        if not 0 <= args.mask <= 0x7fff or args.offset is None or not 0 <= args.offset < 32767:
            raise ValueError('arbitrary mask requires --mask 0..32767 and --offset 0..32766')
        if args.initialization != 'offset' or args.skip != 0 or args.reset != 'burst':
            raise ValueError('arbitrary mask plugin currently uses offset initialization and burst reset with zero skip')
        source = ('"""Experimental explicitly selected mask and additive offset."""\n'
                  'from std_t98_cracker.enhanced_sequence import arbitrary_pn\n'
                  'def generate(key):\n'
                  '    if not 0 <= key <= 32767:\n'
                  '        raise ValueError("key must be 0..32767")\n'
                  f'    state = (key + {args.offset} - 1) % 32767 + 1 if key else 0\n'
                  f'    return arbitrary_pn({args.mask}, state)\n')
    else:
        if args.offset is not None:
            raise ValueError('--offset is only used with --mask')
        seq = CandidateSequence(args.family, args.initialization, args.skip, args.reset)
        source = ('"""Experimental sequence for an explicitly selected candidate."""\n'
                  'from std_t98_cracker.enhanced_sequence import CandidateSequence\n'
                  f'_candidate = CandidateSequence({seq.family}, {seq.initialization!r}, {seq.skip}, {seq.reset!r})\n'
                  'generate = _candidate.generate\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(source)
    print(args.output)


def summarize(args):
    rows = {i: [] for i in range(1, 16)}
    seen = set()
    for path in args.reports:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('schema') != 'enhanced-probe-v1':
            raise ValueError(f'{path.name}: not an enhanced probe report')
        series = report['radio_series']
        if series not in rows:
            continue  # Normal privacy/control probes do not label enhanced settings.
        identity = (report['input']['sha256'], series, report['radio_key'],
                    json.dumps(report['parameters'], sort_keys=True), report['model']['model_sha256'])
        if identity in seen:
            continue
        seen.add(identity)
        best = report['top_candidates'][0]
        p = report['parameters']
        # An unknown-key success supports a recurrence, but does not verify the displayed-key mapping.
        key_matches = best['key'] == report['radio_key'] and p['initialization'] == 'offset'
        rows[series].append(dict(input_sha256=report['input']['sha256'], radio_key=report['radio_key'],
                                 family=best['family'], candidate_key=best['key'],
                                 initialization=p['initialization'], skip=p['skip'], reset=p['reset'],
                                 key_mapping_matches=key_matches, all_blocks_plain=best['all_blocks_plain'],
                                 plaintext_control_positive=report['plaintext_control']['all_blocks_plain'],
                                 minimum_margin=best['minimum_margin']))
    coverage = []
    for series, observations in rows.items():
        signatures = {(r['family'], r['initialization'], r['skip'], r['reset']) for r in observations}
        keys = sorted({r['radio_key'] for r in observations})
        coverage.append(dict(series=series, keys=keys, distinct_captures=len({r['input_sha256'] for r in observations}),
                             conflicting_candidates=len(signatures) > 1, observations=observations, confirmed=False))
        print(f'Series {series:02d}: {len(keys)} keys, {coverage[-1]["distinct_captures"]} captures, candidates={sorted(signatures)}')
    write_json(args.json, dict(schema='enhanced-coverage-v1', coverage=coverage, confirmed=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    def hypothesis(p):
        p.add_argument('--initialization', choices=['offset', 'state'], default='offset')
        p.add_argument('--skip', type=int, default=0)
        p.add_argument('--reset', choices=['burst', 'frame'], default='burst')
    def gpu(p):
        p.add_argument('--device', default='cuda')
        p.add_argument('--batch-size', type=int, default=512)
    p = commands.add_parser('probe', help='compare every candidate with the heavy model on every selected block')
    p.add_argument('input', type=Path)
    p.add_argument('--llr', type=Path)
    label = p.add_mutually_exclusive_group()
    label.add_argument('--setting', type=setting)
    label.add_argument('--key', type=int)
    p.add_argument('--all-keys', action='store_true')
    p.add_argument('--families', type=integer_list, default=list(range(17)))
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--block-starts', type=integer_list)
    p.add_argument('--top', type=int, default=10)
    p.add_argument('--json', type=Path, required=True)
    hypothesis(p)
    gpu(p)
    p.set_defaults(run=probe)
    p = commands.add_parser('plugin', help='export a selected candidate for the existing descramble command')
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--family', type=int)
    choice.add_argument('--mask', type=lambda value: int(value, 0), help='15-bit feedback mask (without the leading polynomial bit)')
    p.add_argument('--offset', type=int)
    p.add_argument('-o', '--output', type=Path, required=True)
    hypothesis(p)
    p.set_defaults(run=plugin)
    p = commands.add_parser('summary', help='compare recordings across all 15 radio settings')
    p.add_argument('reports', type=Path, nargs='+')
    p.add_argument('--json', type=Path, required=True)
    p.set_defaults(run=summarize)
    p = commands.add_parser('benchmark', help='tune batch size and measure sustained FP32 hybrid throughput')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batches', type=integer_list, default=[128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--json', type=Path, required=True)
    from .enhanced_benchmark import benchmark
    p.set_defaults(run=benchmark)
    args = parser.parse_args(argv)
    if getattr(args, 'top', 1) < 1:
        parser.error('--top must be positive')
    try:
        args.run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
