"""Blind linear-series evaluation; labels are used only after solver decisions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from std_t98_cracker.burst import read_bursts, read_llr, unpack49
from std_t98_cracker.fec import FecDecoder
from std_t98_cracker.series_solver import SeriesSolver, DEFAULT_THRESHOLD, DEFAULT_MARGIN


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def solve(frames, valid, solver, threshold, margin):
    # The solver receives only payload/FEC validity, never the path or labels.
    session = solver.new_session(threshold, margin)
    first, decisions, step_ms = None, [], []
    if len(frames) == 0:
        raise ValueError('Evaluation requires at least one burst')
    started = time.perf_counter()
    for index, (burst, good) in enumerate(zip(frames, valid)):
        step_started = time.perf_counter()
        result = session.push(burst, good)
        step_ms.append((time.perf_counter() - step_started) * 1000)
        if result['key'] is not None:
            decision = dict(series=result['series'], key=result['key'],
                            end_burst=index+1, traffic_seconds=(index+1)*.08)
            if first is None:
                first = decision
            if not decisions or (decision['series'], decision['key']) != (decisions[-1]['series'], decisions[-1]['key']):
                decisions.append(decision)
    return dict(final=result, first_confirmation=first, accepted_pairs=decisions,
                compute_seconds=time.perf_counter()-started, step_ms=step_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='JSON list: input, optional llr, series, key; paths relative to manifest')
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--method', choices=('linear', 'guided-ai'), default='linear')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--margin', type=float)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto', help='guided-ai device')
    parser.add_argument('--threads', type=int, default=4, help='PyTorch CPU threads')
    args = parser.parse_args()
    if args.json.exists():
        raise FileExistsError(args.json)
    entries = json.loads(args.manifest.read_text(encoding='utf-8'))
    solver, fec, rows = SeriesSolver(), FecDecoder(), []
    verification = None
    threshold, margin = DEFAULT_THRESHOLD, DEFAULT_MARGIN
    if args.method == 'guided-ai':
        from std_t98_cracker.enhanced import HeavyScorer
        from std_t98_cracker.guided_ai import GuidedAISolver
        if not 1 <= args.threads <= 256:
            raise ValueError('--threads must be 1..256')
        scorer = HeavyScorer(device=args.device, threads=args.threads)
        verification = scorer.describe()
        solver = GuidedAISolver(solver, scorer)
        threshold, margin = 20.0, 6.0
    args.threshold = threshold if args.threshold is None else args.threshold
    args.margin = margin if args.margin is None else args.margin
    for entry in entries:
        path = args.manifest.parent / entry['input']
        llr_path = args.manifest.parent / entry['llr'] if entry.get('llr') else None
        bits, payload = read_bursts(path)
        if bits == 72:
            frames, valid, _ = fec.fec(payload, read_llr(llr_path, len(payload)) if llr_path else None)
        else:
            import numpy as np
            if llr_path:
                raise ValueError('LLR applies only to 72-bit input')
            frames, valid = unpack49(payload), np.ones(payload.shape[:2], dtype=bool)
        result = solve(frames, valid, solver, args.threshold, args.margin)
        expected = (int(entry['series']), int(entry['key']))
        if not 0 <= expected[0] <= 15 or not 0 <= expected[1] <= 32767:
            raise ValueError('Invalid evaluation label')
        def outcome(decision):
            if decision is None or decision['key'] is None:
                return 'unresolved'
            return 'correct' if (decision['series'], decision['key']) == expected else 'wrong'
        row = dict(input=path.name, input_sha256=sha(path), llr_sha256=sha(llr_path) if llr_path else None,
                   expected_series=expected[0], expected_key=expected[1],
                   invalid_frames=int((~valid).sum()), **result,
                   final_outcome=outcome(result['final']), first_outcome=outcome(result['first_confirmation']),
                   any_wrong_confirmation=any(outcome(d) == 'wrong' for d in result['accepted_pairs']))
        rows.append(row)
        print(path.name, row['final_outcome'], 'first', row['first_outcome'], 'any_wrong', row['any_wrong_confirmation'], flush=True)
    summary = {phase: {status: sum(r[f'{phase}_outcome'] == status for r in rows)
                       for status in ('correct', 'unresolved', 'wrong')} for phase in ('final', 'first')}
    summary['any_wrong_confirmation'] = sum(r['any_wrong_confirmation'] for r in rows)
    summary['correct_first_within_seconds'] = {str(seconds): sum(
        r['first_outcome'] == 'correct' and r['first_confirmation']['traffic_seconds'] <= seconds
        for r in rows) for seconds in (1, 3)}
    code_root = Path(__file__).resolve().parents[1]
    report = dict(schema='blind-linear-series-v1', created_utc=datetime.now(timezone.utc).isoformat(),
                  method=args.method, series=list(range(16)), ai_used=verification is not None,
                  verifier=verification, threads=args.threads, labels_used_by_solver=False,
                  threshold=args.threshold, margin=args.margin, manifest_sha256=sha(args.manifest),
                  source_sha256={name: sha(code_root/'std_t98_cracker'/name) for name in
                                 ('linear.py', 'series_solver.py', 'guided_ai.py', 'enhanced.py', 'sequence.py', 'enhanced_sequence.py')},
                  summary=summary, captures=rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    with args.json.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
