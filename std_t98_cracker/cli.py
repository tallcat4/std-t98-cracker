"""Small offline command interface. Key search never modifies the capture."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys

import numpy as np

from .burst import read_bursts, read_llr, unpack49, pack49, write_bursts
from .fec import FecDecoder
from .linear import SilenceSolver, DEFAULT_THRESHOLD, DEFAULT_MARGIN
from .sequence import load_sequence


def _decoded(args):
    bits, payload = read_bursts(args.input)
    if bits == 49:
        if args.llr:
            raise ValueError("--llr only applies to 72-bit input")
        return unpack49(payload), np.ones(payload.shape[:2], dtype=bool), 0
    llr = read_llr(args.llr, len(payload)) if args.llr else None
    frames, valid, corrected = FecDecoder().fec(payload, llr)
    return frames, valid, int(np.maximum(corrected, 0).sum())


def _write_json(path, report):
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path:
        with Path(path).open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(text, end="")


def _crack(args):
    automatic_series = args.series == 'auto'
    sequence = None if automatic_series else load_sequence(args.sequence, args.series)
    frames, valid, corrected = _decoded(args)
    if automatic_series:
        from .series_solver import SeriesSolver
        solver = SeriesSolver()
    else:
        solver = SilenceSolver(sequence) if args.method != "ai" else None
    verification = None
    if args.method == 'guided-ai':
        from .enhanced import HeavyScorer
        from .guided_ai import GuidedAISolver
        if not automatic_series and not args.sequence:
            from .series_solver import SeriesSolver
            solver = SeriesSolver(series=args.series)
        model_path = Path(args.models_dir)/'ambe2_hybrid.safetensors' if args.models_dir else None
        heavy = HeavyScorer(device=args.device or 'auto', model_path=model_path, threads=args.threads)
        verification = heavy.describe()
        solver = GuidedAISolver(solver, heavy, sequence=sequence,
                                series=None if automatic_series or args.sequence else args.series)
    ai = None
    windows = []
    # A linear input is one caller-delimited session unless windows are requested.
    window_size = args.window or len(frames)
    with (Path(args.trace).open('x', encoding='utf-8') if args.trace else nullcontext()) as trace:
        for start in range(0, len(frames), window_size):
            end = min(len(frames), start + window_size)
            chunk, good = frames[start:end], valid[start:end]
            key, analysis, first_confirmation = None, None, None
            if solver:
                session = solver.new_session(args.threshold, args.margin)
                for offset, (burst, fec_good) in enumerate(zip(chunk, good)):
                    analysis = session.push(burst, fec_good)
                    if first_confirmation is None and analysis['key'] is not None:
                        first_confirmation = dict(key=analysis['key'], end_burst=start+offset+1,
                                                  traffic_seconds=(offset+1)*.08,
                                                  score=analysis['score'], margin=analysis['margin'])
                        if automatic_series or args.method == 'guided-ai':
                            first_confirmation['series'] = analysis['series']
                    if trace:
                        trace.write(json.dumps(dict(first_burst=start, end_burst=start+offset+1, **analysis))+'\n')
                key = analysis['key']
            method = args.method if key is not None else None
            if args.method == "ai":
                # Models were trained on 5-burst blocks, not mixtures across gaps.
                # Skip corrupt bursts without compressing time or changing phase.
                run_start = next((i for i in range(max(0, len(chunk)-4)) if good[i:i+5].all()), None)
                if run_start is not None:
                    if ai is None:
                        import torch
                        from .ai import SecretCracker
                        torch.set_num_threads(args.threads)
                        model_dir = Path(args.models_dir) if args.models_dir else Path(__file__).resolve().parent / "models"
                        ai = SecretCracker(model_dir/"ambe2_ffnn.safetensors", model_dir/"ambe2_hybrid.safetensors",
                                           generate=sequence.bits, device=args.device or 'cpu')
                    count = 5
                    if len(chunk) >= run_start+10 and good[run_start:run_start+10].all():
                        count = 10
                    ai.global_cache.clear()  # Each requested window is an independent search.
                    result = ai.resolve_key(0, pack49(chunk[run_start:run_start+count]).tobytes(), count)
                    if result.resolved_key:
                        key, method = result.resolved_key, "ai-candidate"
            window = {"first_burst": start, "end_burst": end, "key": key, "method": method}
            window['series'] = (analysis['series'] if automatic_series else args.series if not args.sequence else None)
            if analysis is not None:
                window.update(candidate=analysis['candidate'], score=analysis['score'], margin=analysis['margin'],
                              threshold=analysis['threshold'], required_margin=analysis['required_margin'], first_confirmation=first_confirmation)
                if automatic_series:
                    window['candidate_series'] = analysis['candidate_series']
                if args.method == 'guided-ai':
                    window.update(linear=analysis['linear'], verification=analysis['verification'], candidates=analysis['candidates'])
            windows.append(window)
            print(f"bursts {start}:{end}: {key if key is not None else 'no match'} ({method or args.method})", file=sys.stderr)
    keys = sorted({w["key"] for w in windows if w["key"] is not None})
    report = {
        "input": str(Path(args.input)), "sequence": 'builtin-series-auto' if automatic_series else sequence.name, "bursts": len(frames),
        "uncorrectable_frames": int((~valid).sum()), "corrected_bits": corrected,
        "keys": keys, "windows": windows,
        "solutions": [dict(series=series, key=key) for series, key in sorted(
            {(w['series'], w['key']) for w in windows if w['key'] is not None})],
        "note": "Linear scores are evidence units, not probabilities. First confirmation can be retracted by later evidence; key is the final session decision. AI results are candidates. Key 0 means plaintext.",
    }
    if verification:
        report['verifier'] = verification
        report['note'] += ' Solver-guided AI uses only the heavy FP32 hybrid to judge the linear shortlist; its scores are logit margins, not probabilities.'
    _write_json(args.json, report)
    return 0 if keys else 2


def _descramble(args):
    if not 0 <= args.key <= 32767:
        raise ValueError("--key must be 0..32767 (0 = plaintext)")
    sequence = load_sequence(args.sequence, args.series)
    frames, valid, _ = _decoded(args)
    plain = frames ^ sequence.thumbdv(args.key)[None, :, :]
    base = Path(args.output)
    burst_path = Path(str(base) + ".burst")
    if burst_path.exists():
        raise ValueError(f"Output exists: {burst_path}; choose another --output")
    base.parent.mkdir(parents=True, exist_ok=True)
    write_bursts(burst_path, pack49(plain), bits=49)
    print(f"Wrote {len(plain)} decrypted bursts to {burst_path}")
    if not valid.all():
        print(f"{int((~valid).sum())} frames had uncorrectable FEC; their payloads remain unreliable.", file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline STD-T98 .burst key analysis (no audio or automatic fallback)")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Check the burst format without a codec or model")
    inspect.add_argument("input")
    for name, help_text in (("crack", "Run the explicitly selected key-search method"), ("descramble", "Apply an explicit key to payload bits")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", help=".burst file (72-bit or 49-bit ThumbDV frames)")
        command.add_argument("--llr", help="Optional receiver .llr file for soft FEC")
        sequence = command.add_mutually_exclusive_group()
        sequence.add_argument("--series", type=lambda v: 'auto' if v == 'auto' else int(v),
                              choices=('auto', *range(16)) if name == 'crack' else range(16), default=0,
                              help="0: standard (default); 1..15: Alinco enhanced; auto: linear comparison of all series (crack only)")
        sequence.add_argument("--sequence", help="Python file defining generate(key) -> 196 bits")
        if name == "crack":
            command.add_argument("--method", choices=("linear", "guided-ai", "ai"), default="linear",
                                 help="linear: constraints; guided-ai: solver shortlist + heavy hybrid; ai: full key search")
            command.add_argument("--window", type=int, help="Bursts per independent session (default: entire file; method ai: 10)")
            command.add_argument("--threshold", type=float, help="Linear evidence threshold (default 20; auto linear 40); shortlist floor for guided-ai")
            command.add_argument("--margin", type=float, help="Required lead (default 6; auto linear 20); hybrid logit units for guided-ai")
            command.add_argument("--trace", help="Write each incremental decision to a new JSONL file (linear/guided-ai)")
            command.add_argument("--models-dir")
            command.add_argument("--threads", type=int, default=4)
            command.add_argument("--device", choices=('cpu', 'cuda'), help="Force AI device (guided-ai: automatic; ai: CPU by default)")
            command.add_argument("--json", help="Also save results to a new JSON file")
        else:
            command.add_argument("--key", required=True, type=int)
            command.add_argument("-o", "--output", required=True, help="Output basename; writes 49-bit .burst")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            bits, payload = read_bursts(args.input)
            print(json.dumps({"bits_per_frame": bits, "bursts": len(payload), "frames": len(payload)*4, "traffic_seconds": len(payload)*.08}, indent=2))
            return 0
        if args.command == "crack":
            if args.device and args.method == 'linear':
                raise ValueError('--device applies only to guided-ai or ai')
            if args.series == 'auto' and args.method == 'ai':
                raise ValueError('--series auto supports linear and guided-ai; select a series for full AI search')
            if args.method == 'ai' and (args.threshold is not None or args.margin is not None or args.trace):
                raise ValueError('--threshold, --margin and --trace apply only to --method linear')
            if args.window is None and args.method == 'ai':
                args.window = 10
            if (args.window is not None and not 1 <= args.window <= 10000) or not 1 <= args.threads <= 256:
                raise ValueError("--window must be 1..10000 and --threads 1..256")
            threshold, margin = DEFAULT_THRESHOLD, DEFAULT_MARGIN
            if args.series == 'auto' and args.method == 'linear':
                from .series_solver import DEFAULT_THRESHOLD as threshold, DEFAULT_MARGIN as margin
            args.threshold = threshold if args.threshold is None else args.threshold
            args.margin = margin if args.margin is None else args.margin
            if not math.isfinite(args.threshold) or args.threshold <= 0 or not math.isfinite(args.margin) or args.margin < 0:
                raise ValueError('--threshold must be finite and positive; --margin must be finite and nonnegative')
            if args.json and Path(args.json).exists():
                raise ValueError("JSON output already exists; choose a new path")
            if args.trace and Path(args.trace).exists():
                raise ValueError('Trace output already exists; choose a new path')
            if args.json and args.trace and Path(args.json).resolve() == Path(args.trace).resolve():
                raise ValueError('--json and --trace must use different paths')
            return _crack(args)
        return _descramble(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
