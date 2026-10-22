"""Small offline command interface. Key search never modifies the capture."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .burst import read_bursts, read_llr, unpack49, pack49, write_bursts
from .fec import FecDecoder
from .linear import SilenceSolver
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
    sequence = load_sequence(args.sequence)
    frames, valid, corrected = _decoded(args)
    solver = SilenceSolver(sequence) if args.method == "linear" else None
    ai = None
    windows = []
    # Non-overlapping windows keep key changes visible and bound AI memory/work.
    for start in range(0, len(frames), args.window):
        end = min(len(frames), start + args.window)
        chunk, good = frames[start:end], valid[start:end]
        key = solver.recover(chunk, good) if solver else None
        method = "linear" if key is not None else None
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
                    ai = SecretCracker(model_dir/"ambe2_ffnn.safetensors", model_dir/"ambe2_hybrid.safetensors", generate=sequence.bits)
                count = 5
                if len(chunk) >= run_start+10 and good[run_start:run_start+10].all():
                    count = 10
                ai.global_cache.clear()  # Each requested window is an independent search.
                result = ai.resolve_key(0, pack49(chunk[run_start:run_start+count]).tobytes(), count)
                if result.resolved_key:
                    key, method = result.resolved_key, "ai-candidate"
        windows.append({"first_burst": start, "end_burst": end, "key": key, "method": method})
        print(f"bursts {start}:{end}: {key if key is not None else 'no match'} ({method or args.method})", file=sys.stderr)
    keys = sorted({w["key"] for w in windows if w["key"] is not None})
    report = {
        "input": str(Path(args.input)), "sequence": sequence.name, "bursts": len(frames),
        "uncorrectable_frames": int((~valid).sum()), "corrected_bits": corrected,
        "keys": keys, "windows": windows,
        "note": "AI results are candidates. Linear recovery requires the measured AMBE-3000R silence profile and evidence at two different TCH phases. Key 0 means plaintext.",
    }
    _write_json(args.json, report)
    return 0 if keys else 2


def _descramble(args):
    if not 0 <= args.key <= 32767:
        raise ValueError("--key must be 0..32767 (0 = plaintext)")
    sequence = load_sequence(args.sequence)
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
        command.add_argument("--sequence", help="Python file defining generate(key) -> 196 bits")
        if name == "crack":
            command.add_argument("--method", choices=("linear", "ai"), default="linear")
            command.add_argument("--window", type=int, default=10, help="Bursts per independent search window (default 10)")
            command.add_argument("--models-dir")
            command.add_argument("--threads", type=int, default=4)
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
            if not 1 <= args.window <= 10000 or not 1 <= args.threads <= 256:
                raise ValueError("--window must be 1..10000 and --threads 1..256")
            if args.json and Path(args.json).exists():
                raise ValueError("JSON output already exists; choose a new path")
            return _crack(args)
        return _descramble(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
