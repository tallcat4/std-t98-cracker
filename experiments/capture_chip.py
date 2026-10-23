"""Explicit ThumbDV corpus capture for solver development (not a product command).

Requires numpy, scipy, soundfile, pyserial and the user's neural-ambe2 driver.
Stores actual chip frames and input PCM; never calls a software AMBE encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--driver-root', type=Path, required=True)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--minutes-per-group', type=float, default=8)
    ap.add_argument('--seed', type=int, default=98023)
    args = ap.parse_args()
    sys.path.insert(0, str(args.driver_root.resolve()))
    from neural_ambe.chip import open_chip
    from neural_ambe.audio import load_8k, to_chip_pcm
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    ja = sorted((args.source_root/'jsut_ver1.1/basic5000/wav').glob('*.wav'))
    rng.shuffle(ja)
    groups = {
        ('train', 'jsut'): ja[:len(ja)//2],
        ('eval', 'jsut'): ja[len(ja)//2:],
        ('train', 'libri'): sorted((args.source_root/'LibriSpeech/dev-clean').glob('*/*/*.flac')),
        ('eval', 'libri'): sorted((args.source_root/'LibriSpeech/test-clean').glob('*/*/*.flac')),
    }
    plan = []
    for (split, corpus), files in groups.items():
        rng.shuffle(files)
        seconds = 0
        for source in files:
            info = sf.info(str(source))
            duration = min(info.duration, 20.0)
            if duration < 2:
                continue
            uid = source.relative_to(args.source_root).as_posix()
            ident = hashlib.sha256(uid.encode()).hexdigest()[:16]
            plan.append(dict(id=ident, split=split, corpus=corpus, source=uid,
                             duration=duration, level_db=int(rng.choice([-20, -26, -32])),
                             ecmode=int(rng.choice([0, 0x0800]))))
            seconds += duration
            if seconds >= args.minutes_per_group * 60:
                break
        if seconds < args.minutes_per_group * 60:
            raise RuntimeError(f'Insufficient source audio for {split}/{corpus}: {seconds}s')
    plan_file = args.output/'plan.json'
    plan_text = json.dumps(plan, indent=2)
    if plan_file.exists() and plan_file.read_text() != plan_text:
        raise RuntimeError('Existing plan differs; choose a new output directory')
    plan_file.write_text(plan_text)
    print(f'{len(plan)} utterances, {sum(p["duration"] for p in plan)/60:.2f} natural minutes', flush=True)
    started = time.perf_counter()
    with open_chip(args.port) as chip:
        device = dict(product=chip.prodid(), version=chip.version(), port=args.port, rate=34,
                      seed=args.seed, driver_sha256=hashlib.sha256((args.driver_root/'neural_ambe/chip.py').read_bytes()).hexdigest())
        (args.output/'device.json').write_text(json.dumps(device, indent=2))
        for i, row in enumerate(plan):
            target = args.output/(row['id']+'.npz')
            if target.exists():
                continue
            source = args.source_root/row['source']
            audio = load_8k(source)[:int(row['duration']*8000)]
            # Natural duration excludes rounding and the separate 1-second zero control.
            natural_frames = len(audio)//160
            pcm = to_chip_pcm(audio, row['level_db'], pad_frames=50)
            chip.reset()
            chip.set_ecmode(row['ecmode'])
            bits = chip.encode(pcm)
            if chip.nbits != 49 or bits.shape != (len(pcm)//160, 7) or np.any(bits[:, -1] & 127):
                raise RuntimeError('Unexpected chip frame count/format/padding')
            metadata = dict(row, natural_frames=natural_frames, frames=len(bits),
                            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            temporary = target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, bits=bits, pcm=pcm, metadata=json.dumps(metadata))
            temporary.replace(target)
            print(f'{i+1}/{len(plan)} {row["split"]}/{row["corpus"]} '
                  f'{natural_frames*.02:.2f}s natural, {len(bits)} frames, '
                  f'elapsed {time.perf_counter()-started:.1f}s', flush=True)
    print('Capture complete; all outputs are from the hardware encoder.', flush=True)


if __name__ == '__main__':
    main()
