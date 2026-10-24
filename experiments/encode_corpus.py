"""Build ambe2-c and re-encode stored evaluation PCM; development use only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Clean ambe2-c checkout')
    parser.add_argument('--corpus', type=Path, required=True, help='ThumbDV NPZ corpus with input PCM')
    parser.add_argument('--split', choices=('train', 'eval'), default='eval')
    parser.add_argument('--output', type=Path, required=True, help='New local directory')
    parser.add_argument('--compiler', default=shutil.which('gcc'))
    args = parser.parse_args()
    if not args.compiler:
        raise ValueError('GCC is required; use --compiler')
    source, output = args.source.resolve(), args.output.resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Use a clean encoder source so the revision is reproducible')
    output.mkdir(parents=True, exist_ok=False)
    encoder = source/'encoder'
    suffix = '.exe' if sys.platform == 'win32' else ''
    executable = output/('encode_raw'+suffix)
    sources = ['encoder.c', 'analysis.c', 'quant.c', 'fft.c', 'enc_tables.c', 'fec.c',
               'ap_quant.c', 'ap_fec.c', 'ap_tables.c']
    command = [args.compiler, '-std=c99', '-O2', '-I'+str(encoder/'include'), '-I'+str(encoder/'src'),
               str(encoder/'tests/encode_raw.c'), *[str(encoder/'src'/s) for s in sources], '-lm']
    if sys.platform == 'win32':
        command.append('-static')
    subprocess.run([*command, '-o', str(executable)], check=True)
    vectors = []
    for name in ('en', 'noisy'):
        for background in (False, True):
            target = output/(name+('.background' if background else '')+'.ambe')
            expected = encoder/'tests/vectors'/(name+('.background' if background else '')+'.expected.ambe')
            subprocess.run([str(executable), str(encoder/'tests/vectors'/(name+'.pcm')), str(target),
                            *(['--background'] if background else [])], check=True)
            if target.read_bytes() != expected.read_bytes():
                raise RuntimeError('Encoder reference vector mismatch: '+expected.name)
            vectors.append(dict(name=expected.name, sha256=sha(target)))
    # Check that rate 34 really is the 49-bit ThumbDV order used by the solver.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from std_t98_cracker.burst import unpack49
    from std_t98_cracker.fec import FecDecoder
    rate33 = output/'en.rate33.ambe'
    subprocess.run([str(executable), str(encoder/'tests/vectors/en.pcm'), str(rate33), '--rate', '33'], check=True)
    raw49 = np.frombuffer((output/'en.ambe').read_bytes(), dtype=np.uint8).reshape(-1, 7)
    raw72 = np.frombuffer(rate33.read_bytes(), dtype=np.uint8).reshape(-1, 9)
    n = len(raw49)//4*4
    recovered, valid, _ = FecDecoder().fec(raw72[:n].reshape(-1, 4, 9))
    if not valid.all() or not np.array_equal(recovered, unpack49(raw49[:n].reshape(-1, 4, 7))):
        raise RuntimeError('Rate 33 -> FEC -> rate 34 bit order mismatch')
    started = time.perf_counter()
    rows = []
    pcm_path, bits_path = output/'input.pcm', output/'encoded.ambe'
    for path in sorted(args.corpus.glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['split'] != args.split:
                continue
            frames = int(metadata['natural_frames'])
            pcm = np.asarray(data['pcm'], dtype='<i2').reshape(-1)[:frames*160]
            if len(pcm) != frames*160:
                raise ValueError('Incomplete natural PCM in '+path.name)
        pcm_path.write_bytes(pcm.tobytes())
        subprocess.run([str(executable), str(pcm_path), str(bits_path),
                        *(['--background'] if metadata['ecmode'] & 0x0800 else [])], check=True)
        bits = np.frombuffer(bits_path.read_bytes(), dtype=np.uint8).reshape(-1, 7)
        if len(bits) != frames or np.any(bits[:, -1] & 127):
            raise RuntimeError('Unexpected 49-bit frame count/padding')
        metadata.update(encoder='ambe2-c', encoder_revision=revision, frames=frames,
                        encoder_variant='double', pcm_sha256=sha(pcm_path))
        np.savez_compressed(output/path.name, bits=bits, metadata=json.dumps(metadata))
        rows.append(dict(input=path.name, input_sha256=sha(path), output_sha256=sha(output/path.name), frames=frames))
    report = dict(revision=revision, ratet=34, source=str(source), compiler=args.compiler,
                  compiler_version=subprocess.check_output([args.compiler, '--version'], text=True).splitlines()[0],
                  executable_sha256=sha(executable), vectors=vectors, fec_roundtrip_frames=n,
                  elapsed_seconds=time.perf_counter()-started, split=args.split, files=rows)
    (output/'encoding.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Encoded {len(rows)} utterances, {sum(r["frames"] for r in rows)*.02:.2f}s; vectors and FEC roundtrip passed', flush=True)


if __name__ == '__main__':
    main()
