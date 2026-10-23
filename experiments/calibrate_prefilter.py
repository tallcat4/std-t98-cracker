"""Measure conservative multi-block FFNN retention on local natural chip frames."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from std_t98_cracker.ai import _load_checkpoint
from std_t98_cracker.enhanced import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    path = Path(__file__).resolve().parents[1]/'std_t98_cracker/models/ambe2_ffnn.safetensors'
    model, _ = _load_checkpoint(path, torch.device(args.device))
    observations = []
    with torch.inference_mode():
        for path_npz in sorted(args.corpus.glob('*.npz')):
            with np.load(path_npz) as data:
                metadata = json.loads(str(data['metadata']))
                frames = np.unpackbits(data['bits'], axis=-1)[:metadata['natural_frames'], :49]
                count = len(frames)//60
                if not count:
                    continue
                inputs = torch.as_tensor(frames[:count*60].reshape(-1, 980), device=args.device, dtype=torch.float32)
                logits = model(inputs)
                scores = (logits[:, 0]-logits[:, 1]).reshape(count, 3).cpu().numpy()
                for index, values in enumerate(scores):
                    observations.append(dict(sample=metadata['id'], split=metadata['split'], corpus=metadata['corpus'],
                                             first_frame=index*60, margins=values.tolist(), best_margin=float(values.max())))
    train = [r['best_margin'] for r in observations if r['split'] == 'train']
    # Never require a positive score; stay at least 10 logit units below the
    # worst training window, and use only training data to choose the threshold.
    threshold = min(0.0, min(train))-10.0
    stats = {}
    for split in ('train', 'eval'):
        scores = [r['best_margin'] for r in observations if r['split'] == split]
        stats[split] = dict(windows=len(scores), rejected=sum(s < threshold for s in scores), minimum=max(-1e6, min(scores)))
    result = dict(schema='enhanced-prefilter-calibration-v1', threshold=threshold, rule='retain if any of three blocks passes',
                  model_sha256=sha256(path), stats=stats, observations=observations,
                  limitation='Nonoverlapping natural 1.2-second windows within each recording, no RF errors. '
                  'Zero observed misses is not a guarantee for new audio, FEC errors, or another block count.')
    write_json(args.output, result)
    print(json.dumps(dict(threshold=threshold, stats=stats), indent=2))


if __name__ == '__main__':
    main()
