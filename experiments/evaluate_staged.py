"""Validate spaced-stage retention on natural ThumbDV utterances."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from std_t98_cracker.enhanced import HeavyScorer, write_json
from std_t98_cracker.ai import _load_checkpoint
from std_t98_cracker.enhanced_staged import spaced_blocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    items, inputs, skipped = [], [], 0
    for path in sorted(args.corpus.glob('*.npz')):
        with np.load(path) as data:
            meta = json.loads(str(data['metadata']))
            count = meta['natural_frames']//4*4
            frames = np.unpackbits(data['bits'], axis=-1)[:count, :49].reshape(-1, 4, 49)
            try:
                blocks, plan = spaced_blocks(frames, np.ones((len(frames), 4), dtype=bool))
            except ValueError:
                skipped += 1
                continue
            items.append(dict(sample=meta['id'], split=meta['split'], corpus=meta['corpus'], plan=plan))
            inputs.append(blocks.reshape(7, 20, 49))
    scorer = HeavyScorer()
    light, _ = _load_checkpoint(Path(__file__).resolve().parents[1]/'std_t98_cracker/models/ambe2_ffnn.safetensors', scorer.device)
    values = torch.as_tensor(np.stack(inputs).reshape(-1, 20, 49), dtype=torch.float32, device=scorer.device)
    heavy_scores, light_scores = [], []
    with torch.inference_mode():
        for part in values.split(1024):
            h = scorer.model(part)
            l = light(part.flatten(1))
            heavy_scores.extend((h[:, 0]-h[:, 1]).cpu().tolist())
            light_scores.extend((l[:, 0]-l[:, 1]).cpu().tolist())
    heavy_scores = np.array(heavy_scores).reshape(-1, 7)
    light_scores = np.array(light_scores).reshape(-1, 7)
    for item, h, l in zip(items, heavy_scores, light_scores):
        item.update(light_margins=l[:2].tolist(), hybrid_margins=h[2:4].tolist(), final_margins=h[4:].tolist(),
                    light_pass_zero=bool(max(l[:2]) >= 0), light_pass_minus10=bool(max(l[:2]) >= -10),
                    hybrid_pass=bool(max(h[2:4]) >= 0), final_pass=bool(sum(h[4:] > 0) >= 2),
                    baseline_all_three=bool((h[4:] > 0).all()))
    stats = {}
    for split in ('train', 'eval'):
        subset = [r for r in items if r['split'] == split]
        stats[split] = dict(utterances=len(subset),
            light_zero_misses=sum(not r['light_pass_zero'] for r in subset),
            light_minus10_misses=sum(not r['light_pass_minus10'] for r in subset),
            hybrid_misses=sum(not r['hybrid_pass'] for r in subset),
            final_misses=sum(not r['final_pass'] for r in subset),
            baseline_all_three_misses=sum(not r['baseline_all_three'] for r in subset),
            complete_zero_pass=sum(r['light_pass_zero'] and r['hybrid_pass'] and r['final_pass'] for r in subset),
            complete_minus10_pass=sum(r['light_pass_minus10'] and r['hybrid_pass'] and r['final_pass'] for r in subset))
    report = dict(stats=stats, skipped_short=skipped, observations=items, model=scorer.describe(),
                  limitation='Natural plaintext, not RF errors. Retention is not unique-key identification. Some utterances are too short for this protocol.')
    write_json(args.output, report)
    print(json.dumps(dict(stats=stats, skipped_short=skipped), indent=2))


if __name__ == '__main__':
    main()
