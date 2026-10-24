"""Solver-guided AI: linear shortlist -> heavy FP32 hybrid verdict."""
from collections import deque
import math

import numpy as np

from .sequence import load_sequence

DEFAULT_THRESHOLD = 20.0
DEFAULT_MARGIN = 6.0


class GuidedAISolver:
    """The scorer only evaluates up to five linearly selected candidates.

    threshold gates linear candidate evidence; margin is the required hybrid
    logit lead over the next candidate (or zero). Neither is a probability.
    """

    def __init__(self, linear_solver, scorer, sequence=None, series=None):
        self.linear_solver, self.scorer = linear_solver, scorer
        self.sequence, self.series = sequence, series

    def new_session(self, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
        return GuidedAISession(self, threshold, margin)


class GuidedAISession:
    def __init__(self, solver, threshold, margin):
        if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(margin) or margin < 0:
            raise ValueError('threshold must be positive; margin nonnegative; both finite')
        self.solver, self.threshold, self.margin = solver, threshold, margin
        self.reset()

    def reset(self):
        self.linear = self.solver.linear_solver.new_session(self.threshold, 0)
        self.pending = []
        self.blocks = deque(maxlen=3)
        self.cache = {}
        self.evaluations = 0

    def push(self, burst, valid=None):
        linear = self.linear.push(burst, valid)
        good = np.ones(4, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if good.all():
            self.pending.append(np.asarray(burst, dtype=np.uint8).copy())
            if len(self.pending) == 5:
                start = linear['bursts'] - 5
                self.blocks.append((start, np.stack(self.pending)))
                self.pending.clear()
                retained = {s for s, _ in self.blocks}
                self.cache = {k: v for k, v in self.cache.items() if k[0] in retained}
        else:
            # Do not compress time or stitch the two sides of a corrupt burst.
            self.pending.clear()
        candidates = [dict(c, series=c.get('series', self.solver.series))
                      for c in linear['candidates'][:5] if c['score'] >= self.threshold]
        ranking = []
        if len(self.blocks) >= 2 and candidates:
            for start, block in self.blocks:
                missing = [c for c in candidates if (start, c['series'], c['key']) not in self.cache]
                if missing:
                    masks = np.stack([(self.solver.sequence or load_sequence(series=c['series'])).thumbdv(c['key'])
                                      for c in missing])
                    margins = np.asarray(self.solver.scorer.score(block[None], masks))[:, 0]
                    if not np.isfinite(margins).all():
                        raise RuntimeError('Hybrid produced non-finite logits')
                    for c, score in zip(missing, margins):
                        self.cache[start, c['series'], c['key']] = float(score)
                    self.evaluations += len(missing)
            for c in candidates:
                scores = [self.cache[start, c['series'], c['key']] for start, _ in self.blocks]
                ranking.append(dict(series=c['series'], key=c['key'], linear_score=c['score'],
                                    block_margins=scores, score=float(np.mean(scores)),
                                    positive_blocks=sum(s > 0 for s in scores)))
            ranking.sort(key=lambda c: (-c['score'], c['key']))
        top = ranking[0] if ranking else None
        runner = max(0.0, ranking[1]['score']) if len(ranking) > 1 else 0.0
        gap = top['score'] - runner if top else 0.0
        accepted = bool(top and top['score'] > 0 and gap >= self.margin
                        and top['positive_blocks'] >= math.ceil(2 * len(self.blocks) / 3))
        # The final decision belongs to the hybrid stage. A linear-only key is
        # deliberately never used here, even when its score is very strong.
        return dict(key=top['key'] if accepted else None, series=top['series'] if accepted else None,
                    candidate=top['key'] if top else None, candidate_series=top['series'] if top else None,
                    score=top['score'] if top else 0.0, margin=gap, threshold=0.0, required_margin=self.margin,
                    bursts=linear['bursts'], traffic_seconds=linear['traffic_seconds'],
                    candidates=ranking, linear=linear,
                    verification=dict(model='hybrid', dtype='float32', lightweight_model_used=False,
                                      linear_candidate_threshold=self.threshold, block_starts=[s for s, _ in self.blocks],
                                      required_positive_blocks=max(2, math.ceil(2 * len(self.blocks) / 3)),
                                      evaluated_candidate_blocks=self.evaluations))
