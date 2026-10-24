"""Blind series/state recovery from weighted GF(2) temporal constraints.

No speech model, known code, filename, or silence template is used. A Walsh-
Hadamard transform scores all 15-bit states for the 16 supported polynomials.
Evidence is a heuristic, not a calibrated probability (speech bits correlate).
"""
import math

import numpy as np

from .enhanced_sequence import MASKS, CandidateSequence, coefficient_table

DEFAULT_THRESHOLD = 40.0
DEFAULT_MARGIN = 20.0


def _fwht(values):
    """Unnormalized Walsh-Hadamard transform along the last dimension."""
    result = values.copy()
    width = 1
    while width < result.shape[-1]:
        blocks = result.reshape(-1, result.shape[-1] // (2 * width), 2 * width)
        left, right = blocks[:, :, :width].copy(), blocks[:, :, width:].copy()
        blocks[:, :, :width] = left + right
        blocks[:, :, width:] = left - right
        width *= 2
    return result


def _evidence(score, count):
    # One-sided agreement evidence against fair random transitions, minus the
    # log2(16 * 32768) search size. Correlated bits invalidate a probability
    # interpretation. Keeping this separate exposes the acceptance controls.
    if count == 0:
        return 0.0
    p = min(1 - 1e-12, max(.5, .5 + score / (2 * count)))
    return max(0.0, count * (1 + p * math.log2(p) + (1-p) * math.log2(1-p)) - 19)


class SeriesSolver:
    """Search standard privacy plus enhanced series 1..15, without AI."""

    def __init__(self, series=None):
        if series is not None and (not isinstance(series, int) or not 0 <= series <= 15):
            raise ValueError('series must be None (all) or 0..15')
        self.series = series
        coefficients = coefficient_table(np.asarray(MASKS[:16]) & 32767).reshape(16, 4, 49)
        # PN resets each burst. Include phase 3 -> 0 across adjacent bursts.
        self.differences = coefficients ^ np.roll(coefficients, 1, axis=1)
        self.sequences = [CandidateSequence(i) for i in range(16)]

    def new_session(self, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
        return SeriesSession(self, threshold, margin)

    def assess(self, frames, valid=None, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
        session = self.new_session(threshold, margin)
        for index, burst in enumerate(frames):
            session._add(burst, None if valid is None else valid[index])
        return session.snapshot()


class SeriesSession:
    """One caller-delimited PTT/session; reset before a new transmission."""

    def __init__(self, solver, threshold, margin):
        if not math.isfinite(threshold) or threshold <= 0 or not math.isfinite(margin) or margin < 0:
            raise ValueError('threshold must be finite and positive; margin finite and nonnegative')
        self.solver, self.threshold, self.margin = solver, threshold, margin
        self.reset()

    def reset(self):
        self.sums = np.zeros((4, 49), dtype=np.float64)
        self.pairs = 0
        self.bursts = 0
        self.previous = None

    def _add(self, burst, valid=None):
        burst = np.asarray(burst)
        good = np.ones(4, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if burst.shape != (4, 49) or np.any((burst != 0) & (burst != 1)) or good.shape != (4,):
            raise ValueError('Expected one binary (4, 49) burst and four FEC validity flags')
        for phase, frame in enumerate(burst):
            if good[phase] and self.previous is not None:
                self.sums[phase] += 1 - 2 * (frame.astype(np.uint8) ^ self.previous).astype(np.float64)
                self.pairs += 1
            # Never bridge an invalid frame; retain actual burst phase.
            self.previous = frame.astype(np.uint8).copy() if good[phase] else None
        self.bursts += 1

    def push(self, burst, valid=None):
        self._add(burst, valid)
        return self.snapshot()

    def snapshot(self):
        candidates = []
        if self.pairs:
            spectrum = np.zeros((16, 32768), dtype=np.float64)
            for series in range(16):
                np.add.at(spectrum[series], self.solver.differences[series].ravel(), self.sums.ravel())
            scores = _fwht(spectrum)
            # State zero produces the same plaintext for every polynomial.
            if self.solver.series is None:
                scores[1:, 0] = -np.inf
            else:
                scores[np.arange(16) != self.solver.series] = -np.inf
            flat = scores.ravel()
            indices = np.argpartition(flat, -5)[-5:]
            indices = indices[np.lexsort((indices, -flat[indices]))]
            for index in indices:
                series, state = divmod(int(index), 32768)
                candidates.append(dict(series=series, key=self.solver.sequences[series].key_for_state(state),
                                       score=_evidence(float(flat[index]), self.pairs * 49)))
        top = candidates[0] if candidates else None
        runner = candidates[1]['score'] if len(candidates) > 1 else 0.0
        gap = top['score'] - runner if top else 0.0
        accepted = bool(top and top['score'] >= self.threshold and gap >= self.margin)
        return dict(key=top['key'] if accepted else None,
                    series=top['series'] if accepted else None,
                    candidate=top['key'] if top else None,
                    candidate_series=top['series'] if top else None,
                    score=top['score'] if top else 0.0, margin=gap,
                    threshold=self.threshold, required_margin=self.margin,
                    bursts=self.bursts, traffic_seconds=self.bursts * .08,
                    transition_pairs=self.pairs, candidates=candidates)
