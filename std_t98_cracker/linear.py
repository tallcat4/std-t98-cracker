"""GF(2) recovery using observed AMBE silence, unvoiced and voiced constraints.

Retains SPEAR's strict silence test and adds independently confirmed evidence
across phases and bursts. Unsupported patterns and conflicting supported keys
produce no result. These are conservative speech assumptions, not authentication.
"""
import itertools
import math
from collections import defaultdict

import numpy as np
from .sequence import Sequence, THUMBDV_MAP

POSITIONS = (0,1,2,3,37,38,39,4,5,6,7,35,8,9,10,11,36,28,29,30,45,26)
PLAIN = (1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0)
FIELDS = (
    ((0,1,2,3,37,38,39), {119}), ((4,5,6,7,35), {16}),
    ((8,9,10,11,36), {0}), ((12,13,14,15,16,17,18,19,43), {86,87,90,91,308}),
    ((20,21,22,23,40,41,42), {27,36,78}), ((24,25,26,27,44), {2,8}),
    ((28,29,30,45), {14}), ((31,32,33,46), {1,3,13}), ((34,47,48), {1,3,5,7}),
)


def matches_silence(bits):
    for positions, accepted in FIELDS:
        value = 0
        for pos in positions:
            value = (value << 1) | int(bits[pos])
        if value not in accepted:
            return False
    return True


def linear_equations(sequence):
    basis_pn = np.array([sequence.bits(1 << i) for i in range(15)])
    if np.any(sequence.bits(0)):
        raise ValueError("Linear solver requires a linear sequence with generate(0) == 0")
    # Detect accidental nonlinear generators; candidates are also checked
    # against the actual generator before acceptance.
    for key in (3, 7, 12345, 32767):
        predicted = np.bitwise_xor.reduce(basis_pn[[i for i in range(15) if key & (1 << i)]], axis=0)
        if not np.array_equal(predicted, sequence.bits(key)):
            raise ValueError("Sequence is not linear over GF(2); use AI search or an explicit key")
    return [sum(int(basis_pn[b, t]) << b for b in range(15)) for t in range(196)]


class _StrictSilence:
    def __init__(self, sequence=None):
        self.public_sequence = sequence or Sequence()
        self.sequence = self.public_sequence.linearized()
        equations = linear_equations(self.sequence)
        self.solvers = []
        for phase in range(4):
            eq = [equations[phase * 49 + pos] for pos in POSITIONS]
            basis = [(0, 0)] * 15
            for i, coeff in enumerate(eq):
                obs = 1 << i
                for b in range(14, -1, -1):
                    if not coeff & (1 << b):
                        continue
                    if not basis[b][0]:
                        basis[b] = (coeff, obs)
                        break
                    coeff ^= basis[b][0]
                    obs ^= basis[b][1]
            if any(coeff == 0 for coeff, _ in basis):
                raise ValueError(f"Silence constraints do not determine all 15 key bits at phase {phase}")
            self.solvers.append((eq, basis))

    def candidate(self, thumbdv, phase):
        frame = np.asarray(thumbdv)
        if phase not in range(4) or frame.shape != (49,) or not np.all((frame == 0) | (frame == 1)):
            return None
        d = np.empty(49, dtype=np.uint8)
        d[THUMBDV_MAP] = frame
        observations = sum((int(d[pos]) ^ bit) << i for i, (pos, bit) in enumerate(zip(POSITIONS, PLAIN)))
        equations, basis = self.solvers[phase]
        key = 0
        for b, (coeff, obs) in enumerate(basis):
            if ((obs & observations).bit_count() ^ (coeff & key).bit_count()) & 1:
                key |= 1 << b
        if any(((eq & key).bit_count() & 1) != ((observations >> i) & 1) for i, eq in enumerate(equations)):
            return None
        plain = d ^ self.sequence.bits(key)[phase*49:(phase+1)*49]
        return key if matches_silence(plain) else None



class _ReducedSystem:
    """Eliminate linear key equations, retaining small null spaces explicitly."""

    def __init__(self, equations, max_free=4):
        self.equations = equations
        self.basis = [(0, 0)] * 15
        for i, coeff in enumerate(equations):
            observation = 1 << i
            for bit in range(14, -1, -1):
                if not coeff & (1 << bit):
                    continue
                if not self.basis[bit][0]:
                    self.basis[bit] = (coeff, observation)
                    break
                coeff ^= self.basis[bit][0]
                observation ^= self.basis[bit][1]
        free = [bit for bit, (coeff, _) in enumerate(self.basis) if not coeff]
        self.offsets = [0] if len(free) <= max_free else []
        for bit in free:
            null = self._solve(0, 1 << bit)
            self.offsets += [offset ^ null for offset in self.offsets]

    def _solve(self, rhs, key=0):
        for bit, (coeff, observation) in enumerate(self.basis):
            if coeff and ((observation & rhs).bit_count() ^ (coeff & key).bit_count()) & 1:
                key ^= 1 << bit
        return key

    def candidates(self, observed):
        rhs = sum(int(value) << i for i, value in enumerate(observed))
        key = self._solve(rhs)
        if any(((coeff & key).bit_count() & 1) != int(observed[i])
               for i, coeff in enumerate(self.equations)):
            return ()
        return (key ^ offset for offset in self.offsets)


_PITCH_POSITIONS = (0, 1, 2, 3, 37, 38, 39)
_VOICING_POSITIONS = (4, 5, 6, 7, 35)
_QUIET_POSITIONS = _PITCH_POSITIONS + _VOICING_POSITIONS
_QUIET_PATTERNS = tuple(tuple(int(bit) for bit in f'{pitch:07b}{16:05b}') for pitch in (119, 124))
_UNUSED_PITCH = (120, 121, 122, 123, 125, 126, 127)
DEFAULT_THRESHOLD = 20.0
DEFAULT_MARGIN = 6.0


class SilenceSolver(_StrictSilence):
    """Generate keys with GF(2), then rank observed speech/silence evidence.

    Scores are evidence units, NOT calibrated correctness probabilities.
    Use new_session() at each PTT boundary for incremental analysis.
    """

    def __init__(self, sequence=None):
        super().__init__(sequence)
        self.equations = linear_equations(self.sequence)
        self.pitch_equations = [[self.equations[phase*49+pos] for pos in _PITCH_POSITIONS]
                                for phase in range(4)]
        self._pitch_cache = {}
        self.quiet_systems = [_ReducedSystem([self.equations[phase*49+pos] for pos in _QUIET_POSITIONS])
                              for phase in range(4)]
        self.voiced_systems = []
        for phases in itertools.combinations(range(4), 3):
            indices = [phase*49+pos for phase in phases for pos in _VOICING_POSITIONS]
            system = _ReducedSystem([self.equations[index] for index in indices])
            self.voiced_systems.append((phases, indices, system))
        # Uniform independent 7-bit pitch values are only a reference null model.
        # Real AMBE frames are correlated, so these credits are not probabilities.
        self.pitch_credit = {}
        for count in (3, 4):
            combinations = sum((min(24, 119-low)+1)**count - min(24, 119-low)**count
                               for low in range(120))
            self.pitch_credit[count] = -math.log2(combinations / 128**count)

    def _pitch_mask(self, key):
        if key not in self._pitch_cache:
            self._pitch_cache[key] = np.array([
                sum(((coeff & key).bit_count() & 1) << (6-i) for i, coeff in enumerate(equations))
                for equations in self.pitch_equations])
        return self._pitch_cache[key]

    def new_session(self, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
        return LinearSession(self, threshold, margin)

    def assess(self, bursts, valid=None, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
        frames = np.asarray(bursts)
        if frames.ndim != 3 or frames.shape[1:] != (4, 49):
            raise ValueError('Expected frames with shape (bursts, 4, 49)')
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
            if valid.shape != frames.shape[:2]:
                raise ValueError('FEC mask does not match bursts')
        session = self.new_session(threshold, margin)
        for index, burst in enumerate(frames):
            session._add(burst, None if valid is None else valid[index])
        return session.snapshot()

    def recover(self, bursts, valid=None):
        """Compatibility convenience: the currently confirmed key, or None."""
        return self.assess(bursts, valid)['key']


class LinearSession:
    """One caller-delimited PTT. No RF I/O, PTT detection, AI, or audio.

    push() returns the current decision after each burst. It does not latch a
    past decision: later contradictions can retract it. The caller decides
    when to accept the key and whether to keep monitoring. reset() starts a PTT.
    """

    def __init__(self, solver, threshold=DEFAULT_THRESHOLD, margin=DEFAULT_MARGIN):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError('threshold must be finite and greater than zero')
        if not math.isfinite(margin) or margin < 0:
            raise ValueError('margin must be finite and nonnegative')
        self.solver, self.threshold, self.margin = solver, float(threshold), float(margin)
        self.reset()

    def reset(self):
        self.bursts = 0
        self._strict = defaultdict(set)
        self._quiet = defaultdict(lambda: defaultdict(set))
        self._voice = defaultdict(dict)
        self._observed_pitch = [set() for _ in range(4)]

    def push(self, burst, valid=None):
        self._add(burst, valid)
        return self.snapshot()

    def _add(self, burst, valid=None):
        burst = np.asarray(burst)
        if burst.shape != (4, 49) or not np.all((burst == 0) | (burst == 1)):
            raise ValueError('Expected a binary (4, 49) burst')
        valid = np.ones(4, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        if valid.shape != (4,):
            raise ValueError('Expected a (4,) FEC mask')
        natural = np.empty((4, 49), dtype=np.uint8)
        natural[:, THUMBDV_MAP] = burst
        pitch = natural[:, _PITCH_POSITIONS] @ np.array([64, 32, 16, 8, 4, 2, 1])
        index = self.bursts
        for phase in range(4):
            if not valid[phase]:
                continue
            self._observed_pitch[phase].add(int(pitch[phase]))
            key = self.solver.candidate(burst[phase], phase)
            if key is not None:
                self._strict[key].add(phase)
            observed = natural[phase, list(_QUIET_POSITIONS)]
            for pattern in _QUIET_PATTERNS:
                for key in self.solver.quiet_systems[phase].candidates(np.bitwise_xor(observed, pattern)):
                    self._quiet[key][phase].add(index)
        voiced = defaultdict(dict)
        for phases, indices, system in self.solver.voiced_systems:
            if not valid[list(phases)].all():
                continue
            for key in system.candidates(natural.reshape(-1)[indices]):
                plain_pitch = (pitch ^ self.solver._pitch_mask(key))[list(phases)]
                if plain_pitch.max() > 119 or np.ptp(plain_pitch) > 24:
                    continue
                voiced[key].update(zip(phases, map(int, plain_pitch)))
        for key, phases in voiced.items():
            self._voice[key][index] = phases
        self.bursts += 1

    def snapshot(self):
        candidates = []
        keys = set(self._strict) | {key for key, phases in self._quiet.items() if len(phases) >= 2} | set(self._voice)
        for key in keys:
            # Each measured strict profile permits 5*3*2*3*4 field combinations.
            strict = max(0, len(self._strict.get(key, ())) * (49-math.log2(5*3*2*3*4)) - 15)
            phases = self._quiet.get(key, {})
            witnesses = set().union(*phases.values())
            reliable = sum(len(indices) >= 2 for indices in phases.values())
            # Four simultaneous quiet phases can decide immediately. Scattered
            # observations require repetition so unrelated coincidences do not add up.
            if len(phases) == 4 and set.intersection(*phases.values()):
                reliable = 4
            quiet = max(0, 11*reliable-15)
            if reliable >= 2 and witnesses:
                quiet += 2*math.log2(len(witnesses))
            voiced = self._voice.get(key, {})
            coverage = set().union(*(set(pitches) for pitches in voiced.values()))
            unique = {tuple(sorted(pitches.items())) for pitches in voiced.values()}
            voice = 0.0
            if len(coverage) == 4 and len(unique) >= 2:
                voice = (max(0, 5*len(coverage)-15)
                         + sum(self.solver.pitch_credit[len(pitches)] for pitches in unique)
                         + 2*math.log2(len(voiced)))
            mask = self.solver._pitch_mask(key)
            contradictions = sum((observed ^ int(mask[phase])) in _UNUSED_PITCH
                                 for phase, values in enumerate(self._observed_pitch) for observed in values)
            # These families share fields, so do not sum their overlapping support.
            score = max(strict, quiet, voice) - 4*contradictions
            candidates.append(dict(key=key, score=float(score), strict_score=float(strict),
                                   quiet_score=float(quiet), voiced_score=float(voice),
                                   contradictory_pitch_values=contradictions))
        candidates.sort(key=lambda candidate: (-candidate['score'], candidate['key']))
        top = candidates[0] if candidates else None
        runner_score = candidates[1]['score'] if len(candidates) > 1 else 0.0
        gap = top['score'] - max(0, runner_score) if top else 0.0
        key = (top['key'] if top and top['score'] >= self.threshold and gap >= self.margin
               and runner_score < self.threshold else None)
        if key is not None:
            # Basis-derived candidates must also match the actual sequence plugin.
            expected = np.array([(coeff & key).bit_count() & 1 for coeff in self.solver.equations])
            if not np.array_equal(self.solver.sequence.bits(key), expected):
                key = None
        display_key = self.solver.public_sequence.key_from_linear
        display_candidates = [dict(item, key=display_key(item['key'])) for item in candidates[:5]]
        return dict(key=display_key(key), candidate=display_key(top['key']) if top else None,
                    score=top['score'] if top else 0.0, margin=gap,
                    threshold=self.threshold, required_margin=self.margin,
                    bursts=self.bursts, traffic_seconds=self.bursts*.08,
                    candidates=display_candidates)
