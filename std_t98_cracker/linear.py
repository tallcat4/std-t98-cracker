"""SPEAR's measured AMBE-3000R silence profile and GF(2) key recovery.

Ported from apps/std_t98/secret/silence.cpp. A candidate must satisfy all
profile fields, and a confirmed key needs evidence at two distinct phases.
Unsupported silence patterns and conflicting keys produce no result.
"""
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


class SilenceSolver:
    def __init__(self, sequence=None):
        self.sequence = sequence or Sequence()
        basis_pn = np.array([self.sequence.bits(1 << i) for i in range(15)])
        if np.any(self.sequence.bits(0)):
            raise ValueError("Linear solver requires a linear sequence with generate(0) == 0")
        # Detect accidental nonlinear generators; each candidate is also checked
        # against the actual generator before it can be accepted.
        for key in (3, 7, 12345, 32767):
            predicted = np.bitwise_xor.reduce(basis_pn[[i for i in range(15) if key & (1 << i)]], axis=0)
            if not np.array_equal(predicted, self.sequence.bits(key)):
                raise ValueError("Sequence is not linear over GF(2); use AI search or an explicit key")
        equations = [sum(int(basis_pn[b, t]) << b for b in range(15)) for t in range(196)]
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

    def recover(self, bursts, valid=None):
        evidence = {}
        for b, burst in enumerate(bursts):
            for phase, frame in enumerate(burst):
                if valid is not None and not valid[b, phase]:
                    continue
                key = self.candidate(frame, phase)
                if key is not None:
                    evidence[key] = evidence.get(key, 0) | (1 << phase)
        supported = [key for key, phases in evidence.items() if phases.bit_count() >= 2]
        return supported[0] if len(supported) == 1 else None
