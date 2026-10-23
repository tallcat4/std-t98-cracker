"""Explicit experimental b0=124, b1=16 profile observed in plaintext radio IQ.

Three TCH phases supply 36 constraints for 15 unknown key bits. Require
three separate bursts supporting the same key; no known-key input or AI.
This is a profile assumption, not a universal definition of silence.
"""
from itertools import combinations
import numpy as np
from .linear import linear_equations
from .sequence import Sequence, THUMBDV_MAP

POSITIONS = (0, 1, 2, 3, 37, 38, 39, 4, 5, 6, 7, 35)
PLAIN = tuple(int(bit) for bit in f"{124:07b}{16:05b}")


class UnvoicedSolver:
    def __init__(self, sequence=None):
        self.sequence = sequence or Sequence()
        equations = linear_equations(self.sequence)
        self.solvers = []
        for phases in combinations(range(4), 3):
            indices = [phase * 49 + pos for phase in phases for pos in POSITIONS]
            eq = [equations[i] for i in indices]
            basis = [(0, 0)] * 15
            for i, coeff in enumerate(eq):
                obs = 1 << i
                for bit in range(14, -1, -1):
                    if not coeff & (1 << bit):
                        continue
                    if not basis[bit][0]:
                        basis[bit] = (coeff, obs)
                        break
                    coeff ^= basis[bit][0]
                    obs ^= basis[bit][1]
            if all(coeff for coeff, _ in basis):
                self.solvers.append((phases, indices, eq, basis))
        if not self.solvers:
            raise ValueError("unvoiced-124 constraints do not determine all 15 key bits")

    def evidence(self, bursts, valid=None):
        frames = np.asarray(bursts)
        if frames.ndim != 3 or frames.shape[1:] != (4, 49) or not np.all((frames == 0) | (frames == 1)):
            raise ValueError("Expected binary frames with shape (bursts, 4, 49)")
        if valid is None:
            valid = np.ones(frames.shape[:2], dtype=bool)
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != frames.shape[:2]:
            raise ValueError("FEC mask does not match bursts")
        natural = np.empty(frames.shape, dtype=np.uint8)
        natural[:, :, THUMBDV_MAP] = frames
        evidence = {}
        expected = np.array(PLAIN * 3, dtype=np.uint8)
        for b, burst in enumerate(natural):
            for phases, indices, eq, basis in self.solvers:
                if not valid[b, list(phases)].all():
                    continue
                observed = burst.reshape(-1)[indices] ^ expected
                rhs = sum(int(value) << i for i, value in enumerate(observed))
                key = 0
                for bit, (coeff, obs) in enumerate(basis):
                    if ((obs & rhs).bit_count() ^ (coeff & key).bit_count()) & 1:
                        key |= 1 << bit
                # All 21 dependent equations must also hold.
                if any(((coeff & key).bit_count() & 1) != int(observed[i]) for i, coeff in enumerate(eq)):
                    continue
                if not np.array_equal(self.sequence.bits(key)[indices], observed):
                    continue
                evidence.setdefault(key, set()).add(b)
        return evidence

    def recover(self, bursts, valid=None):
        supported = [key for key, indices in self.evidence(bursts, valid).items() if len(indices) >= 3]
        return supported[0] if len(supported) == 1 else None
