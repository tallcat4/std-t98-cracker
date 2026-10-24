"""PN15 sequences: standard index 0, verified radio series 1..15, experimental 16."""
from dataclasses import dataclass

import numpy as np

from .sequence import Sequence, THUMBDV_MAP


MASKS = (0x8003, 0xFB8D, 0xDC6F, 0x8999, 0xF499, 0x9331, 0xAE1D, 0xCC85,
         0xF4F9, 0xE7AD, 0xD9A3, 0x944B, 0xC929, 0xDC71, 0xB62D, 0x8A7B, 0xC2B1)
OFFSETS = (0, 0x1B95, 0x741B, 0x3679, 0x4D57, 0x55E8, 0x3444, 0x3471,
           0x68A3, 0x7B2C, 0x76FD, 0x7B59, 0x0C53, 0x7769, 0x4597, 0x5748, 0x4AAF)


class EnhancedSequence(Sequence):
    """Radio display code -> PN; solve linear constraints in LFSR state space."""

    def __init__(self, series):
        if not isinstance(series, int) or not 1 <= series <= 15:
            raise ValueError('Enhanced series must be 1..15')
        self.candidate = CandidateSequence(series)
        super().__init__(self.candidate.generate, f'alinco-enhanced-{series}')

    def linearized(self):
        # Integer addition modulo 32767 is not linear over GF(2). Only the
        # LFSR state participates in elimination; translate results afterwards.
        state_sequence = CandidateSequence(self.candidate.family, initialization='state')
        return Sequence(state_sequence.generate, self.name + '-state')

    def key_from_linear(self, value):
        return None if value is None else self.candidate.key_for_state(value)


def arbitrary_pn(mask, state):
    """Scalar reference, natural order, old-LSB output and burst reset."""
    if not 0 <= mask <= 0x7fff or not 0 <= state <= 0x7fff:
        raise ValueError('mask and state must be 15-bit integers')
    out = np.empty(196, dtype=np.uint8)
    for i in range(196):
        out[i] = state & 1
        state = (state >> 1) | (((state & mask).bit_count() & 1) << 14)
    return out


def coefficient_table(masks):
    """PN bit coefficients over the initial state's 15 bits, in ThumbDV order."""
    masks = np.asarray(masks, dtype=np.int64)
    if masks.ndim != 1 or np.any(masks < 0) or np.any(masks > 32767):
        raise ValueError('masks must be a one-dimensional array in 0..32767')
    coeff = np.empty((196, len(masks)), dtype=np.uint16)
    coeff[:15] = (1 << np.arange(15, dtype=np.uint16))[:, None]
    for i in range(15, 196):
        coeff[i] = 0
        for bit in range(15):
            coeff[i] ^= np.where(masks & (1 << bit), coeff[i-15+bit], 0).astype(np.uint16)
    return coeff.T.reshape(-1, 4, 49)[:, :, THUMBDV_MAP].reshape(-1, 196)


@dataclass(frozen=True)
class CandidateSequence:
    family: int
    initialization: str = 'offset'
    skip: int = 0
    reset: str = 'burst'

    def __post_init__(self):
        if not 0 <= self.family < len(MASKS):
            raise ValueError('family must be 0..16')
        if self.initialization not in ('offset', 'state'):
            raise ValueError('initialization must be offset or state')
        if not 0 <= self.skip < 32767:
            raise ValueError('skip must be 0..32766')
        if self.reset not in ('burst', 'frame'):
            raise ValueError('reset must be burst or frame')

    def initial_state(self, key):
        if not 0 <= key <= 32767:
            raise ValueError('key must be 0..32767')
        if key == 0 or self.initialization == 'state':
            return key
        value = key + OFFSETS[self.family]
        return value - 32767 if value > 32767 else value

    def key_for_state(self, state):
        if not 0 <= state <= 32767:
            raise ValueError('state must be 0..32767')
        if state == 0 or self.initialization == 'state':
            return state
        return (state - OFFSETS[self.family] - 1) % 32767 + 1

    def generate(self, key):
        state = self.initial_state(key)
        mask = MASKS[self.family] & 0x7fff
        output = []
        count = 196 if self.reset == 'burst' else 49
        for index in range(self.skip + count):
            if index >= self.skip:
                output.append(state & 1)
            state = (state >> 1) | (((state & mask).bit_count() & 1) << 14)
        return np.array(output if self.reset == 'burst' else output*4, dtype=np.uint8)

    def batch(self, keys):
        """Vectorized generation, returned in (keys, 4, 49) ThumbDV order."""
        keys = np.asarray(keys)
        if keys.ndim != 1 or keys.dtype.kind not in 'iu' or np.any(keys < 0) or np.any(keys > 32767):
            raise ValueError('keys must be a one-dimensional integer array in 0..32767')
        states = keys.astype(np.uint32)
        if self.initialization == 'offset':
            states = np.where(states == 0, 0, states + OFFSETS[self.family])
            states = np.where(states > 32767, states - 32767, states)
        count = 196 if self.reset == 'burst' else 49
        output = np.empty((len(keys), count), dtype=np.uint8)
        for index in range(self.skip + count):
            if index >= self.skip:
                output[:, index-self.skip] = states & 1
            parity = states & (MASKS[self.family] & 0x7fff)
            for shift in (8, 4, 2, 1):
                parity ^= parity >> shift
            states = (states >> 1) | ((parity & 1) << 14)
        if self.reset == 'frame':
            output = np.tile(output, (1, 4))
        return output.reshape(-1, 4, 49)[:, :, THUMBDV_MAP]
