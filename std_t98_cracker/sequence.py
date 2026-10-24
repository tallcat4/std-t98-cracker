"""A sequence is one Python function: key (0..32767) -> 196 binary values."""
import importlib.util
from pathlib import Path
import numpy as np

THUMBDV_MAP = np.array([
    0,18,36,1,19,37,2,20,38,3,21,39,4,22,40,5,23,41,6,24,42,
    7,25,43,8,26,44,9,27,45,10,28,46,11,29,47,12,30,48,13,31,
    14,32,15,33,16,34,17,35,
])


def generate_pn_sequence_196(key):
    state = int(key)
    result = []
    for _ in range(196):
        result.append(state & 1)
        feedback = (state ^ (state >> 1)) & 1
        state = (state >> 1) | (feedback << 14)
    return result


class Sequence:
    def __init__(self, generate=generate_pn_sequence_196, name="std-t98"):
        self.generate = generate
        self.name = name

    def bits(self, key):
        if not 0 <= key <= 32767:
            raise ValueError("Key must be 0..32767")
        result = np.asarray(self.generate(key))
        if result.shape != (196,) or not np.all((result == 0) | (result == 1)):
            raise ValueError(f"Sequence {self.name} must return exactly 196 binary values")
        return result.astype(np.uint8)

    def thumbdv(self, key):
        return self.bits(key).reshape(4, 49)[:, THUMBDV_MAP]

    def linearized(self):
        """Generator indexed by the GF(2) unknown; normally the key itself."""
        return self

    def key_from_linear(self, value):
        """Convert a solved GF(2) unknown to the externally visible key."""
        return value


def load_sequence(path=None, series=0):
    if not 0 <= series <= 15:
        raise ValueError("--series must be 0..15 (0 = standard privacy)")
    if path is not None and series != 0:
        raise ValueError("Choose either --series or --sequence")
    if series:
        from .enhanced_sequence import EnhancedSequence
        return EnhancedSequence(series)
    if path is None:
        return Sequence()
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("t98_user_sequence", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load sequence: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "generate", None)):
        raise ValueError("Sequence file must define generate(key)")
    result = Sequence(module.generate, path.stem)
    result.bits(0)
    result.bits(32767)
    return result
