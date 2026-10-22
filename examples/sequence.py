"""Example plugin; identical to STD-T98. Copy and edit generate() to experiment.

python -m std_t98_cracker crack capture.burst --sequence examples/sequence.py
Keys are 0..32767. Return 196 bits in natural AMBE d order; the caller handles
ThumbDV reordering. The PN restarts at each four-frame traffic burst.
"""
def generate(key):
    bits = []
    for _ in range(196):
        bits.append(key & 1)
        key = (key >> 1) | (((key ^ (key >> 1)) & 1) << 14)
    return bits
