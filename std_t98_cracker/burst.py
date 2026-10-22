"""Strict legacy .burst I/O: FF + four (48 + 9-byte) records.

The 49-bit decrypted variant uses 31 + 7 bytes. Markers are lengths in bits,
not ASCII. A file must contain one format throughout. Never scan for FF inside
damaged data: FF is also valid payload and guessing would lose cipher phase.
"""
from pathlib import Path
import numpy as np


def read_bursts(path):
    data = Path(path).read_bytes()
    if not data:
        raise ValueError("Burst file is empty; no traffic was captured")
    if len(data) < 2 or data[0] != 0xFF or data[1] not in (0x48, 0x31):
        raise ValueError("Expected .burst: FF followed by 48 (72 bits) or 31 (49 bits)")
    bits = data[1]
    width = (bits + 7) // 8
    size = 1 + 4 * (1 + width)
    if len(data) % size:
        raise ValueError(f"Truncated .burst: {len(data)} bytes is not a multiple of {size}")
    records = np.frombuffer(data, dtype=np.uint8).reshape(-1, size)
    if np.any(records[:, 0] != 0xFF):
        raise ValueError("Invalid burst boundary marker")
    frames = records[:, 1:].reshape(-1, 4, width + 1)
    if np.any(frames[:, :, 0] != bits):
        raise ValueError("Invalid or mixed frame length markers")
    payload = frames[:, :, 1:].copy()
    if bits == 49 and np.any(payload[:, :, -1] & 0x7F):
        raise ValueError("49-bit payload has nonzero padding")
    return bits, payload


def write_bursts(path, payload, bits=49):
    if bits not in (49, 72):
        raise ValueError("Only 49-bit and 72-bit frames are supported")
    payload = np.asarray(payload, dtype=np.uint8)
    width = (bits + 7) // 8
    if payload.ndim != 3 or payload.shape[1:] != (4, width):
        raise ValueError("Payload must have shape (bursts, 4, bytes_per_frame)")
    frames = np.empty((len(payload), 4, width + 1), dtype=np.uint8)
    frames[:, :, 0] = bits
    frames[:, :, 1:] = payload
    records = np.empty((len(payload), 1 + 4 * (width + 1)), dtype=np.uint8)
    records[:, 0] = 0xFF
    records[:, 1:] = frames.reshape(len(payload), -1)
    if bits == 49 and np.any(payload[:, :, -1] & 0x7F):
        raise ValueError("49-bit payload has nonzero padding")
    with Path(path).open("xb") as stream:
        stream.write(records.tobytes())


def unpack49(payload):
    return np.unpackbits(payload, axis=-1)[..., :49].copy()


def pack49(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8), axis=-1)


def read_llr(path, count):
    raw = Path(path).read_bytes()
    if len(raw) != count * 288:
        raise ValueError("LLR size must be exactly 288 bytes per .burst record")
    return np.frombuffer(raw, dtype=np.int8).reshape(count, 4, 72).copy()
