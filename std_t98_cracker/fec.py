"""Thin ctypes adapter to SPEAR's blip25c hard/soft FEC. No voice synthesis."""
import ctypes as C
import os
from pathlib import Path
import sys
import numpy as np
from .sequence import THUMBDV_MAP


class FecDecoder:
    def __init__(self, path=None):
        package = Path(__file__).resolve().parent
        name = "t98_fec.dll" if sys.platform == "win32" else "t98_fec.dylib" if sys.platform == "darwin" else "t98_fec.so"
        override = path or os.environ.get("STD_T98_FEC")
        candidates = [Path(override)] if override else [package/name, package.parent/"build"/name, package.parent/"build"/"Release"/name]
        selected = next((p for p in candidates if p.is_file()), None)
        if selected is None:
            raise RuntimeError("Native FEC is not built. Run build.ps1 (Windows), or cmake -S . -B build && cmake --build build --config Release.")
        self.lib = C.CDLL(str(selected.resolve()))
        u8 = C.POINTER(C.c_uint8)
        self.lib.blip25_fec_3600.argtypes = [u8, u8, u8]
        self.lib.blip25_fec_3600.restype = C.c_int
        self.lib.blip25_fec_3600_soft.argtypes = [C.POINTER(C.c_int8), u8, u8]
        self.lib.blip25_fec_3600_soft.restype = C.c_int

    def fec(self, payload, llr=None):
        if payload.dtype != np.uint8 or payload.ndim != 3 or payload.shape[1:] != (4, 9):
            raise ValueError("FEC expects (bursts, 4, 9) payload bytes")
        if llr is not None and (llr.dtype != np.int8 or llr.shape != (len(payload), 4, 72)):
            raise ValueError("LLR expects signed int8 (bursts, 4, 72)")
        output = np.empty((len(payload), 4, 49), dtype=np.uint8)
        valid = np.ones((len(payload), 4), dtype=bool)
        corrected = np.zeros((len(payload), 4), dtype=np.int16)
        d = (C.c_uint8 * 49)()
        errors = (C.c_uint8 * 4)()
        for b, burst in enumerate(payload):
            for phase, frame in enumerate(burst):
                if llr is None:
                    source = (C.c_uint8 * 9).from_buffer_copy(frame.tobytes())
                    e = self.lib.blip25_fec_3600(source, d, errors)
                else:
                    source = (C.c_int8 * 72).from_buffer_copy(llr[b, phase].tobytes())
                    e = self.lib.blip25_fec_3600_soft(source, d, errors)
                output[b, phase] = np.ctypeslib.as_array(d)[THUMBDV_MAP]
                valid[b, phase] = e >= 0
                corrected[b, phase] = e
        return output, valid, corrected
