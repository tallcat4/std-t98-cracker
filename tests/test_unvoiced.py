import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from std_t98_cracker.burst import pack49, write_bursts
from std_t98_cracker.cli import main
from std_t98_cracker.sequence import Sequence, THUMBDV_MAP
from std_t98_cracker.linear import SilenceSolver


def fixture(key, count=4, phases=(0, 1, 2, 3), seed=124, pitch=124):
    # Vary every unspecified bit; only two measured fields are constrained.
    d = np.random.default_rng(seed).integers(0, 2, (count, 4, 49), dtype=np.uint8)
    for phase in phases:
        d[:, phase, [0, 1, 2, 3, 37, 38, 39]] = [int(x) for x in f'{pitch:07b}']
        d[:, phase, [4, 5, 6, 7, 35]] = [1, 0, 0, 0, 0]
    return d[:, :, THUMBDV_MAP] ^ Sequence().thumbdv(key)[None]


class UnvoicedTests(unittest.TestCase):
    def test_scattered_quiet_frames_and_varying_spectral_fields(self):
        for key in (0, 777, 22222, 32767):
            frames = np.concatenate([fixture(key, count=2, pitch=pitch, seed=phase)
                                     for phase, pitch in enumerate((119, 124, 119))])
            good = np.zeros((6, 4), dtype=bool)
            for phase in range(3):
                good[phase*2:phase*2+2, phase] = True
            self.assertEqual(SilenceSolver().recover(frames, good), key)

    def test_keys_and_every_three_phase_combination(self):
        solver = SilenceSolver()
        for key in (0, 1, 777, 22222, 32767):
            for omitted in range(4):
                phases = tuple(p for p in range(4) if p != omitted)
                self.assertEqual(solver.recover(fixture(key, phases=phases)), key)

    def test_independent_evidence_conflicts_and_bad_fec(self):
        solver = SilenceSolver()
        a, b = fixture(777), fixture(22222, seed=5)
        # Four simultaneous quiet phases permit an immediate 80 ms decision.
        self.assertEqual(solver.recover(a[:1]), 777)
        self.assertIsNone(solver.assess(a[:1], threshold=30)['key'])
        self.assertEqual(solver.recover(a[:2]), 777)
        self.assertEqual(solver.recover(a[:3]), 777)
        self.assertIsNone(solver.recover(np.concatenate([a, b])))
        self.assertIsNone(solver.recover(np.concatenate([fixture(0), b])))
        good = np.ones((4, 4), dtype=bool)
        good[:, :2] = False
        self.assertIsNone(solver.recover(a, good))
        rng = np.random.default_rng(42)
        self.assertIsNone(solver.recover(rng.integers(0, 2, (1000, 4, 49), dtype=np.uint8)))
        with self.assertRaises(ValueError):
            solver.recover(a, np.ones((1, 4)))

    def test_standard_linear_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'synthetic.burst'
            write_bursts(source, pack49(fixture(22222)))
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['crack', str(source), '--method', 'linear']), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result['keys'], [22222])
                self.assertEqual(result['windows'][0]['method'], 'linear')

    def test_cross_constraint_agreement_and_conflicts(self):
        from test_core import MEASURED
        old_plain = np.unpackbits(np.array([list(x) for x in MEASURED[:4]], dtype=np.uint8), axis=-1)[:, :49]
        def old(key):
            return (old_plain ^ Sequence().thumbdv(key))[None]
        solver = SilenceSolver()
        self.assertEqual(solver.recover(np.concatenate([old(777), fixture(777)])), 777)
        self.assertIsNone(solver.recover(np.concatenate([old(777), fixture(22222)])))
        self.assertIsNone(solver.recover(np.concatenate([old(0), fixture(777)])))
        # A conflict within one family must not be hidden by the other family.
        self.assertIsNone(solver.recover(np.concatenate([old(777), old(22222), fixture(777)])))
        self.assertIsNone(solver.recover(np.concatenate([old(777), fixture(777), fixture(22222)])))


if __name__ == '__main__':
    unittest.main()
