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
from std_t98_cracker.unvoiced import UnvoicedSolver


def fixture(key, count=4, phases=(0, 1, 2, 3), seed=124):
    # Vary every unspecified bit; only two measured fields are constrained.
    d = np.random.default_rng(seed).integers(0, 2, (count, 4, 49), dtype=np.uint8)
    for phase in phases:
        d[:, phase, [0, 1, 2, 3, 37, 38, 39]] = [1, 1, 1, 1, 1, 0, 0]
        d[:, phase, [4, 5, 6, 7, 35]] = [1, 0, 0, 0, 0]
    return d[:, :, THUMBDV_MAP] ^ Sequence().thumbdv(key)[None]


class UnvoicedTests(unittest.TestCase):
    def test_keys_and_every_three_phase_combination(self):
        solver = UnvoicedSolver()
        for key in (0, 1, 777, 22222, 32767):
            for omitted in range(4):
                phases = tuple(p for p in range(4) if p != omitted)
                self.assertEqual(solver.recover(fixture(key, phases=phases)), key)

    def test_independent_evidence_conflicts_and_bad_fec(self):
        solver = UnvoicedSolver()
        a, b = fixture(777), fixture(22222, seed=5)
        # Four matching phase triples in one burst are still only one witness.
        self.assertIsNone(solver.recover(a[:1]))
        self.assertIsNone(solver.recover(a[:2]))
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

    def test_explicit_cli_profile_and_no_ai_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'synthetic.burst'
            write_bursts(source, pack49(fixture(22222)))
            with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['crack', str(source), '--method', 'linear', '--silence-profile', 'unvoiced-124']), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result['keys'], [22222])
                self.assertEqual(result['silence_profile'], 'unvoiced-124')
                self.assertEqual(result['windows'][0]['method'], 'linear')
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['crack', str(source), '--method', 'ai', '--silence-profile', 'unvoiced-124']), 1)


if __name__ == '__main__':
    unittest.main()
