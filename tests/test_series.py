"""Public series selection, display-code mapping and linear state recovery."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from std_t98_cracker.burst import pack49, read_bursts, write_bursts
from std_t98_cracker.cli import main
from std_t98_cracker.enhanced_sequence import CandidateSequence, OFFSETS
from std_t98_cracker.linear import SilenceSolver
from std_t98_cracker.sequence import Sequence, load_sequence
from test_unvoiced import fixture


class PublicSeries(unittest.TestCase):
    def test_linear_recovers_display_code_at_offset_boundaries(self):
        plain = fixture(0)
        for series in range(1, 16):
            sequence = load_sequence(series=series)
            solver = SilenceSolver(sequence)
            for key in (0, 1, 12345, 32767 - OFFSETS[series], 32768 - OFFSETS[series], 32767):
                with self.subTest(series=series, key=key):
                    encrypted = plain ^ CandidateSequence(series).batch(np.array([key]))[0]
                    result = solver.assess(encrypted)
                    self.assertEqual(result['key'], key)
                    self.assertEqual(result['candidate'], key)
                    self.assertEqual(result['candidates'][0]['key'], key)
                    np.testing.assert_array_equal(encrypted ^ sequence.thumbdv(key), plain)
            session = solver.new_session()
            self.assertEqual(session.push(plain[0] ^ sequence.thumbdv(12345))['key'], 12345)
            session.reset()
            self.assertIsNone(session.snapshot()['key'])
            self.assertEqual(session.push(plain[0])['key'], 0)

    def test_cli_trace_and_explicit_descramble(self):
        plain = fixture(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for series in (0, 1, 10, 15):
                key = 32767
                source, trace, report = (root / f'{series}{suffix}' for suffix in ('.burst', '.jsonl', '.json'))
                encrypted = plain ^ CandidateSequence(series).batch(np.array([key]))[0]
                write_bursts(source, pack49(encrypted))
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['crack', str(source), '--series', str(series), '--trace', str(trace), '--json', str(report)]), 0)
                    self.assertEqual(main(['descramble', str(source), '--series', str(series), '--key', str(key), '-o', str(root / f'{series}-clear')]), 0)
                result = json.loads(report.read_text())
                self.assertEqual(result['keys'], [key])
                self.assertEqual(result['windows'][0]['first_confirmation']['key'], key)
                self.assertTrue(all(json.loads(line)['key'] == key for line in trace.read_text().splitlines()))
                np.testing.assert_array_equal(read_bursts(root / f'{series}-clear.burst')[1], pack49(plain))

    def test_default_and_option_validation(self):
        np.testing.assert_array_equal(load_sequence().bits(12345), Sequence().bits(12345))
        for series in (-1, 16):
            with self.assertRaises(ValueError):
                load_sequence(series=series)
        with self.assertRaises(ValueError):
            load_sequence('plugin.py', series=1)
        for args in (['--series', '16'], ['--series', '-1'], ['--series', '0', '--sequence', 'plugin.py']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(['crack', 'missing.burst', *args])


if __name__ == '__main__':
    unittest.main()
