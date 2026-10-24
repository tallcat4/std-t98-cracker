"""Blind GF(2) solver: independent parity oracle and streaming behavior."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from std_t98_cracker.burst import pack49, write_bursts
from std_t98_cracker.cli import main
from std_t98_cracker.enhanced_sequence import CandidateSequence
from std_t98_cracker.series_solver import SeriesSolver, _fwht


def continuous_frames(count=12, seed=42):
    rng = np.random.default_rng(seed)
    flips = rng.random((count * 4, 49)) < .08
    return np.logical_xor.accumulate(flips, axis=0).astype(np.uint8).reshape(count, 4, 49)


class BlindSeriesTests(unittest.TestCase):
    def test_transform_matches_direct_parity_sum(self):
        rng = np.random.default_rng(9)
        weights = rng.integers(-20, 21, (3, 32)).astype(float)
        direct = np.array([[sum(row[c] * (-1 if (c & state).bit_count() % 2 else 1)
                                for c in range(32)) for state in range(32)] for row in weights])
        np.testing.assert_array_equal(_fwht(weights), direct)

    def test_all_series_display_keys_and_plaintext(self):
        solver = SeriesSolver()
        plain = continuous_frames()
        for series in range(16):
            for key in (1, 12345, 32767):
                with self.subTest(series=series, key=key):
                    encrypted = plain ^ CandidateSequence(series).batch(np.array([key]))[0]
                    result = solver.assess(encrypted)
                    self.assertEqual((result['series'], result['key']), (series, key))
        result = solver.assess(plain)
        self.assertEqual((result['series'], result['key']), (0, 0))
        self.assertEqual(sum(c['key'] == 0 for c in result['candidates']), 1)

    def test_explicit_series_uses_same_temporal_constraints(self):
        plain = continuous_frames()
        for series in range(16):
            solver = SeriesSolver(series=series)
            for key in (0, 29439):
                encrypted = plain ^ CandidateSequence(series).batch(np.array([key]))[0]
                result = solver.assess(encrypted)
                self.assertEqual((result['series'], result['key']), (series, key))
                self.assertTrue(all(c['series'] == series for c in result['candidates']))
        for invalid in (-1, 16, 'auto'):
            with self.assertRaises(ValueError):
                SeriesSolver(series=invalid)

    def test_stream_reset_invalid_gaps_and_no_latched_decision(self):
        solver = SeriesSolver()
        plain = continuous_frames()
        encrypted = plain ^ CandidateSequence(10).batch(np.array([12345]))[0]
        good = np.ones((len(plain), 4), dtype=bool)
        good[3, 2] = False
        session = solver.new_session()
        for burst, valid in zip(encrypted, good):
            result = session.push(burst, valid)
        self.assertEqual(result, solver.assess(encrypted, good))
        self.assertEqual(result['transition_pairs'], len(plain) * 4 - 3)
        self.assertEqual((result['series'], result['key']), (10, 12345))
        session.reset()
        self.assertIsNone(session.snapshot()['candidate'])
        self.assertEqual(session.snapshot()['bursts'], 0)
        # Isolated valid frames must not acquire artificial neighbors.
        isolated = np.tile([True, False, True, False], (len(plain), 1))
        result = solver.assess(encrypted, isolated)
        self.assertEqual(result['transition_pairs'], 0)
        self.assertIsNone(result['key'])
        self.assertIsNone(solver.assess(encrypted, threshold=1e9)['key'])
        self.assertIsNone(solver.assess(encrypted, margin=1e9)['key'])
        # A decision can be withdrawn by contradictory observations.
        session = solver.new_session()
        for burst in encrypted:
            session.push(burst)
        rng = np.random.default_rng(4)
        for burst in rng.integers(0, 2, (500, 4, 49), dtype=np.uint8):
            session._add(burst)
        self.assertIsNone(session.snapshot()['key'])

    def test_random_and_unsupported_polynomial(self):
        solver = SeriesSolver()
        rng = np.random.default_rng(2026)
        self.assertIsNone(solver.assess(rng.integers(0, 2, (50, 4, 49), dtype=np.uint8))['key'])
        unknown = continuous_frames() ^ CandidateSequence(16).batch(np.array([23157]))[0]
        self.assertIsNone(solver.assess(unknown)['key'])
        for threshold, margin in ((0, 20), (40, -1), (float('nan'), 20), (40, float('inf'))):
            with self.assertRaises(ValueError):
                solver.new_session(threshold, margin)
        session = solver.new_session()
        for burst, valid in ((np.zeros((3, 49)), None), (np.full((4, 49), 2), None),
                             (np.zeros((4, 49)), [True])):
            with self.assertRaises(ValueError):
                session.push(burst, valid)

    def test_cli_auto_and_explicit_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, trace, report = root/'anonymous.burst', root/'trace.jsonl', root/'result.json'
            encrypted = continuous_frames() ^ CandidateSequence(15).batch(np.array([32767]))[0]
            write_bursts(path, pack49(encrypted))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['crack', str(path), '--series', 'auto', '--trace', str(trace), '--json', str(report)]), 0)
                self.assertEqual(main(['crack', str(path), '--series', 'auto', '--method', 'ai']), 1)
                with self.assertRaises(SystemExit):
                    main(['descramble', str(path), '--series', 'auto', '--key', '1', '-o', str(root/'out')])
            result = json.loads(report.read_text())
            self.assertEqual(result['solutions'], [{'series': 15, 'key': 32767}])
            self.assertEqual(result['windows'][0]['threshold'], 40)
            self.assertEqual(result['windows'][0]['required_margin'], 20)
            self.assertEqual(result['windows'][0]['first_confirmation']['series'], 15)
            self.assertEqual(len(trace.read_text().splitlines()), len(encrypted))


if __name__ == '__main__':
    unittest.main()
