import argparse
import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from std_t98_cracker.enhanced_sequence import CandidateSequence, MASKS, OFFSETS, arbitrary_pn, coefficient_table
from std_t98_cracker.sequence import THUMBDV_MAP, generate_pn_sequence_196, load_sequence
from std_t98_cracker.enhanced import compare, main, select_blocks, setting, write_json


class SequenceTests(unittest.TestCase):
    def test_arbitrary_mask_coefficients(self):
        rng = np.random.default_rng(98)
        masks = np.r_[0, 1, 3, 32767, rng.integers(0, 32768, 40)]
        coefficients = coefficient_table(masks)
        for mask, coeff in zip(masks, coefficients):
            for key in (0, 1, 12345, 32767):
                bits = np.array([(int(c) & key).bit_count() & 1 for c in coeff], dtype=np.uint8)
                scalar = arbitrary_pn(int(mask), key).reshape(4, 49)[:, THUMBDV_MAP].ravel()
                np.testing.assert_array_equal(bits, scalar)
        for family, mask in enumerate(MASKS):
            seq = CandidateSequence(family)
            np.testing.assert_array_equal(arbitrary_pn(mask & 32767, seq.initial_state(12345)), seq.generate(12345))

    def test_standard_equivalence(self):
        for key in (0, 1, 2, 12345, 32767):
            np.testing.assert_array_equal(CandidateSequence(0).generate(key), generate_pn_sequence_196(key))

    def test_full_period_and_offset_inverse(self):
        for family, mask in enumerate(MASKS):
            seq = CandidateSequence(family)
            state, seen = 1, set()
            while state not in seen:
                seen.add(state)
                state = (state >> 1) | (((state & (mask & 0x7fff)).bit_count() % 2) << 14)
            self.assertEqual(state, 1)
            self.assertEqual(len(seen), 32767)
            for key in range(32768):
                self.assertEqual(seq.key_for_state(seq.initial_state(key)), key)

    def test_scalar_batch_and_boundary(self):
        for family in range(17):
            keys = np.array([0, 1, 32767-OFFSETS[family], 32767])
            for reset in ('burst', 'frame'):
                for initialization in ('offset', 'state'):
                    seq = CandidateSequence(family, initialization, 19, reset)
                    expected = np.stack([seq.generate(int(k)).reshape(4, 49)[:, THUMBDV_MAP] for k in keys])
                    np.testing.assert_array_equal(seq.batch(keys), expected)
                    self.assertFalse(seq.batch(np.array([0])).any())
        self.assertEqual(CandidateSequence(1).initial_state(32767-OFFSETS[1]), 32767)
        self.assertEqual(CandidateSequence(1).initial_state(32768-OFFSETS[1]), 1)

    def test_input_validation(self):
        for args in ((17,), (-1,), (1, 'unknown'), (1, 'state', -1), (1, 'offset', 32767), (1, 'offset', 0, 'bad')):
            with self.assertRaises(ValueError):
                CandidateSequence(*args)
        for keys in ([-1], [32768], [1.5], [[1]]):
            with self.assertRaises(ValueError):
                CandidateSequence(1).batch(keys)


class ProbeTests(unittest.TestCase):
    def test_no_joining_across_invalid_bursts(self):
        frames = np.zeros((21, 4, 49), np.uint8)
        valid = np.ones((21, 4), bool)
        valid[4, 1] = False
        _, starts = select_blocks(frames, valid, 3)
        self.assertEqual(starts, [5, 10, 15])
        for starts in ([0, 5, 10], [5, 6, 10], [-1, 5, 10], [5, 10, 20]):
            with self.assertRaises(ValueError):
                select_blocks(frames, valid, 3, starts)
        with self.assertRaises(ValueError):
            select_blocks(frames[:9], valid[:9], 3)

    def test_exhaustive_scoring_and_ranking(self):
        plain = np.random.default_rng(7).integers(0, 2, (3, 5, 4, 49), dtype=np.uint8)
        encrypted = plain ^ CandidateSequence(7).batch(np.array([12345]))[0]
        class ExactScorer:
            count = 0
            def score(self, blocks, pn):
                self.count += len(pn)
                return np.stack([1 - (blocks ^ candidate ^ plain).sum(axis=(1, 2, 3)).astype(float) for candidate in pn])
        scorer = ExactScorer()
        result = compare(encrypted, scorer, range(17), np.array([1, 12345, 32767]))
        self.assertEqual(scorer.count, 52)  # Every candidate plus the clear control.
        best = result['top_candidates'][0]
        self.assertEqual((best['family'], best['key']), (7, 12345))
        self.assertTrue(best['all_blocks_plain'])
        self.assertFalse(result['confirmed'])

    def test_plugin_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'candidate.py'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['plugin', '--family', '5', '-o', str(path)])
            np.testing.assert_array_equal(load_sequence(path).bits(12345), CandidateSequence(5).generate(12345))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(['plugin', '--family', '6', '-o', str(path)])
            arbitrary = Path(temp)/'arbitrary.py'
            with contextlib.redirect_stdout(io.StringIO()):
                main(['plugin', '--mask', '0x1235', '--offset', '456', '-o', str(arbitrary)])
            np.testing.assert_array_equal(load_sequence(arbitrary).bits(32767), arbitrary_pn(0x1235, 456))
            report = Path(temp)/'report.json'
            write_json(report, {'a': 1})
            with self.assertRaises(FileExistsError):
                write_json(report, {'a': 2})

    def test_coverage_deduplicates_and_does_not_confirm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = dict(schema='enhanced-probe-v1', radio_series=5, radio_key=1,
                          input={'sha256': 'one-capture'}, model={'model_sha256': 'model'},
                          parameters=dict(initialization='offset', skip=0, reset='burst', all_keys=False),
                          top_candidates=[dict(family=7, key=1, all_blocks_plain=True, minimum_margin=10)],
                          plaintext_control=dict(all_blocks_plain=False))
            source, output = root/'probe.json', root/'coverage.json'
            write_json(source, report)
            with contextlib.redirect_stdout(io.StringIO()):
                main(['summary', str(source), str(source), '--json', str(output)])
            coverage = json.loads(output.read_text(encoding='utf-8'))['coverage']
            self.assertEqual(len(coverage), 15)
            self.assertEqual(coverage[4]['distinct_captures'], 1)
            self.assertEqual(len(coverage[4]['observations']), 1)
            self.assertFalse(coverage[4]['confirmed'])
            self.assertTrue(coverage[4]['observations'][0]['key_mapping_matches'])
            self.assertEqual(coverage[0]['keys'], [])

    def test_sweep_resume_and_input_mismatch(self):
        from std_t98_cracker.enhanced_sweep import run_sweep
        calls = []
        class FakeHeavy:
            def describe(self):
                return {'model': 'test'}
        class FakeSweep:
            light = None
            heavy = FakeHeavy()
            def __init__(self, *args):
                pass
            def score_mask(self, coefficients, top, cutoff):
                calls.append(1)
                return [dict(initial_state=10, margins=[5, 5, 5], minimum_margin=5,
                             mean_margin=5, all_blocks_plain=True)], 32767
        with tempfile.TemporaryDirectory() as temp, patch('std_t98_cracker.enhanced_sweep.MaskSweep', FakeSweep), contextlib.redirect_stdout(io.StringIO()):
            blocks = np.zeros((3, 5, 4, 49), dtype=np.uint8)
            report = run_sweep(blocks, temp, 5, mask_stop=4, shard_size=2)
            self.assertEqual(report['candidates'], 4*32767)
            self.assertEqual(report['top_candidates'][0]['offset'], 5)
            self.assertEqual(len(calls), 4)
            resumed = run_sweep(blocks, temp, 5, mask_stop=4, shard_size=2, resume=True)
            self.assertEqual(report, resumed)
            self.assertEqual(len(calls), 4)
            with self.assertRaises(ValueError):
                run_sweep(blocks, temp, 6, mask_stop=4, shard_size=2, resume=True)
            blocks[0, 0, 0, 0] = 1
            with self.assertRaises(ValueError):
                run_sweep(blocks, temp, 5, mask_stop=4, shard_size=2, resume=True)

    def test_setting(self):
        self.assertEqual(setting('5-00001'), (5, 1))
        for value in ('0-00001', '16-00001', '1-00000', '1-32768', 'bad'):
            with self.assertRaises(argparse.ArgumentTypeError):
                setting(value)


if __name__ == '__main__':
    unittest.main()
