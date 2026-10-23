import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from std_t98_cracker.linear import SilenceSolver
from std_t98_cracker.burst import pack49, write_bursts
from std_t98_cracker.cli import main
from std_t98_cracker.sequence import Sequence, THUMBDV_MAP
from test_unvoiced import fixture as quiet_fixture
from test_voiced import voiced_fixture


class StreamTests(unittest.TestCase):
    def test_cli_trace_first_confirmation_and_later_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, trace, report = root/'input.burst', root/'trace.jsonl', root/'result.json'
            write_bursts(source, pack49(np.concatenate([quiet_fixture(777), quiet_fixture(22222)])))
            args = ['crack', str(source), '--threshold', '20', '--margin', '6',
                    '--trace', str(trace), '--json', str(report)]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 2)
                self.assertEqual(main(args), 1)
                self.assertEqual(main(['crack', str(source), '--method', 'ai', '--threshold', '20']), 1)
                same = root/'same.json'
                self.assertEqual(main(['crack', str(source), '--trace', str(same), '--json', str(same)]), 1)
                self.assertFalse(same.exists())
            steps = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(len(steps), 8)
            self.assertEqual(steps[0]['key'], 777)
            self.assertIsNone(steps[-1]['key'])
            window = json.loads(report.read_text())['windows'][0]
            self.assertIsNone(window['key'])
            self.assertEqual(window['first_confirmation']['key'], 777)
            self.assertEqual(window['first_confirmation']['end_burst'], 1)

    def test_immediate_quiet_threshold_and_reset(self):
        solver = SilenceSolver()
        burst = quiet_fixture(777)[0]
        session = solver.new_session()
        result = session.push(burst)
        self.assertEqual(result['key'], 777)
        self.assertEqual(result['traffic_seconds'], .08)
        self.assertEqual(result['score'], 29)
        self.assertIsNone(solver.new_session(threshold=30).push(burst)['key'])
        session.reset()
        self.assertIsNone(session.snapshot()['candidate'])
        self.assertEqual(session.snapshot()['bursts'], 0)
        self.assertEqual(session.push(quiet_fixture(22222)[0])['key'], 22222)

    def test_accumulation_and_threshold_control(self):
        solver = SilenceSolver()
        frames = voiced_fixture(777)
        session = solver.new_session()
        first = session.push(frames[0])
        second = session.push(frames[1])
        third = session.push(frames[2])
        self.assertIsNone(first['key'])
        self.assertIsNone(second['key'])
        self.assertEqual(third['key'], 777)
        self.assertGreater(third['score'], second['score'])
        self.assertEqual(solver.assess(frames[:2], threshold=14)['key'], 777)
        self.assertEqual(third, solver.assess(frames[:3]))

    def test_conflict_and_contradiction_retract_decision(self):
        solver = SilenceSolver()
        session = solver.new_session()
        self.assertEqual(session.push(quiet_fixture(777)[0])['key'], 777)
        result = session.push(quiet_fixture(22222)[0])
        self.assertIsNone(result['key'])
        session.reset()
        for burst in voiced_fixture(777, count=3):
            before = session.push(burst)
        self.assertEqual(before['key'], 777)
        # Unused pitch codes in valid frames contradict this candidate.
        natural = np.zeros((4, 49), dtype=np.uint8)
        natural[:, [0, 1, 2, 3, 37, 38, 39]] = [1, 1, 1, 1, 0, 0, 0]
        bad = natural[:, THUMBDV_MAP] ^ Sequence().thumbdv(777)
        after = session.push(bad)
        self.assertIsNone(after['key'])
        matching = next(c for c in after['candidates'] if c['key'] == 777)
        self.assertLess(matching['score'], before['score'])
        self.assertEqual(matching['contradictory_pitch_values'], 4)

    def test_invalid_frames_are_not_evidence_or_contradictions(self):
        session = SilenceSolver().new_session()
        before = session.push(quiet_fixture(777)[0])
        after = session.push(quiet_fixture(22222)[0], [False]*4)
        self.assertEqual(after['key'], 777)
        self.assertEqual(after['score'], before['score'])
        for threshold in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                SilenceSolver().new_session(threshold=threshold)
        with self.assertRaises(ValueError):
            SilenceSolver().new_session(margin=-1)
        with self.assertRaises(ValueError):
            session.push(np.zeros((4, 48)))
        with self.assertRaises(ValueError):
            session.push(np.zeros((4, 49)), [True])


if __name__ == '__main__':
    unittest.main()
