"""Hybrid owns the verdict; no lightweight model or exhaustive search needed."""
import unittest
import numpy as np
from std_t98_cracker.guided_ai import GuidedAISolver
from std_t98_cracker.sequence import load_sequence


class StubLinear:
    def new_session(self, *args):
        return StubSession()


class StubSession:
    def __init__(self):
        self.bursts = 0

    def push(self, burst, valid=None):
        self.bursts += 1
        # Deliberately advertise a wrong linear-only key with strong evidence.
        return dict(bursts=self.bursts, traffic_seconds=self.bursts * .08, key=12345,
                    series=1, candidate=12345, candidate_series=1, score=100, margin=50,
                    candidates=[dict(series=1, key=12345, score=100),
                                dict(series=2, key=32767, score=50)])


class ControlledScorer:
    def __init__(self, values):
        self.values = values
        self.calls = 0
        self.samples = 0
        self.blocks = []
        self.masks = [load_sequence(series=s).thumbdv(k) for s, k in ((1, 12345), (2, 32767))]

    def score(self, blocks, masks):
        self.calls += 1
        self.samples += len(masks) * len(blocks)
        self.blocks.extend(blocks.copy())
        return np.array([[self.values[next(i for i, p in enumerate(self.masks) if np.array_equal(p, mask))]]
                         for mask in masks])


class GuidedAITests(unittest.TestCase):
    def test_hybrid_can_choose_linear_runner_up(self):
        scorer = ControlledScorer([-10, 12])
        session = GuidedAISolver(StubLinear(), scorer).new_session()
        burst = np.zeros((4, 49), dtype=np.uint8)
        for _ in range(9):
            self.assertIsNone(session.push(burst)['key'])
        result = session.push(burst)
        self.assertEqual((result['series'], result['key']), (2, 32767))
        self.assertEqual(result['linear']['key'], 12345)
        self.assertFalse(result['verification']['lightweight_model_used'])
        self.assertEqual(result['verification']['block_starts'], [0, 5])
        self.assertEqual(scorer.samples, 4)
        session.push(burst)
        self.assertEqual(scorer.samples, 4)  # No re-scoring identical evidence.
        session.reset()
        self.assertIsNone(session.push(burst)['key'])
        self.assertEqual(session.evaluations, 0)

    def test_rejection_ambiguity_threshold_and_nonfinite(self):
        burst = np.zeros((4, 49), dtype=np.uint8)
        for values, threshold in (([-10, -5], 20), ([12, 11], 20), ([-10, 12], 60)):
            scorer = ControlledScorer(values)
            session = GuidedAISolver(StubLinear(), scorer).new_session(threshold)
            for _ in range(10):
                result = session.push(burst)
            self.assertIsNone(result['key'])
        scorer = ControlledScorer([float('nan'), 12])
        session = GuidedAISolver(StubLinear(), scorer).new_session()
        with self.assertRaises(RuntimeError):
            for _ in range(10):
                session.push(burst)

    def test_bad_burst_breaks_blocks_and_recent_three_are_disjoint(self):
        scorer = ControlledScorer([-10, 12])
        session = GuidedAISolver(StubLinear(), scorer).new_session()
        burst = np.zeros((4, 49), dtype=np.uint8)
        for i in range(31):
            result = session.push(burst, [True, True, i != 3, True])
        self.assertEqual(result['verification']['block_starts'], [14, 19, 24])
        self.assertTrue(all(k[0] in [14, 19, 24] for k in session.cache))
        self.assertEqual(result['verification']['required_positive_blocks'], 2)
        self.assertEqual(result['key'], 32767)


if __name__ == '__main__':
    unittest.main()
