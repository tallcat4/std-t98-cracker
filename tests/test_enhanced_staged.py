import unittest
import importlib.util
from types import SimpleNamespace
import tempfile
from pathlib import Path
from unittest.mock import patch
import contextlib
import io
import numpy as np

from std_t98_cracker.enhanced_staged import spaced_blocks, STAGES


class WindowTests(unittest.TestCase):
    def test_checkpoint_resume_and_parameter_guard(self):
        from std_t98_cracker.enhanced_staged import run_staged
        calls = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root/'weights.bin'
            weights.write_bytes(b'test model')
            class FakeEngine:
                graphs = False
                light_path = weights
                heavy = SimpleNamespace(describe=lambda: {'model': 'test'})
                def __init__(self, *args):
                    pass
                def score_mask(self, coefficients, top):
                    calls.append(1)
                    return [dict(initial_state=10, final_margins=[3, 2, -1], final_plain_votes=2,
                                 median_margin=2.0, mean_margin=4/3)], dict(candidates=32767, light_pass=2, hybrid_pass=1, final_pass=1)
            blocks = np.zeros((7, 5, 4, 49), dtype=np.uint8)
            plan = {'gap_seconds': 2.0}
            with patch('std_t98_cracker.enhanced_staged.StagedSweep', FakeEngine), contextlib.redirect_stdout(io.StringIO()):
                first = run_staged(blocks, root/'search', 5, plan, mask_stop=3)
                second = run_staged(blocks, root/'search', 5, plan, mask_stop=3, resume=True)
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 3)
                self.assertEqual(first['candidates'], 3*32767)
                self.assertEqual(first['top_candidates'][0]['offset'], 5)
                with self.assertRaises(ValueError):
                    run_staged(blocks, root/'search', 5, plan, mask_stop=3, light_margin=-10, resume=True)
                blocks[0, 0, 0, 0] = 1
                with self.assertRaises(ValueError):
                    run_staged(blocks, root/'search', 5, plan, mask_stop=3, resume=True)

    def test_disjoint_spaced_holdout(self):
        frames = np.arange(130*4*49).reshape(130, 4, 49)
        valid = np.ones((130, 4), dtype=bool)
        valid[40:45] = False
        blocks, plan = spaced_blocks(frames, valid)
        self.assertEqual(blocks.shape, (7, 5, 4, 49))
        starts = plan['chronological_starts']
        used = []
        for group in STAGES:
            selected = [starts[i] for i in group]
            self.assertTrue(all(b-a >= 25 for a, b in zip(selected, selected[1:])))
            for start in selected:
                used.extend(range(start, start+5))
                self.assertTrue(valid[start:start+5].all())
        self.assertEqual(len(set(used)), 35)
        for block, start in zip(blocks, plan['light_starts']+plan['hybrid_starts']+plan['final_starts']):
            np.testing.assert_array_equal(block, frames[start:start+5])

    def test_short_capture_and_reused_windows_rejected(self):
        frames = np.zeros((100, 4, 49), dtype=np.uint8)
        valid = np.ones((100, 4), dtype=bool)
        for count in (0, 4, 20, 54):
            with self.assertRaises(ValueError):
                spaced_blocks(frames[:count], valid[:count])
        with self.assertRaises(ValueError):
            spaced_blocks(frames, valid, starts=[0, 10, 20, 20, 40, 50, 60])
        with self.assertRaises(ValueError):
            spaced_blocks(frames, valid, starts=[0, 5, 10, 15, 20, 25, 30])
        with self.assertRaises(ValueError):
            spaced_blocks(frames, valid, gap_seconds=float('nan'))

    def test_minimum_duration(self):
        frames = np.zeros((55, 4, 49), dtype=np.uint8)
        _, plan = spaced_blocks(frames, np.ones((55, 4), bool))
        self.assertEqual(plan['final_starts'], [0, 25, 50])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional AI dependency')
class StageLogicTests(unittest.TestCase):
    def test_or_stages_and_unseen_two_of_three(self):
        import torch
        from std_t98_cracker.enhanced_staged import StagedSweep
        # Deliberately independent outcomes: candidates pass different sides
        # of each OR, and a huge single positive must fail the final vote.
        margins = torch.tensor([
            [1, -1, 1, -1, 4, 3, -100],
            [-1, 1, -1, 1, 5, -100, 6],
            [-1, -1, 9, 9, 9, 9, 9],
            [1, 1, -1, -1, 9, 9, 9],
            [1, 1, 1, 1, 999, -1, -1],
            [1, 1, 1, 1, 8, 7, 9],
        ], dtype=torch.float32)
        engine = StagedSweep.__new__(StagedSweep)
        engine.t = torch
        engine.heavy = SimpleNamespace(device=torch.device('cpu'))
        engine.prefilter_margin = engine.hybrid_margin = 0.0
        engine.states = torch.arange(1, 7)
        pn = torch.zeros((6, 4, 49), dtype=torch.uint8)
        pn[:, 0, 0] = torch.arange(6)
        engine.pn = lambda coefficients: pn
        observed = []
        def score(selected, slot, light=False):
            ids = selected[:, 0, 0].long()
            observed.append((slot, ids.tolist()))
            return margins[ids, slot]
        engine.margins = score
        rows, counts = engine.score_mask(None)
        self.assertEqual([r['initial_state'] for r in rows], [6, 2, 1])
        self.assertEqual([r['final_plain_votes'] for r in rows], [3, 2, 2])
        self.assertEqual((counts['light_pass'], counts['hybrid_pass'], counts['final_pass']), (5, 4, 3))
        self.assertEqual(observed[1], (1, [1, 2]))
        self.assertEqual(observed[3], (3, [1, 3]))
        self.assertTrue(all(2 not in ids and 3 not in ids for slot, ids in observed if slot >= 4))


if __name__ == '__main__':
    unittest.main()
