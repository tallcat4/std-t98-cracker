"""Opt-in GPU kernel/ranking checks: T98_RUN_CUDA_TESTS=1."""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('T98_RUN_CUDA_TESTS') == '1', 'explicit CUDA integration test')
class CudaSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from std_t98_cracker.enhanced_sweep import MaskSweep
        blocks = np.random.default_rng(98).integers(0, 2, (3, 5, 4, 49), dtype=np.uint8)
        cls.engine = MaskSweep(blocks, prefilter_margin=-10)

    def test_gpu_pn_against_scalar_recurrence(self):
        from std_t98_cracker.enhanced_sequence import arbitrary_pn, coefficient_table
        from std_t98_cracker.sequence import THUMBDV_MAP
        masks = [0, 3, 127, 17893, 32767]
        with self.engine.t.inference_mode():
            for mask, coeff in zip(masks, coefficient_table(masks)):
                pn = self.engine.pn(coeff).cpu().numpy()
                for state in (1, 24, 12345, 32767):
                    np.testing.assert_array_equal(pn[state-1], arbitrary_pn(mask, state).reshape(4, 49)[:, THUMBDV_MAP])

    def test_rank_bound_preserves_top_candidates(self):
        from std_t98_cracker.enhanced_sequence import coefficient_table
        for coefficients in coefficient_table([3, 255, 1023]):
            full, count = self.engine.score_mask(coefficients, 20)
            self.assertTrue(full)
            for cutoff in (full[0]['minimum_margin']+1, full[-1]['minimum_margin']-0.01,
                           (full[0]['minimum_margin']+full[-1]['minimum_margin'])/2):
                bounded, retained = self.engine.score_mask(coefficients, 20, cutoff)
                expected = [r for r in full if r['minimum_margin'] >= cutoff]
                self.assertEqual(retained, count)
                self.assertEqual([r['initial_state'] for r in expected], [r['initial_state'] for r in bounded])

    def test_shared_graph_pool_matches_fp32_and_owns_outputs(self):
        from std_t98_cracker.cuda_graph import GraphRunner
        t = self.engine.t
        runner = GraphRunner(self.engine.heavy.model, self.engine.heavy.device, 512)
        runner.warmup()
        stored = []
        with t.inference_mode():
            for size in (1, 17, 65, 257, 512, 3, 128):
                inputs = t.randint(0, 2, (size, 20, 49), device=self.engine.heavy.device).float()
                expected = self.engine.heavy.model(inputs)
                found = runner(inputs)
                t.testing.assert_close(found, expected, rtol=1e-5, atol=1e-4)
                self.assertTrue(t.equal(found.argmax(1), expected.argmax(1)))
                stored.append((found, found.clone()))
            for found, snapshot in stored:
                self.assertTrue(t.equal(found, snapshot))


if __name__ == '__main__':
    unittest.main()
