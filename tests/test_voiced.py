import unittest

import numpy as np

from std_t98_cracker.linear import SilenceSolver
from std_t98_cracker.sequence import Sequence, THUMBDV_MAP
from test_unvoiced import fixture as quiet_fixture


def voiced_fixture(key, count=4, stationary=False, illegal=False, jump=False):
    natural = np.random.default_rng(98025).integers(0, 2, (count, 4, 49), dtype=np.uint8)
    natural[:, :, [4, 5, 6, 7, 35]] = 0
    for burst in range(count):
        for phase in range(4):
            pitch = 40 + phase + (0 if stationary else burst)
            if illegal:
                pitch = 120 + phase
            elif jump and phase == 3:
                pitch += 30
            natural[burst, phase, [0, 1, 2, 3, 37, 38, 39]] = [int(x) for x in f'{pitch:07b}']
    return natural[:, :, THUMBDV_MAP] ^ Sequence().thumbdv(key)[None]


class VoicedTests(unittest.TestCase):
    def test_keys_and_minimum_independent_bursts(self):
        solver = SilenceSolver()
        for key in (0, 1, 777, 22222, 32767):
            data = voiced_fixture(key)
            self.assertIsNone(solver.recover(data[:2]))
            self.assertEqual(solver.recover(data[:3]), key)

    def test_pitch_and_fec_guards(self):
        solver = SilenceSolver()
        for guard in ('stationary', 'illegal', 'jump'):
            self.assertIsNone(solver.recover(voiced_fixture(777, **{guard: True})))
        data = voiced_fixture(777)
        valid = np.ones((4, 4), dtype=bool)
        valid[:, 0] = False
        self.assertIsNone(solver.recover(data, valid))
        valid[1, 0] = True
        self.assertEqual(solver.recover(data, valid), 777)

    def test_conflicts_are_not_hidden_by_other_constraints(self):
        solver = SilenceSolver()
        a, b = voiced_fixture(777), voiced_fixture(22222)
        self.assertIsNone(solver.recover(np.concatenate([a, b])))
        self.assertIsNone(solver.recover(np.concatenate([a, quiet_fixture(22222)])))
        self.assertIsNone(solver.recover(np.concatenate([a, b, quiet_fixture(777)])))
        self.assertEqual(solver.recover(np.concatenate([a, quiet_fixture(777)])), 777)


if __name__ == '__main__':
    unittest.main()
