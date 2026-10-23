import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np

from std_t98_cracker.burst import read_bursts, write_bursts, pack49, unpack49, read_llr
from std_t98_cracker.cli import main
from std_t98_cracker.fec import FecDecoder
from std_t98_cracker.linear import SilenceSolver
from std_t98_cracker.sequence import Sequence, THUMBDV_MAP, load_sequence

MEASURED = [bytes.fromhex(x) for x in (
    "deba0301a6e600", "9f9ac301a64500", "de9cc909b6e600",
    "df9ac201a64c00", "9f9ac301a6e700", "deba0109b6e600")]


class SolverTests(unittest.TestCase):
    def setUp(self):
        self.solver = SilenceSolver()
        self.sequence = Sequence()
        self.plain = np.unpackbits(np.array([list(x) for x in MEASURED], dtype=np.uint8), axis=-1)[:, :49]

    def test_measured_frames_at_every_phase(self):
        keys = [0, 1, 2, 12345, 32767] + list(np.random.default_rng(98).integers(1, 32768, 128))
        for key in keys:
            for phase in range(4):
                for frame in self.plain:
                    encrypted = frame ^ self.sequence.thumbdv(int(key))[phase]
                    self.assertEqual(self.solver.candidate(encrypted, phase), key)

    def pair(self, key):
        return self.plain[:4] ^ self.sequence.thumbdv(key)

    def test_confirmation_conflicts_and_invalid_frames(self):
        a, b = self.pair(12345), self.pair(777)
        mask = np.array([[1, 0, 0, 0]], dtype=bool)
        # A full strict profile carries enough evidence at the default threshold.
        self.assertEqual(self.solver.recover(a[None], mask), 12345)
        self.assertIsNone(self.solver.assess(a[None], mask, threshold=30)['key'])
        mask[0, 2] = True
        self.assertEqual(self.solver.recover(a[None], mask), 12345)
        self.assertIsNone(self.solver.recover(np.array([a, b])))
        self.assertIsNone(self.solver.recover(np.array([self.pair(0), b])))
        self.assertEqual(self.solver.recover(self.pair(0)[None]), 0)
        bad = self.plain[0].copy()
        bad[np.where(THUMBDV_MAP == 13)[0][0]] ^= 1
        for phase in range(4):
            self.assertIsNone(self.solver.candidate(bad ^ self.sequence.thumbdv(12345)[phase], phase))

    def test_plugin_and_rejection(self):
        plugin = load_sequence(Path(__file__).resolve().parents[1] / 'examples/sequence.py')
        np.testing.assert_array_equal(plugin.bits(12345), self.sequence.bits(12345))
        self.assertEqual(SilenceSolver(plugin).recover(self.pair(12345)[None]), 12345)
        with self.assertRaises(ValueError):
            SilenceSolver(Sequence(lambda key: np.ones(196, dtype=np.uint8)))
        with self.assertRaises(ValueError):
            Sequence(lambda key: [0] * 195).bits(1)


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_roundtrip_and_no_clobber(self):
        for bits, width in ((72, 9), (49, 7)):
            p = np.random.default_rng(bits).integers(0, 256, (8, 4, width), dtype=np.uint8)
            if bits == 49:
                p[:, :, -1] &= 128
            path = self.root / f'{bits}.burst'
            write_bursts(path, p, bits)
            found, q = read_bursts(path)
            self.assertEqual(found, bits)
            np.testing.assert_array_equal(p, q)
            with self.assertRaises(FileExistsError):
                write_bursts(path, p, bits)

    def test_reject_damage(self):
        good = b'\xff' + (b'\x31' + b'\0' * 7) * 4
        for data in (b'', good[:-1], b'\x00'+good[1:], good[:9]+b'\x48'+good[10:], good[:-1]+b'\x01'):
            path = self.root / 'bad.burst'
            path.write_bytes(data)
            with self.assertRaises(ValueError):
                read_bursts(path)
        with self.assertRaises(ValueError):
            read_llr(path, 1)

    def test_cli_linear_and_explicit_descramble(self):
        plain = np.tile(np.array([list(x) for x in MEASURED[:4]], dtype=np.uint8)[None], (6, 1, 1))
        encrypted = pack49(unpack49(plain) ^ Sequence().thumbdv(12345)[None])
        source = self.root / 'input.burst'
        write_bursts(source, encrypted)
        with contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['crack', str(source), '--method', 'linear']), 0)
            self.assertIn('12345', output.getvalue())
            base = self.root / 'clear'
            self.assertEqual(main(['descramble', str(source), '--key', '12345', '-o', str(base)]), 0)
            self.assertEqual(main(['descramble', str(source), '--key', '12345', '-o', str(base)]), 1)
        np.testing.assert_array_equal(read_bursts(self.root / 'clear.burst')[1], plain)

    def test_native_fec_sign_and_validation(self):
        fec = FecDecoder()
        p = np.zeros((1, 4, 9), dtype=np.uint8)
        llr = (np.unpackbits(p, axis=-1).astype(np.int8)*2-1)*60
        a, good, errors = fec.fec(p)
        b, soft_good, soft_errors = fec.fec(p, llr)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(good, soft_good)
        with self.assertRaises(ValueError):
            fec.fec(p, llr.astype(np.float32))
        with self.assertRaises(ValueError):
            fec.fec(p, llr[:, :, :71])


if __name__ == '__main__':
    unittest.main()
