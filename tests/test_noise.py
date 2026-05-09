import random
import unittest

from qec.noise import bit_flip_noise, sample_depolarizing_pauli


class NoiseTests(unittest.TestCase):
    def test_zero_noise_leaves_state_unchanged(self) -> None:
        rng = random.Random(1)
        self.assertEqual(bit_flip_noise((0, 1, 0), 0.0, rng), (0, 1, 0))

    def test_full_noise_flips_all_qubits(self) -> None:
        rng = random.Random(1)
        self.assertEqual(bit_flip_noise((0, 1, 0), 1.0, rng), (1, 0, 1))

    def test_full_noise_flips_all_qubits_for_nine_qubit_state(self) -> None:
        rng = random.Random(1)
        self.assertEqual(
            bit_flip_noise((0, 1, 0, 1, 0, 1, 0, 1, 0), 1.0, rng),
            (1, 0, 1, 0, 1, 0, 1, 0, 1),
        )

    def test_depolarizing_sampler_returns_valid_paulis(self) -> None:
        rng = random.Random(1)
        sample = sample_depolarizing_pauli(0.5, rng)
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(pauli in {"I", "X", "Y", "Z"} for pauli in sample))


if __name__ == "__main__":
    unittest.main()
