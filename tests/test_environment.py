import unittest

from qec.environment import RepetitionCodeEnv


class EnvironmentTests(unittest.TestCase):
    def test_reset_returns_clean_syndrome(self) -> None:
        """Resetting the environment should return the clean syndrome."""
        env = RepetitionCodeEnv(physical_error_rate=0.0, seed=1)
        self.assertEqual(env.reset(), (0, 0))

    def test_single_flip_updates_syndrome(self) -> None:
        """Flipping one qubit should update the visible syndrome immediately."""
        env = RepetitionCodeEnv(physical_error_rate=0.0, seed=1)
        env.reset()
        step = env.step(1)
        self.assertEqual(step["syndrome"], (1, 0))
        self.assertFalse(step["done"])

    def test_episode_ends_after_fixed_length(self) -> None:
        """The environment should stop once the episode-length limit is reached."""
        env = RepetitionCodeEnv(physical_error_rate=0.0, episode_length=2, seed=1)
        env.reset()
        env.step(0)
        step = env.step(0)
        self.assertTrue(step["done"])

    def test_nine_qubit_single_flip_updates_extended_syndrome(self) -> None:
        """The 9-qubit environment should produce the longer expected syndrome."""
        env = RepetitionCodeEnv(physical_error_rate=0.0, code_length=9, seed=1)
        env.reset()
        step = env.step(5)
        self.assertEqual(step["syndrome"], (0, 0, 0, 1, 1, 0, 0, 0))
        self.assertFalse(step["done"])


if __name__ == "__main__":
    unittest.main()
