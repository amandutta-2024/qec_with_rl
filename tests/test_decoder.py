import unittest

from qec.decoders import lookup_policy, optimal_lookup_action


class DecoderTests(unittest.TestCase):
    def test_lookup_actions_match_expected_table(self) -> None:
        expected = {
            (0, 0): 0,
            (1, 0): 1,
            (1, 1): 2,
            (0, 1): 3,
        }
        for syndrome, action in expected.items():
            self.assertEqual(optimal_lookup_action(syndrome), action)

    def test_lookup_policy_has_all_syndromes(self) -> None:
        policy = lookup_policy(3)
        self.assertEqual(len(policy), 4)
        self.assertEqual(policy[(1, 1)], 2)

    def test_lookup_policy_covers_nine_qubit_syndromes(self) -> None:
        policy = lookup_policy(9)
        self.assertEqual(len(policy), 256)
        self.assertEqual(policy[(0, 0, 0, 0, 0, 0, 0, 0)], 0)
        self.assertEqual(policy[(0, 0, 0, 1, 1, 0, 0, 0)], 5)


if __name__ == "__main__":
    unittest.main()
