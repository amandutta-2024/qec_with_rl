import unittest

from qec.environment import RepetitionCodeEnv
from rl.q_learning import QLearningAgent, evaluate_policy, train_agent


class QLearningTests(unittest.TestCase):
    def test_update_changes_q_value(self) -> None:
        agent = QLearningAgent(alpha=0.5, gamma=0.0, seed=1)
        agent.update((0, 0), 1, 1.0, (1, 0), True)
        self.assertGreater(agent.q_table[(0, 0)][1], 0.0)

    def test_training_learns_lookup_like_policy_at_zero_noise(self) -> None:
        train_env = RepetitionCodeEnv(physical_error_rate=0.0, seed=1)
        eval_env = RepetitionCodeEnv(physical_error_rate=0.0, seed=2)
        agent = QLearningAgent(epsilon=0.2, seed=1)
        train_agent(train_env, agent, episodes=3000)
        metrics = evaluate_policy(eval_env, agent.greedy_policy(), episodes=200)
        self.assertGreaterEqual(metrics["success_rate"], 0.95)

    def test_nine_qubit_agent_uses_expanded_action_space(self) -> None:
        agent = QLearningAgent(code_length=9, seed=1)
        self.assertEqual(len(agent.q_table), 256)
        self.assertEqual(len(agent.q_table[(0, 0, 0, 0, 0, 0, 0, 0)]), 10)


if __name__ == "__main__":
    unittest.main()
