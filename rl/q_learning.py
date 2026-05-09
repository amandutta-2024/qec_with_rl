"""Tabular Q-learning and evaluation helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from config import DEFAULT_ALPHA, DEFAULT_EPSILON, DEFAULT_GAMMA, DEFAULT_UCB_C
from qec.codes import Syndrome, action_names, syndrome_labels
from qec.decoders import lookup_policy
from qec.environment import RepetitionCodeEnv
from rl.policies import best_action, epsilon_greedy_action, ucb_action


@dataclass
class QLearningAgent:
    code_length: int = 3
    alpha: float = DEFAULT_ALPHA
    gamma: float = DEFAULT_GAMMA
    epsilon: float = DEFAULT_EPSILON
    exploration: str = "epsilon_greedy"
    ucb_c: float = DEFAULT_UCB_C
    seed: Optional[int] = None
    q_table: Dict[Syndrome, List[float]] = field(init=False)
    visit_counts: Dict[Syndrome, List[int]] = field(init=False)
    rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        syndromes = syndrome_labels(self.code_length)
        num_actions = len(action_names(self.code_length))
        # Each syndrome gets one value per possible action.
        self.q_table = {syndrome: [0.0] * num_actions for syndrome in syndromes}
        self.visit_counts = {syndrome: [0] * num_actions for syndrome in syndromes}

    def choose_action(self, syndrome: Syndrome, timestep: int) -> int:
        if self.exploration == "epsilon_greedy":
            return epsilon_greedy_action(self.q_table, syndrome, self.epsilon, self.rng)
        if self.exploration == "ucb":
            return ucb_action(
                self.q_table,
                self.visit_counts,
                syndrome,
                timestep,
                self.ucb_c,
                self.rng,
            )
        raise ValueError(f"Unknown exploration strategy: {self.exploration}")

    def update(
        self,
        syndrome: Syndrome,
        action: int,
        reward: float,
        next_syndrome: Syndrome,
        done: bool,
    ) -> None:
        self.visit_counts[syndrome][action] += 1
        best_next = 0.0 if done else max(self.q_table[next_syndrome])
        # Standard one-step Q-learning update.
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[syndrome][action]
        self.q_table[syndrome][action] += self.alpha * td_error

    def greedy_policy(self) -> Dict[Syndrome, int]:
        return {syndrome: best_action(self.q_table, syndrome, self.rng) for syndrome in self.q_table}


def train_agent(
    env: RepetitionCodeEnv,
    agent: QLearningAgent,
    episodes: int,
) -> Dict[str, object]:
    successes = 0
    rewards: List[float] = []
    global_timestep = 1

    for _ in range(episodes):
        syndrome = env.reset()
        done = False
        episode_reward = 0.0
        last_step: Optional[dict[str, object]] = None

        while not done:
            action = agent.choose_action(syndrome, global_timestep)
            step = env.step(action)
            next_syndrome = step["syndrome"]
            reward = step["reward"]
            done = step["done"]

            agent.update(syndrome, action, reward, next_syndrome, done)
            syndrome = next_syndrome
            episode_reward += reward
            last_step = step
            global_timestep += 1

        rewards.append(episode_reward)
        if last_step and last_step["info"]["success"]:
            successes += 1

    return {
        "training_success_rate": successes / episodes,
        "average_training_reward": sum(rewards) / episodes,
        "policy": agent.greedy_policy(),
        "q_table": agent.q_table,
    }


def evaluate_policy(
    env: RepetitionCodeEnv,
    policy: Dict[Syndrome, int],
    episodes: int,
) -> Dict[str, float]:
    successes = 0
    total_reward = 0.0

    for _ in range(episodes):
        syndrome = env.reset()
        done = False
        last_step: Optional[dict[str, object]] = None

        while not done:
            action = policy[syndrome]
            last_step = env.step(action)
            syndrome = last_step["syndrome"]
            total_reward += last_step["reward"]
            done = last_step["done"]

        if last_step and last_step["info"]["success"]:
            successes += 1

    return {
        "success_rate": successes / episodes,
        "average_reward": total_reward / episodes,
    }


def run_sweep(
    error_rates: Sequence[float],
    episodes: int,
    eval_episodes: int,
    exploration: str,
    use_reward_shaping: bool,
    code_length: int,
    seed: Optional[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for index, error_rate in enumerate(error_rates):
        base_seed = None if seed is None else seed + index * 1000

        # Evaluate the fixed lookup decoder before training RL at the same noise level.
        lookup_env = RepetitionCodeEnv(
            physical_error_rate=error_rate,
            code_length=code_length,
            seed=base_seed,
            use_reward_shaping=use_reward_shaping,
        )
        lookup_metrics = evaluate_policy(lookup_env, lookup_policy(code_length), eval_episodes)

        train_env = RepetitionCodeEnv(
            physical_error_rate=error_rate,
            code_length=code_length,
            seed=None if base_seed is None else base_seed + 1,
            use_reward_shaping=use_reward_shaping,
        )
        eval_env = RepetitionCodeEnv(
            physical_error_rate=error_rate,
            code_length=code_length,
            seed=None if base_seed is None else base_seed + 2,
            use_reward_shaping=use_reward_shaping,
        )
        agent = QLearningAgent(
            code_length=code_length,
            exploration=exploration,
            seed=base_seed,
        )
        # Train and then evaluate on a fresh environment so train-time randomness does not leak in.
        train_agent(train_env, agent, episodes)
        q_metrics = evaluate_policy(eval_env, agent.greedy_policy(), eval_episodes)

        rows.append(
            {
                "physical_error_rate": error_rate,
                "lookup_success_rate": lookup_metrics["success_rate"],
                "q_learning_success_rate": q_metrics["success_rate"],
                "q_learning_average_reward": q_metrics["average_reward"],
            }
        )
    return rows
