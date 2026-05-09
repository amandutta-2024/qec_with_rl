"""Exploration and policy helpers."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def best_action(
    q_table: Dict[Tuple[int, int], List[float]],
    syndrome: Tuple[int, int],
    rng: random.Random,
) -> int:
    """Return a highest-value action, breaking ties at random."""
    values = q_table[syndrome]
    best_value = max(values)
    best_actions = [index for index, value in enumerate(values) if value == best_value]
    return rng.choice(best_actions)


def epsilon_greedy_action(
    q_table: Dict[Tuple[int, int], List[float]],
    syndrome: Tuple[int, int],
    epsilon: float,
    rng: random.Random,
) -> int:
    """Choose a random action with probability epsilon, otherwise go greedy."""
    if rng.random() < epsilon:
        return rng.randrange(len(q_table[syndrome]))
    return best_action(q_table, syndrome, rng)


def ucb_action(
    q_table: Dict[Tuple[int, int], List[float]],
    visit_counts: Dict[Tuple[int, int], List[int]],
    syndrome: Tuple[int, int],
    timestep: int,
    ucb_c: float,
    rng: random.Random,
) -> int:
    """Choose an action using an upper-confidence-bound exploration score."""
    total_visits = max(1, timestep)
    scores = []
    for action, value in enumerate(q_table[syndrome]):
        count = visit_counts[syndrome][action]
        if count == 0:
            return action
        # UCB adds a bonus for actions that have been tried less often.
        bonus = ucb_c * math.sqrt(math.log(total_visits) / count)
        scores.append(value + bonus)
    best_score = max(scores)
    best_actions = [action for action, score in enumerate(scores) if score == best_score]
    return rng.choice(best_actions)
