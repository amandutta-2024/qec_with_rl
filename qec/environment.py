"""Simple environment for repetition-code decoding."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from config import (
    DEFAULT_REWARD_FAILURE,
    DEFAULT_REWARD_SUCCESS,
)
from qec.codes import (
    ErrorPattern,
    Syndrome,
    action_names,
    apply_action,
    logical_qubit_preserved,
    syndrome_from_errors,
    validate_code_length,
)
from qec.noise import bit_flip_noise
from rl.rewards import potential_from_syndrome


@dataclass
class RepetitionCodeEnv:
    physical_error_rate: float
    code_length: int = 3
    episode_length: Optional[int] = None
    reward_success: float = DEFAULT_REWARD_SUCCESS
    reward_failure: float = DEFAULT_REWARD_FAILURE
    use_reward_shaping: bool = False
    seed: Optional[int] = None
    rng: random.Random = field(init=False)
    errors: ErrorPattern = field(init=False)
    steps_taken: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Initialize RNG state and the clean starting error pattern."""
        validate_code_length(self.code_length)
        self.rng = random.Random(self.seed)
        if self.episode_length is None:
            self.episode_length = self.code_length
        self.errors = (0,) * self.code_length
        self.reset()

    @property
    def action_names(self) -> list[str]:
        """Return the allowed action names for this environment size."""
        return action_names(self.code_length)

    def reset(self) -> Syndrome:
        """Reset the environment to the clean state and return its syndrome."""
        self.errors = (0,) * self.code_length
        self.steps_taken = 0
        return syndrome_from_errors(self.errors)

    def step(self, action: int) -> dict[str, object]:
        """Apply one correction action, then one round of physical noise."""
        if action not in range(len(self.action_names)):
            raise ValueError(f"Invalid action {action}.")

        # One environment step means: apply the chosen correction, then apply fresh noise.
        current_syndrome = syndrome_from_errors(self.errors)
        current_potential = potential_from_syndrome(
            current_syndrome, enabled=self.use_reward_shaping
        )

        self.errors = apply_action(self.errors, action)
        self.errors = bit_flip_noise(self.errors, self.physical_error_rate, self.rng)
        self.steps_taken += 1

        next_syndrome = syndrome_from_errors(self.errors)
        done = self.steps_taken >= self.episode_length
        reward = 0.0

        if done:
            if logical_qubit_preserved(self.errors):
                reward = self.reward_success
            else:
                reward = self.reward_failure

        reward += (
            potential_from_syndrome(next_syndrome, enabled=self.use_reward_shaping)
            - current_potential
        )

        return {
            "syndrome": next_syndrome,
            "reward": reward,
            "done": done,
            "info": {
                "errors": self.errors,
                "success": logical_qubit_preserved(self.errors) if done else None,
            },
        }


# Keep the old name as an alias so older imports still work.
ThreeQubitRepetitionEnv = RepetitionCodeEnv
