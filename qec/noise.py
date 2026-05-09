"""Noise models for the physical qubits."""

from __future__ import annotations

import random
from typing import Sequence, Tuple


def bit_flip_noise(
    errors: Sequence[int],
    physical_error_rate: float,
    rng: random.Random,
) -> Tuple[int, ...]:
    updated = list(errors)
    for qubit in range(len(updated)):
        # Each qubit flips independently with probability p.
        if rng.random() < physical_error_rate:
            updated[qubit] ^= 1
    return tuple(updated)


def sample_depolarizing_pauli(
    physical_error_rate: float,
    rng: random.Random,
) -> Tuple[str, str, str]:
    paulis = []
    for _ in range(3):
        if rng.random() >= physical_error_rate:
            paulis.append("I")
            continue
        paulis.append(rng.choice(["X", "Y", "Z"]))
    return tuple(paulis)
