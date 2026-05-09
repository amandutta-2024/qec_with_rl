"""Code-specific logic for odd-length repetition codes."""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Sequence, Tuple

Syndrome = Tuple[int, ...]
ErrorPattern = Tuple[int, ...]


def validate_code_length(code_length: int) -> None:
    """Reject code lengths that are not valid odd repetition-code sizes."""
    if code_length < 3 or code_length % 2 == 0:
        raise ValueError("Repetition code length must be an odd integer >= 3.")


def action_names(code_length: int) -> list[str]:
    """Return the list of allowed actions for a repetition code of this size."""
    validate_code_length(code_length)
    return ["noop", *[f"flip_q{index}" for index in range(code_length)]]


def all_syndromes(code_length: int) -> Iterable[Syndrome]:
    """Enumerate every possible syndrome bitstring for a given code length."""
    validate_code_length(code_length)
    return product((0, 1), repeat=code_length - 1)


def syndrome_from_errors(errors: Sequence[int]) -> Syndrome:
    """Convert a hidden error pattern into the observable syndrome."""
    if len(errors) < 2:
        raise ValueError("Need at least two qubits to compute a syndrome.")
    return tuple(errors[index] ^ errors[index + 1] for index in range(len(errors) - 1))


def apply_action(errors: Sequence[int], action: int) -> ErrorPattern:
    """Apply a correction action to an error pattern."""
    updated = list(errors)
    if action > 0:
        qubit = action - 1
        if qubit >= len(updated):
            raise ValueError(f"Invalid action {action} for {len(updated)} qubits.")
        updated[qubit] ^= 1
    return tuple(updated)


def logical_qubit_preserved(errors: Sequence[int]) -> bool:
    """Check whether the final state is completely error-free."""
    return tuple(errors) == (0,) * len(errors)


def canonical_error_from_syndrome(syndrome: Syndrome) -> ErrorPattern:
    """Return the minimum-weight error pattern consistent with a syndrome."""
    base = [0]
    for bit in syndrome:
        base.append(base[-1] ^ bit)

    candidate = tuple(base)
    complement = tuple(1 - bit for bit in candidate)
    return candidate if sum(candidate) <= sum(complement) else complement


def describe_syndrome(syndrome: Syndrome) -> str:
    """Turn a syndrome into a short human-readable label."""
    if all(bit == 0 for bit in syndrome):
        return "clean"

    error_pattern = canonical_error_from_syndrome(syndrome)
    flipped = [f"q{index}" for index, bit in enumerate(error_pattern) if bit]
    return "flip " + ", ".join(flipped)


def syndrome_labels(code_length: int) -> Dict[Syndrome, str]:
    """Build readable labels for every syndrome of a given code length."""
    return {syndrome: describe_syndrome(syndrome) for syndrome in all_syndromes(code_length)}
