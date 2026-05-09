"""Baseline decoders for the repetition code."""

from __future__ import annotations

from typing import Dict

from qec.codes import Syndrome, canonical_error_from_syndrome, syndrome_labels


def optimal_lookup_action(syndrome: Syndrome) -> int:
    """Choose the fixed correction action used by the lookup decoder."""
    error_pattern = canonical_error_from_syndrome(syndrome)
    for qubit, bit in enumerate(error_pattern):
        if bit:
            return qubit + 1
    return 0


def lookup_policy(code_length: int) -> Dict[Syndrome, int]:
    """Build the full lookup policy table for a repetition code."""
    return {
        syndrome: optimal_lookup_action(syndrome)
        for syndrome in syndrome_labels(code_length)
    }
