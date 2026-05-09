"""Reward helpers, including potential-based shaping."""

from __future__ import annotations

from qec.codes import Syndrome


def potential_from_syndrome(syndrome: Syndrome, enabled: bool = False) -> float:
    """Return the shaping potential for a syndrome, or zero if shaping is off."""
    if not enabled:
        return 0.0
    return -float(sum(syndrome))
