"""Shared utilities for output formatting and result persistence."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping

from qec.codes import action_names, describe_syndrome

METRICS_FIELDNAMES = [
    "mode",
    "code_length",
    "physical_error_rate",
    "success_rate",
    "average_reward",
    "episodes",
    "exploration",
    "reward_shaping",
    "lookup_success_rate",
]


def ensure_results_dirs() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("results/policies").mkdir(exist_ok=True)
    Path("results/plots").mkdir(exist_ok=True)


def print_policy(policy: Mapping[tuple[int, ...], int], code_length: int) -> None:
    names = action_names(code_length)
    for syndrome in sorted(policy):
        action = policy[syndrome]
        print(
            f"  syndrome={syndrome} ({describe_syndrome(syndrome)}): "
            f"{names[action]}"
        )


def serialize_policy(policy: Mapping) -> Dict[str, object]:
    return {str(key): value for key, value in policy.items()}


def metrics_row(
    mode: str,
    code_length: int,
    physical_error_rate: float,
    success_rate: float,
    average_reward: float,
    episodes: int,
    exploration: str,
    reward_shaping: bool,
    lookup_success_rate: float | None = None,
) -> Dict[str, object]:
    # Keep every saved metric row in the same CSV shape, even if one field is blank.
    return {
        "mode": mode,
        "code_length": code_length,
        "physical_error_rate": physical_error_rate,
        "success_rate": success_rate,
        "average_reward": average_reward,
        "episodes": episodes,
        "exploration": exploration,
        "reward_shaping": int(reward_shaping),
        "lookup_success_rate": lookup_success_rate if lookup_success_rate is not None else "",
    }


def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_metrics(path: Path) -> Iterable[Dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)
