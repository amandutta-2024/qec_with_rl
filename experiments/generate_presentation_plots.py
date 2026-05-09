"""Generate presentation-ready plots for the repetition-code project."""

from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib.pyplot as plt

from qec.codes import action_names, syndrome_labels
from qec.decoders import lookup_policy
from qec.environment import RepetitionCodeEnv
from qec.utils import ensure_results_dirs
from rl.q_learning import QLearningAgent, evaluate_policy


def train_with_history(
    physical_error_rate: float,
    episodes: int,
    exploration: str,
    reward_shaping: bool,
    seed: int,
) -> Tuple[QLearningAgent, List[float]]:
    env = RepetitionCodeEnv(
        physical_error_rate=physical_error_rate,
        use_reward_shaping=reward_shaping,
        seed=seed,
    )
    agent = QLearningAgent(exploration=exploration, seed=seed)

    success_window: deque[float] = deque(maxlen=250)
    moving_success: List[float] = []
    global_timestep = 1

    for _ in range(episodes):
        syndrome = env.reset()
        done = False
        last_success = 0.0

        while not done:
            action = agent.choose_action(syndrome, global_timestep)
            step = env.step(action)
            next_syndrome = step["syndrome"]
            done = step["done"]
            agent.update(syndrome, action, step["reward"], next_syndrome, done)
            syndrome = next_syndrome
            global_timestep += 1
            if done:
                last_success = 1.0 if step["info"]["success"] else 0.0

        success_window.append(last_success)
        moving_success.append(sum(success_window) / len(success_window))

    return agent, moving_success


def sweep_results(
    error_rates: List[float],
    episodes: int,
    eval_episodes: int,
    exploration: str,
    reward_shaping: bool,
    seed: int,
) -> Dict[str, List[float]]:
    lookup_scores: List[float] = []
    q_scores: List[float] = []

    for index, error_rate in enumerate(error_rates):
        base_seed = seed + index * 1000
        lookup_env = RepetitionCodeEnv(
            physical_error_rate=error_rate,
            seed=base_seed,
            use_reward_shaping=reward_shaping,
        )
        lookup_metrics = evaluate_policy(lookup_env, lookup_policy(3), eval_episodes)
        lookup_scores.append(lookup_metrics["success_rate"])

        agent, _ = train_with_history(
            physical_error_rate=error_rate,
            episodes=episodes,
            exploration=exploration,
            reward_shaping=reward_shaping,
            seed=base_seed + 1,
        )
        eval_env = RepetitionCodeEnv(
            physical_error_rate=error_rate,
            seed=base_seed + 2,
            use_reward_shaping=reward_shaping,
        )
        q_metrics = evaluate_policy(eval_env, agent.greedy_policy(), eval_episodes)
        q_scores.append(q_metrics["success_rate"])

    return {"lookup": lookup_scores, "q_learning": q_scores}


def plot_success_vs_error_rate(output_dir: Path) -> None:
    error_rates = [0.01, 0.03, 0.05, 0.08, 0.10]
    results = sweep_results(
        error_rates=error_rates,
        episodes=12000,
        eval_episodes=2500,
        exploration="epsilon_greedy",
        reward_shaping=False,
        seed=11,
    )

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(
        error_rates,
        results["lookup"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        label="Lookup decoder",
    )
    plt.plot(
        error_rates,
        results["q_learning"],
        color="#d62728",
        marker="s",
        linewidth=2.2,
        label="Q-learning",
    )
    plt.xlabel("Physical error rate p")
    plt.ylabel("Success rate")
    plt.title("Lookup Decoder vs Q-Learning")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "presentation_success_vs_error_rate.png", dpi=220)
    plt.close()


def plot_training_curve(output_dir: Path) -> None:
    _, moving_success = train_with_history(
        physical_error_rate=0.05,
        episodes=8000,
        exploration="epsilon_greedy",
        reward_shaping=False,
        seed=23,
    )

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(
        range(1, len(moving_success) + 1),
        moving_success,
        color="#2ca02c",
        linewidth=2.0,
    )
    plt.xlabel("Training episode")
    plt.ylabel("Moving success rate (window = 250)")
    plt.title("Q-Learning Improves During Training at p = 0.05")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "presentation_training_curve.png", dpi=220)
    plt.close()


def plot_q_table_heatmap(output_dir: Path) -> None:
    agent, _ = train_with_history(
        physical_error_rate=0.05,
        episodes=12000,
        exploration="epsilon_greedy",
        reward_shaping=False,
        seed=37,
    )

    syndromes = sorted(agent.q_table)
    matrix = [agent.q_table[syndrome] for syndrome in syndromes]
    labels = syndrome_labels(3)
    row_labels = [f"{syndrome}\n{labels[syndrome]}" for syndrome in syndromes]

    plt.figure(figsize=(7.8, 4.6))
    image = plt.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(image, label="Q-value")
    names = action_names(3)
    plt.xticks(range(len(names)), names, rotation=20)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title("Learned Q-Table at p = 0.05")

    for row_index, values in enumerate(matrix):
        best_action = max(range(len(values)), key=lambda idx: values[idx])
        for col_index, value in enumerate(values):
            text = f"{value:.2f}"
            weight = "bold" if col_index == best_action else "normal"
            plt.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight=weight,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "presentation_q_table_heatmap.png", dpi=220)
    plt.close()


def main() -> None:
    ensure_results_dirs()
    output_dir = ROOT / "results" / "plots"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    plot_success_vs_error_rate(output_dir)
    plot_training_curve(output_dir)
    plot_q_table_heatmap(output_dir)

    print("Saved plots:")
    print(f"  {output_dir / 'presentation_success_vs_error_rate.png'}")
    print(f"  {output_dir / 'presentation_training_curve.png'}")
    print(f"  {output_dir / 'presentation_q_table_heatmap.png'}")


if __name__ == "__main__":
    main()
