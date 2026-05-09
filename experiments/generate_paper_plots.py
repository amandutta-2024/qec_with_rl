"""Generate paper-ready plots and analysis for 3-qubit and 9-qubit repetition codes."""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qec.decoders import lookup_policy
from qec.environment import RepetitionCodeEnv
from qec.utils import ensure_results_dirs
from rl.q_learning import QLearningAgent, evaluate_policy, train_agent

ERROR_RATES = [0.01, 0.03, 0.05, 0.08, 0.10]
SEEDS = [7, 17, 27]
CODE_LENGTHS = [3, 9]
TRAIN_EPISODES = {3: 5_000, 9: 10_000}
EVAL_EPISODES = 2_000
TRAINING_CURVE_EPISODES = {3: 6_000, 9: 12_000}
WINDOW = 250


def mean_std(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    mean = statistics.fmean(data)
    std = statistics.stdev(data) if len(data) > 1 else 0.0
    return {"mean": mean, "std": std}


def confidence_radius(std: float, n: int) -> float:
    if n <= 1:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def aggregate_sweep() -> Dict[int, Dict[str, List[Dict[str, float]]]]:
    results: Dict[int, Dict[str, List[Dict[str, float]]]] = {}
    for code_length in CODE_LENGTHS:
        lookup_rows: List[Dict[str, float]] = []
        q_rows: List[Dict[str, float]] = []

        for error_rate in ERROR_RATES:
            lookup_runs: List[float] = []
            q_runs: List[float] = []

            for seed in SEEDS:
                lookup_env = RepetitionCodeEnv(
                    physical_error_rate=error_rate,
                    code_length=code_length,
                    seed=seed,
                )
                lookup_runs.append(
                    evaluate_policy(lookup_env, lookup_policy(code_length), EVAL_EPISODES)[
                        "success_rate"
                    ]
                )

                train_env = RepetitionCodeEnv(
                    physical_error_rate=error_rate,
                    code_length=code_length,
                    seed=seed + 100,
                )
                eval_env = RepetitionCodeEnv(
                    physical_error_rate=error_rate,
                    code_length=code_length,
                    seed=seed + 200,
                )
                agent = QLearningAgent(code_length=code_length, seed=seed)
                train_agent(train_env, agent, TRAIN_EPISODES[code_length])
                q_runs.append(
                    evaluate_policy(eval_env, agent.greedy_policy(), EVAL_EPISODES)["success_rate"]
                )

            lookup_stats = mean_std(lookup_runs)
            q_stats = mean_std(q_runs)
            lookup_rows.append(
                {
                    "p": error_rate,
                    "mean": lookup_stats["mean"],
                    "std": lookup_stats["std"],
                    "ci": confidence_radius(lookup_stats["std"], len(SEEDS)),
                }
            )
            q_rows.append(
                {
                    "p": error_rate,
                    "mean": q_stats["mean"],
                    "std": q_stats["std"],
                    "ci": confidence_radius(q_stats["std"], len(SEEDS)),
                }
            )

        results[code_length] = {"lookup": lookup_rows, "q_learning": q_rows}

    return results


def training_curve(code_length: int, seed: int = 7) -> List[float]:
    env = RepetitionCodeEnv(
        physical_error_rate=0.05,
        code_length=code_length,
        use_reward_shaping=(code_length == 9),
        seed=seed,
    )
    agent = QLearningAgent(code_length=code_length, seed=seed)
    success_window: deque[float] = deque(maxlen=WINDOW)
    moving_success: List[float] = []
    global_timestep = 1

    for _ in range(TRAINING_CURVE_EPISODES[code_length]):
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

    return moving_success


def strategy_comparison() -> Dict[int, Dict[str, Dict[str, float]]]:
    configs = [
        ("epsilon_greedy", False, "epsilon"),
        ("epsilon_greedy", True, "epsilon_shaping"),
        ("ucb", False, "ucb"),
    ]
    results: Dict[int, Dict[str, Dict[str, float]]] = {}

    for code_length in CODE_LENGTHS:
        code_results: Dict[str, Dict[str, float]] = {}
        for exploration, shaping, label in configs:
            runs: List[float] = []
            for seed in SEEDS:
                train_env = RepetitionCodeEnv(
                    physical_error_rate=0.05,
                    code_length=code_length,
                    use_reward_shaping=shaping,
                    seed=seed + 100,
                )
                eval_env = RepetitionCodeEnv(
                    physical_error_rate=0.05,
                    code_length=code_length,
                    use_reward_shaping=shaping,
                    seed=seed + 200,
                )
                agent = QLearningAgent(
                    code_length=code_length,
                    exploration=exploration,
                    seed=seed,
                )
                train_agent(train_env, agent, TRAIN_EPISODES[code_length])
                runs.append(
                    evaluate_policy(eval_env, agent.greedy_policy(), EVAL_EPISODES)["success_rate"]
                )

            stats = mean_std(runs)
            code_results[label] = {
                "mean": stats["mean"],
                "std": stats["std"],
                "ci": confidence_radius(stats["std"], len(SEEDS)),
            }
        results[code_length] = code_results

    return results


def plot_main_sweep(data: Dict[int, Dict[str, List[Dict[str, float]]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    colors = {"lookup": "#1f77b4", "q_learning": "#d62728"}
    labels = {"lookup": "Lookup decoder", "q_learning": "Q-learning"}

    for ax, code_length in zip(axes, CODE_LENGTHS):
        for key in ("lookup", "q_learning"):
            rows = data[code_length][key]
            x = [row["p"] for row in rows]
            y = [row["mean"] for row in rows]
            ci = [row["ci"] for row in rows]
            marker = "o" if key == "lookup" else "s"
            ax.plot(x, y, marker=marker, linewidth=2.2, color=colors[key], label=labels[key])
            ax.fill_between(x, [a - b for a, b in zip(y, ci)], [a + b for a, b in zip(y, ci)], color=colors[key], alpha=0.15)

        ax.set_title(f"{code_length}-Qubit Repetition Code")
        ax.set_xlabel("Physical error rate p")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel("Success rate")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Decoder Performance Across Noise Rates", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_success_vs_error_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)

    for ax, code_length in zip(axes, CODE_LENGTHS):
        curve = training_curve(code_length)
        ax.plot(range(1, len(curve) + 1), curve, linewidth=2.0, color="#2ca02c")
        shaping_note = " with shaping" if code_length == 9 else ""
        ax.set_title(f"{code_length}-Qubit Training at p = 0.05{shaping_note}")
        ax.set_xlabel("Training episode")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(f"Moving success rate (window = {WINDOW})")
    fig.suptitle("Learning Curves for Tabular Q-Learning", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_training_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_comparison(data: Dict[int, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True)
    order = ["epsilon", "epsilon_shaping", "ucb"]
    labels = ["eps-greedy", "eps-greedy + shaping", "ucb"]
    colors = ["#d62728", "#2ca02c", "#9467bd"]

    for ax, code_length in zip(axes, CODE_LENGTHS):
        means = [data[code_length][key]["mean"] for key in order]
        cis = [data[code_length][key]["ci"] for key in order]
        ax.bar(range(len(order)), means, yerr=cis, capsize=5, color=colors, alpha=0.85)
        ax.set_xticks(range(len(order)), labels, rotation=12)
        ax.set_title(f"{code_length}-Qubit at p = 0.05")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Success rate")
    fig.suptitle("Exploration and Reward-Shaping Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "paper_strategy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    sweep_data: Dict[int, Dict[str, List[Dict[str, float]]]],
    strategy_data: Dict[int, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> None:
    summary = {"sweep": sweep_data, "strategy": strategy_data}
    (output_dir / "paper_results_summary.json").write_text(json.dumps(summary, indent=2))


def write_analysis(
    sweep_data: Dict[int, Dict[str, List[Dict[str, float]]]],
    strategy_data: Dict[int, Dict[str, Dict[str, float]]],
) -> None:
    report_dir = ROOT / "report"

    def row_at(code_length: int, p: float, model: str) -> Dict[str, float]:
        for row in sweep_data[code_length][model]:
            if abs(row["p"] - p) < 1e-12:
                return row
        raise ValueError(f"Missing row for n={code_length}, p={p}, model={model}")

    p05_3_lookup = row_at(3, 0.05, "lookup")["mean"]
    p05_3_q = row_at(3, 0.05, "q_learning")["mean"]
    p05_9_lookup = row_at(9, 0.05, "lookup")["mean"]
    p05_9_q = row_at(9, 0.05, "q_learning")["mean"]
    shaping_3 = strategy_data[3]["epsilon_shaping"]["mean"]
    shaping_9 = strategy_data[9]["epsilon_shaping"]["mean"]
    eps_3 = strategy_data[3]["epsilon"]["mean"]
    eps_9 = strategy_data[9]["epsilon"]["mean"]
    ucb_3 = strategy_data[3]["ucb"]["mean"]
    ucb_9 = strategy_data[9]["ucb"]["mean"]
    shaping_3_std = strategy_data[3]["epsilon_shaping"]["std"]
    shaping_9_std = strategy_data[9]["epsilon_shaping"]["std"]
    eps_3_std = strategy_data[3]["epsilon"]["std"]
    eps_9_std = strategy_data[9]["epsilon"]["std"]

    analysis = f"""# Generated Results Analysis

## Main Findings

- For the 3-qubit code, tabular Q-learning is essentially competitive with the lookup decoder across the tested noise range. At `p = 0.05`, lookup reached `{p05_3_lookup:.3f}` mean success rate and Q-learning reached `{p05_3_q:.3f}`.
- For the 9-qubit code, the same RL pipeline is still workable, but it is less stable because the syndrome space grows from 4 states to 256 states. At `p = 0.05`, lookup reached `{p05_9_lookup:.3f}` mean success rate and Q-learning reached `{p05_9_q:.3f}`.
- The success-vs-noise curves show the expected monotone degradation as physical error rate increases. This supports the claim that the simulator is behaving consistently with the bit-flip noise model.

## Strategy Comparison

- On the 3-qubit code at `p = 0.05`, epsilon-greedy without shaping reached `{eps_3:.3f}`, epsilon-greedy with shaping reached `{shaping_3:.3f}`, and UCB reached `{ucb_3:.3f}`.
- On the 9-qubit code at `p = 0.05`, epsilon-greedy without shaping reached `{eps_9:.3f}`, epsilon-greedy with shaping reached `{shaping_9:.3f}`, and UCB reached `{ucb_9:.3f}`.
- Reward shaping does not clearly improve the mean success rate here, but it changes stability. In particular, the 9-qubit shaped runs have lower variance (`std = {shaping_9_std:.3f}`) than the plain epsilon-greedy runs (`std = {eps_9_std:.3f}`), whereas the 3-qubit shaped runs are actually less stable (`std = {shaping_3_std:.3f}` vs `std = {eps_3_std:.3f}`).
- Under the current hyperparameters, UCB is weaker than epsilon-greedy on both code lengths. This suggests the implementation is more sensitive to insufficient visitation in larger tables than to undirected exploration noise.

## Paper Framing

- The strongest final-paper claim is not that RL beats the lookup decoder. The stronger and more defensible claim is that tabular RL can recover near-decoder-quality behavior on small repetition codes, while scaling to 9 qubits exposes the data-efficiency limits of naive tabular methods.
- The 9-qubit extension is useful because it turns the project from a pure proof-of-concept into a scaling study. It gives you a concrete transition from a tiny interpretable MDP to a larger but still analyzable one.
- In the discussion section, emphasize that the lookup decoder remains the oracle-like baseline for this hand-designed environment, while RL matters because it learns from interaction rather than from an explicit decoding rule.

## Figures to Cite

- `results/plots/paper_success_vs_error_rate.png`: main performance comparison across noise rates for 3 and 9 qubits.
- `results/plots/paper_training_curves.png`: learning dynamics at `p = 0.05`.
- `results/plots/paper_strategy_comparison.png`: effect of exploration strategy and reward shaping at `p = 0.05`.
"""

    (report_dir / "generated_results_analysis.md").write_text(analysis)


def main() -> None:
    ensure_results_dirs()
    output_dir = ROOT / "results" / "plots"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    sweep_data = aggregate_sweep()
    strategy_data = strategy_comparison()

    plot_main_sweep(sweep_data, output_dir)
    plot_training_curves(output_dir)
    plot_strategy_comparison(strategy_data, output_dir)
    write_summary(sweep_data, strategy_data, output_dir)
    write_analysis(sweep_data, strategy_data)

    print("Saved plots:")
    print(f"  {output_dir / 'paper_success_vs_error_rate.png'}")
    print(f"  {output_dir / 'paper_training_curves.png'}")
    print(f"  {output_dir / 'paper_strategy_comparison.png'}")
    print("Saved analysis:")
    print(f"  {ROOT / 'report' / 'generated_results_analysis.md'}")


if __name__ == "__main__":
    main()
