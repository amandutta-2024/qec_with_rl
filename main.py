"""CLI entry point for repetition-code RL experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import (
    DEFAULT_ALPHA,
    DEFAULT_BASELINE_EPISODES,
    DEFAULT_EPSILON,
    DEFAULT_ERROR_RATE,
    DEFAULT_ERROR_RATES,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_GAMMA,
    DEFAULT_SEED,
    DEFAULT_SWEEP_EPISODES,
    DEFAULT_SWEEP_EVAL_EPISODES,
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_UCB_C,
)
from qec.decoders import lookup_policy, optimal_lookup_action
from qec.environment import RepetitionCodeEnv
from qec.utils import ensure_results_dirs, metrics_row, print_policy, serialize_policy
from rl.q_learning import QLearningAgent, evaluate_policy, run_sweep, train_agent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tabular Q-learning for odd-length repetition codes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline", help="Evaluate lookup decoder.")
    baseline_parser.add_argument("--p", type=float, default=DEFAULT_ERROR_RATE)
    baseline_parser.add_argument("--episodes", type=int, default=DEFAULT_BASELINE_EPISODES)
    baseline_parser.add_argument("--code-length", type=int, default=3)
    baseline_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    baseline_parser.add_argument("--save", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train and evaluate Q-learning.")
    train_parser.add_argument("--p", type=float, default=DEFAULT_ERROR_RATE)
    train_parser.add_argument("--episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    train_parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    train_parser.add_argument("--code-length", type=int, default=3)
    train_parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    train_parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    train_parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    train_parser.add_argument(
        "--exploration",
        choices=["epsilon_greedy", "ucb"],
        default="epsilon_greedy",
    )
    train_parser.add_argument("--ucb-c", type=float, default=DEFAULT_UCB_C)
    train_parser.add_argument("--reward-shaping", action="store_true")
    train_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train_parser.add_argument("--save", action="store_true")

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Compare lookup and Q-learning across multiple noise rates.",
    )
    sweep_parser.add_argument("--error-rates", type=float, nargs="+", default=DEFAULT_ERROR_RATES)
    sweep_parser.add_argument("--episodes", type=int, default=DEFAULT_SWEEP_EPISODES)
    sweep_parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_SWEEP_EVAL_EPISODES,
    )
    sweep_parser.add_argument("--code-length", type=int, default=3)
    sweep_parser.add_argument(
        "--exploration",
        choices=["epsilon_greedy", "ucb"],
        default="epsilon_greedy",
    )
    sweep_parser.add_argument("--reward-shaping", action="store_true")
    sweep_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sweep_parser.add_argument("--save", action="store_true")

    return parser


def command_baseline(args: argparse.Namespace) -> None:
    env = RepetitionCodeEnv(
        physical_error_rate=args.p,
        code_length=args.code_length,
        seed=args.seed,
    )
    policy = lookup_policy(args.code_length)
    metrics = evaluate_policy(env, policy, args.episodes)
    print(f"Baseline lookup decoder for n={args.code_length} at p={args.p:.3f}")
    print(f"  success_rate={metrics['success_rate']:.4f}")
    print(f"  average_reward={metrics['average_reward']:.4f}")
    print("  policy:")
    print_policy(policy, args.code_length)

    if args.save:
        ensure_results_dirs()
        row = metrics_row(
            mode="baseline",
            code_length=args.code_length,
            physical_error_rate=args.p,
            success_rate=metrics["success_rate"],
            average_reward=metrics["average_reward"],
            episodes=args.episodes,
            exploration="lookup",
            reward_shaping=False,
        )
        metrics_path = Path("results/metrics.csv")
        from qec.utils import append_metrics_row

        append_metrics_row(metrics_path, row)


def command_train(args: argparse.Namespace) -> None:
    train_env = RepetitionCodeEnv(
        physical_error_rate=args.p,
        code_length=args.code_length,
        use_reward_shaping=args.reward_shaping,
        seed=args.seed,
    )
    eval_env = RepetitionCodeEnv(
        physical_error_rate=args.p,
        code_length=args.code_length,
        use_reward_shaping=args.reward_shaping,
        seed=args.seed + 1,
    )
    agent = QLearningAgent(
        code_length=args.code_length,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        exploration=args.exploration,
        ucb_c=args.ucb_c,
        seed=args.seed,
    )
    training_metrics = train_agent(train_env, agent, args.episodes)
    eval_metrics = evaluate_policy(eval_env, agent.greedy_policy(), args.eval_episodes)

    print(f"Q-learning decoder for n={args.code_length} at p={args.p:.3f}")
    print(f"  exploration={args.exploration}")
    print(f"  reward_shaping={args.reward_shaping}")
    print(f"  training_success_rate={training_metrics['training_success_rate']:.4f}")
    print(f"  eval_success_rate={eval_metrics['success_rate']:.4f}")
    print("  learned policy:")
    print_policy(training_metrics["policy"], args.code_length)
    print("  q_table:")
    for syndrome, values in sorted(training_metrics["q_table"].items()):
        display = ", ".join(f"{value:.3f}" for value in values)
        print(f"    {syndrome}: [{display}]")

    if args.save:
        ensure_results_dirs()
        metrics_path = Path("results/metrics.csv")
        policy_path = Path("results/policies") / (
            f"policy_n_{args.code_length}_p_{args.p:.3f}_"
            f"{args.exploration}_shaping_{int(args.reward_shaping)}.json"
        )
        from qec.utils import append_metrics_row

        append_metrics_row(
            metrics_path,
            metrics_row(
                mode="train",
                code_length=args.code_length,
                physical_error_rate=args.p,
                success_rate=eval_metrics["success_rate"],
                average_reward=eval_metrics["average_reward"],
                episodes=args.episodes,
                exploration=args.exploration,
                reward_shaping=args.reward_shaping,
            ),
        )
        clean_syndrome = tuple(0 for _ in range(args.code_length - 1))
        policy_payload = {
            "config": vars(args),
            "optimal_lookup_action_clean": optimal_lookup_action(clean_syndrome),
            "policy": serialize_policy(training_metrics["policy"]),
            "q_table": serialize_policy(training_metrics["q_table"]),
        }
        policy_path.write_text(json.dumps(policy_payload, indent=2))


def command_sweep(args: argparse.Namespace) -> None:
    rows = run_sweep(
        error_rates=args.error_rates,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        exploration=args.exploration,
        use_reward_shaping=args.reward_shaping,
        code_length=args.code_length,
        seed=args.seed,
    )
    print("n\tp\tlookup\tq_learning")
    for row in rows:
        print(
            f"{args.code_length}\t"
            f"{row['physical_error_rate']:.3f}\t"
            f"{row['lookup_success_rate']:.3f}\t"
            f"{row['q_learning_success_rate']:.3f}"
        )

    if args.save:
        ensure_results_dirs()
        from qec.utils import append_metrics_row

        metrics_path = Path("results/metrics.csv")
        for row in rows:
            append_metrics_row(
                metrics_path,
                metrics_row(
                    mode="sweep",
                    code_length=args.code_length,
                    physical_error_rate=row["physical_error_rate"],
                    success_rate=row["q_learning_success_rate"],
                    average_reward=row["q_learning_average_reward"],
                    episodes=args.episodes,
                    exploration=args.exploration,
                    reward_shaping=args.reward_shaping,
                    lookup_success_rate=row["lookup_success_rate"],
                ),
            )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        command_baseline(args)
        return
    if args.command == "train":
        command_train(args)
        return
    if args.command == "sweep":
        command_sweep(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
