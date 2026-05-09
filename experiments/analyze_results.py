"""Load saved metrics and generate a simple plot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib.pyplot as plt

from qec.utils import ensure_results_dirs, read_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved experiment metrics.")
    parser.add_argument("--input", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/plots/success_vs_p.png"))
    parser.add_argument("--code-length", type=int, default=3)
    args = parser.parse_args()

    rows = list(read_metrics(args.input))
    if not rows:
        raise ValueError(f"No metrics found in {args.input}.")

    sweep_rows = [
        row
        for row in rows
        if row["mode"] == "sweep" and int(row.get("code_length", 3)) == args.code_length
    ]
    if not sweep_rows:
        raise ValueError(
            f"No sweep rows found for code length {args.code_length}. "
            "Run experiments/run_sweep.py first."
        )

    sweep_rows.sort(key=lambda row: float(row["physical_error_rate"]))
    x_values = [float(row["physical_error_rate"]) for row in sweep_rows]
    q_values = [float(row["success_rate"]) for row in sweep_rows]
    lookup_values = [float(row["lookup_success_rate"]) for row in sweep_rows]

    ensure_results_dirs()
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(x_values, lookup_values, marker="o", label="Lookup decoder")
    plt.plot(x_values, q_values, marker="s", label="Q-learning")
    plt.xlabel("Physical error rate p")
    plt.ylabel("Success rate")
    plt.title(f"Decoder Success Rate vs Physical Error Rate (n={args.code_length})")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
