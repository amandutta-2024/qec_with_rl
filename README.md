# Reinforcement Learning for Decoding Repetition Codes

This project implements a small quantum error-correction simulator and a tabular
Q-learning decoder for odd-length repetition codes. The default configuration
uses the 3-qubit repetition code, and the same pipeline can be run for larger
codes such as the 9-qubit repetition code. The codebase is organized for
coursework experiments: baseline evaluation, RL training, sweeps over physical
error rates, saved results, plots, and a report scaffold.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py baseline
python3 main.py train
python3 main.py sweep
python3 experiments/analyze_results.py --input results/metrics.csv
python3 main.py sweep --code-length 9 --save
python3 experiments/analyze_results.py --input results/metrics.csv --code-length 9
```

## Project Layout

- `main.py`: central CLI for baseline, train, and sweep.
- `qec/`: repetition-code logic, environment, noise, decoders, shared helpers.
- `rl/`: Q-learning, exploration strategies, and reward shaping utilities.
- `experiments/`: scripts for repeatable runs and offline analysis.
- `results/`: generated metrics, policies, and plots.
- `tests/`: unit tests for environment, decoder, RL updates, and noise.
- `report/`: final writeup scaffold.

## Notes

- The repetition-code implementation supports odd code lengths via
  `--code-length`; `3` remains the default.
- The environment models the correctable X-error component directly, so this is
  a repetition-code bit-flip decoder rather than a full Shor-code simulator.
- A simple depolarizing helper is included for extension work, but the core
  experiments use bit-flip noise because that matches the code.
