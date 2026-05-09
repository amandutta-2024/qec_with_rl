"""Run Q-learning training and save metrics/policies."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    sys.argv = ["run_training.py", "train", "--save", *sys.argv[1:]]
    main()
