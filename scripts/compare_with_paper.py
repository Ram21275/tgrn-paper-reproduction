#!/usr/bin/env python3
"""Compare one MOSI result JSON with Table II of the TgRN paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PAPER_MOSI_COMPLETE = {
    "has0_acc2": 0.8299,
    "has0_f1": 0.8305,
    "non0_acc2": 0.8506,
    "non0_f1": 0.8514,
    "mult_acc5": 0.5442,
    "mult_acc7": 0.4643,
    "mae": 0.735,
    "corr": 0.791,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path, help="test_metrics.json produced by train.py")
    args = parser.parse_args()
    observed = json.loads(args.metrics.read_text(encoding="utf-8"))

    print(f"{'metric':<14} {'paper':>10} {'run':>10} {'run-paper':>12}")
    print("-" * 48)
    for name, paper_value in PAPER_MOSI_COMPLETE.items():
        if name not in observed:
            print(f"{name:<14} {paper_value:>10.4f} {'missing':>10} {'-':>12}")
            continue
        run_value = float(observed[name])
        print(f"{name:<14} {paper_value:>10.4f} {run_value:>10.4f} {run_value - paper_value:>+12.4f}")


if __name__ == "__main__":
    main()
