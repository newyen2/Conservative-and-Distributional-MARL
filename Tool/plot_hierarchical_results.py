"""Generate legacy-style reward plots for hierarchical training runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_eval_returns(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read evaluation episode numbers and mean returns from one run."""
    episodes = []
    returns = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            episodes.append(int(row["training_episode"]))
            returns.append(float(row["mean_return"]))

    return (
        np.asarray(episodes, dtype=np.int64),
        np.asarray(returns, dtype=np.float64),
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Match the legacy ten-episode rolling mean without partial windows."""
    smoothed = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size >= window:
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed[window - 1 :] = np.convolve(
            values,
            kernel,
            mode="valid",
        )
    return smoothed


def plot_run(
    csv_path: Path,
    output_path: Path,
    window: int,
) -> None:
    """Render one reward curve using the layout of Stored_Datas/4/result.png."""
    episodes, returns = read_eval_returns(csv_path)
    smoothed_returns = rolling_mean(returns, window)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.plot(episodes, smoothed_returns, label="Reward")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Reward")
    axis.set_title("Online HSAC Reward")
    axis.legend()
    axis.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100)
    plt.close(figure)


def resolve_eval_csv(base_dir: Path, run: str) -> Path:
    """Locate a run CSV; run 6 currently remains in Stored_Datas root."""
    run_csv = base_dir / run / "hierarchical_eval_returns.csv"
    if run_csv.exists():
        return run_csv

    if run == "6":
        root_csv = base_dir / "hierarchical_eval_returns.csv"
        if root_csv.exists():
            return root_csv

    raise FileNotFoundError(
        f"Cannot find hierarchical_eval_returns.csv for run {run}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create legacy-style HSAC evaluation reward plots."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("Stored_Datas"),
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["5", "6", "7"],
    )
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    for run in args.runs:
        csv_path = resolve_eval_csv(args.base_dir, str(run))
        output_path = args.base_dir / str(run) / "result.png"
        plot_run(csv_path, output_path, max(1, int(args.window)))
        print(f"run {run}: {csv_path} -> {output_path}")


if __name__ == "__main__":
    main()
