"""
plot_results.py

Reads results/exp1_raw_chains.json and produces a plot comparing mean
variance trajectories for correct vs. incorrect reasoning chains, with
shaded confidence bands (±1 SEM).

Usage:
    python src/plot_results.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_trajectories():
    with open(RESULTS_DIR / "exp1_raw_chains.json") as f:
        chains = json.load(f)

    correct = np.array([c["trajectory"] for c in chains if c["is_correct"]])
    incorrect = np.array([c["trajectory"] for c in chains if not c["is_correct"]])
    return correct, incorrect


def mean_and_sem(arr: np.ndarray):
    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(max(arr.shape[0], 1))
    return mean, sem


def plot_trajectories(correct: np.ndarray, incorrect: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 1, correct.shape[1] if correct.size else incorrect.shape[1])

    if correct.size:
        mean_c, sem_c = mean_and_sem(correct)
        ax.plot(x, mean_c, label=f"Correct chains (n={correct.shape[0]})", color="#2a9d8f")
        ax.fill_between(x, mean_c - sem_c, mean_c + sem_c, color="#2a9d8f", alpha=0.2)

    if incorrect.size:
        mean_i, sem_i = mean_and_sem(incorrect)
        ax.plot(x, mean_i, label=f"Incorrect chains (n={incorrect.shape[0]})", color="#e76f51")
        ax.fill_between(x, mean_i - sem_i, mean_i + sem_i, color="#e76f51", alpha=0.2)

    ax.set_xlabel("Normalized reasoning progress")
    ax.set_ylabel("Mean token entropy (nats)")
    ax.set_title("Output-distribution variance trajectory: correct vs. incorrect chains")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


def main():
    correct, incorrect = load_trajectories()
    if correct.size == 0 and incorrect.size == 0:
        print("No trajectory data found — run run_experiment.py first.")
        return
    plot_trajectories(correct, incorrect, RESULTS_DIR / "exp1_trajectories.png")


if __name__ == "__main__":
    main()
