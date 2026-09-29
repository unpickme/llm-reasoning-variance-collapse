"""
variance_metrics.py

Compute per-step uncertainty metrics from token distributions, aggregate
tokens into reasoning "steps", and align variable-length chains onto a
common normalized progress axis for cross-chain comparison.
"""

from typing import List

import numpy as np
import torch


def token_entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    """Shannon entropy (in nats) of a softmax distribution over the vocab."""
    p = probs.clamp_min(eps)
    return float(-(p * p.log()).sum().item())


def logit_margin(probs: torch.Tensor) -> float:
    """Top-1 minus top-2 probability. Low margin = high local uncertainty."""
    top2 = torch.topk(probs, k=2).values
    return float((top2[0] - top2[1]).item())


def step_level_metrics(step_probs: List[torch.Tensor]) -> dict:
    """
    Given the list of per-token distributions for ONE reasoning step
    (a step may span several tokens, e.g. a sentence or an equation),
    return aggregated metrics for that step.
    """
    entropies = [token_entropy(p) for p in step_probs]
    margins = [logit_margin(p) for p in step_probs]
    return {
        "mean_entropy": float(np.mean(entropies)),
        "max_entropy": float(np.max(entropies)),
        "mean_margin": float(np.mean(margins)),
        "min_margin": float(np.min(margins)),
    }


def split_tokens_into_steps(token_strs: List[str], step_delimiters=(".", "\n", "=")) -> List[List[int]]:
    """
    Naive step segmentation: split the token stream into steps at delimiter
    tokens. Returns a list of index-lists, each giving the token indices
    belonging to one step. This is a simple heuristic for v0 — reasoning
    steps in chain-of-thought text are often sentence- or line-delimited.
    Replace with a more principled segmenter (e.g. one that recognizes
    explicit numbered steps) if the delimiter heuristic proves noisy.
    """
    steps = []
    current = []
    for i, tok in enumerate(token_strs):
        current.append(i)
        if any(d in tok for d in step_delimiters):
            steps.append(current)
            current = []
    if current:
        steps.append(current)
    return steps


def align_trajectory(step_values: List[float], n_points: int = 20) -> np.ndarray:
    """
    Interpolate a variable-length per-step metric sequence onto a fixed
    grid of n_points over normalized progress [0, 1], so trajectories from
    chains of different lengths can be averaged/compared directly.
    """
    if len(step_values) == 0:
        return np.full(n_points, np.nan)
    if len(step_values) == 1:
        return np.full(n_points, step_values[0])

    x_original = np.linspace(0, 1, len(step_values))
    x_target = np.linspace(0, 1, n_points)
    return np.interp(x_target, x_original, step_values)
