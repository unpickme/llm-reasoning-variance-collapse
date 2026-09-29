"""
detect_signature.py

Detect the early-peak-then-collapse signature in aligned variance
trajectories, and test whether it differs significantly between
correct and incorrect reasoning chains.

Includes a label-shuffle control so an apparent effect can be checked
against a null distribution before being reported as real.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import stats


@dataclass
class SignatureFeatures:
    peak_location: float   # normalized progress [0,1] where trajectory peaks
    peak_height: float
    post_peak_slope: float  # mean slope from peak to end


def extract_signature_features(trajectory: np.ndarray) -> SignatureFeatures:
    """Extract peak location, height, and post-peak slope from one aligned trajectory."""
    valid = ~np.isnan(trajectory)
    if valid.sum() < 3:
        return SignatureFeatures(np.nan, np.nan, np.nan)

    idx = np.argmax(trajectory)
    n = len(trajectory)
    peak_location = idx / (n - 1)
    peak_height = trajectory[idx]

    if idx < n - 1:
        post_peak = trajectory[idx:]
        x = np.arange(len(post_peak))
        slope, _, _, _, _ = stats.linregress(x, post_peak)
    else:
        slope = 0.0

    return SignatureFeatures(peak_location, peak_height, slope)


def compare_groups(
    correct_features: List[SignatureFeatures],
    incorrect_features: List[SignatureFeatures],
    n_shuffles: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Compare peak_location and post_peak_slope between correct and incorrect
    chains using Mann-Whitney U, with a label-shuffle null distribution as
    a guard against spurious significance.
    """
    rng = np.random.default_rng(seed)

    def _values(features: List[SignatureFeatures], attr: str) -> np.ndarray:
        arr = np.array([getattr(f, attr) for f in features])
        return arr[~np.isnan(arr)]

    results = {}
    for attr in ("peak_location", "post_peak_slope"):
        correct_vals = _values(correct_features, attr)
        incorrect_vals = _values(incorrect_features, attr)

        u_stat, p_value = stats.mannwhitneyu(correct_vals, incorrect_vals, alternative="two-sided")
        n1, n2 = len(correct_vals), len(incorrect_vals)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

        # Null distribution via label shuffling
        combined = np.concatenate([correct_vals, incorrect_vals])
        null_effects = []
        for _ in range(n_shuffles):
            shuffled = rng.permutation(combined)
            g1, g2 = shuffled[:n1], shuffled[n1:]
            u, _ = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            null_effects.append(1 - (2 * u) / (n1 * n2))
        null_effects = np.array(null_effects)

        percentile = float((np.abs(null_effects) < abs(rank_biserial)).mean() * 100)

        results[attr] = {
            "p_value": float(p_value),
            "rank_biserial_effect": float(rank_biserial),
            "null_percentile": percentile,  # how extreme the real effect is vs shuffled null
            "n_correct": n1,
            "n_incorrect": n2,
        }

    return results
