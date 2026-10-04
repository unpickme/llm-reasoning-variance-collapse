"""
continuation_divergence.py

Measures how much k independently-sampled continuations from the same
partial reasoning chain diverge from one another, as a proxy for the
model's uncertainty at that point in its reasoning.

v0 metric: token-overlap (Jaccard) distance. This is a coarse, surface-level
proxy — it will treat two continuations with the same meaning but different
wording as divergent, and in principle could miss real semantic divergence
that happens to share vocabulary. It was chosen for v0 because it requires
no additional model download and runs fast on CPU. A sentence-embedding-based
divergence metric (cosine distance between pooled embeddings) is a natural
v1 upgrade — see experiments/exp2_sampled_continuation_variance.md.
"""

from itertools import combinations
from typing import List


def _tokenize_simple(text: str) -> set:
    """Lowercase whitespace tokenization. Simple on purpose — see module docstring."""
    return set(text.lower().split())


def jaccard_distance(text_a: str, text_b: str) -> float:
    """1 - Jaccard similarity between two texts' token sets. 0 = identical token sets, 1 = disjoint."""
    set_a, set_b = _tokenize_simple(text_a), _tokenize_simple(text_b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    intersection = set_a & set_b
    return 1.0 - (len(intersection) / len(union))


def mean_pairwise_divergence(continuations: List[str]) -> float:
    """
    Mean Jaccard distance across all pairs of k continuations. Higher means
    the continuations disagree more — interpreted here as higher local
    uncertainty at the checkpoint they were branched from.
    """
    if len(continuations) < 2:
        return float("nan")
    pairs = list(combinations(continuations, 2))
    distances = [jaccard_distance(a, b) for a, b in pairs]
    return sum(distances) / len(distances)
