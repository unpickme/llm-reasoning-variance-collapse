import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from semantic_divergence import SemanticDivergenceScorer


@pytest.fixture(scope="module")
def scorer():
    # Loaded once for the whole test file, not once per test — loading the
    # embedding model is the slow part, and these tests only need one
    # instance of it.
    return SemanticDivergenceScorer()


def test_identical_continuations_near_zero_distance(scorer):
    continuations = ["the answer is 5", "the answer is 5", "the answer is 5"]
    d = scorer.mean_pairwise_divergence(continuations)
    # Not exactly 0.0: even identical text can pick up tiny floating-point
    # noise through the model, unlike exact-set Jaccard distance.
    assert d < 1e-5


def test_requires_at_least_two(scorer):
    assert math.isnan(scorer.mean_pairwise_divergence(["only one"]))
    assert math.isnan(scorer.mean_pairwise_divergence([]))


def test_paraphrases_score_lower_than_unrelated(scorer):
    # The core motivation for this metric: two continuations that say the
    # same thing in different words should diverge much less than two
    # continuations that say unrelated things — even though Jaccard distance
    # would score both pairs as highly divergent (little token overlap).
    paraphrases = ["the total comes to 12", "that gives us 12 overall"]
    unrelated = ["the total comes to 12", "I enjoy hiking on weekends"]

    d_paraphrase = scorer.mean_pairwise_divergence(paraphrases)
    d_unrelated = scorer.mean_pairwise_divergence(unrelated)

    assert d_paraphrase < d_unrelated


def test_varied_continuations_positive_divergence(scorer):
    continuations = ["the answer is 5", "I think it's 12", "maybe 7 apples"]
    d = scorer.mean_pairwise_divergence(continuations)
    assert d > 0.0
