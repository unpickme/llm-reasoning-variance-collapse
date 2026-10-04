import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from continuation_divergence import jaccard_distance, mean_pairwise_divergence


def test_jaccard_identical_texts_zero_distance():
    assert jaccard_distance("the cat sat", "the cat sat") == 0.0


def test_jaccard_disjoint_texts_max_distance():
    assert jaccard_distance("apple banana", "car truck") == 1.0


def test_jaccard_partial_overlap():
    d = jaccard_distance("the cat sat on the mat", "the cat sat on the rug")
    assert 0.0 < d < 1.0


def test_mean_pairwise_divergence_identical_continuations():
    continuations = ["the answer is 5", "the answer is 5", "the answer is 5"]
    assert mean_pairwise_divergence(continuations) == 0.0


def test_mean_pairwise_divergence_requires_at_least_two():
    import math
    assert math.isnan(mean_pairwise_divergence(["only one"]))


def test_mean_pairwise_divergence_varied_continuations():
    continuations = ["the answer is 5", "I think it's 12", "maybe 7 apples"]
    d = mean_pairwise_divergence(continuations)
    assert d > 0.0
