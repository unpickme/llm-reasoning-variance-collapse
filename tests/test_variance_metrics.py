import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from variance_metrics import token_entropy, logit_margin, align_trajectory, split_tokens_into_steps


def test_token_entropy_uniform_is_max():
    vocab_size = 100
    uniform = torch.full((vocab_size,), 1.0 / vocab_size)
    entropy = token_entropy(uniform)
    assert abs(entropy - np.log(vocab_size)) < 1e-4


def test_token_entropy_certain_is_zero():
    vocab_size = 100
    certain = torch.zeros(vocab_size)
    certain[0] = 1.0
    entropy = token_entropy(certain)
    assert entropy < 1e-6


def test_logit_margin_certain_is_one():
    vocab_size = 10
    certain = torch.zeros(vocab_size)
    certain[0] = 1.0
    assert abs(logit_margin(certain) - 1.0) < 1e-6


def test_align_trajectory_length():
    values = [1.0, 2.0, 3.0, 4.0]
    aligned = align_trajectory(values, n_points=20)
    assert len(aligned) == 20
    assert aligned[0] == 1.0
    assert aligned[-1] == 4.0


def test_align_trajectory_empty():
    aligned = align_trajectory([], n_points=10)
    assert np.all(np.isnan(aligned))


def test_split_tokens_into_steps_basic():
    tokens = ["The", " answer", " is", " 5", ".", " Next", " step", "."]
    steps = split_tokens_into_steps(tokens)
    assert len(steps) == 2
    assert steps[0] == [0, 1, 2, 3, 4]
    assert steps[1] == [5, 6, 7]
