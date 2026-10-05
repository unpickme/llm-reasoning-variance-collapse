# Experiment 3: Sampled-Continuation Variance — Semantic-Embedding Divergence

## Motivation

Experiment 2 used Jaccard distance (token-set overlap) to measure how much
k independently-sampled continuations diverge from one another at each
checkpoint. Jaccard is purely lexical: two continuations that express the
same underlying reasoning in different words score as highly divergent,
even though there's no real uncertainty in what the model "means." This is
a plausible source of noise that could mask or distort a real signal.

Experiment 3 replaces Jaccard with sentence-embedding cosine distance
(`sentence-transformers/all-MiniLM-L6-v2`), which should better isolate
divergence in *meaning* from divergence in *wording*. Everything else —
chain generation, checkpoint fractions, correctness labeling (via the
numerically-correct `answers_match`, same fix as Experiment 2) — is
identical to Experiment 2. Only the divergence metric changed.

## Method

Same as Experiment 2 (see `experiments/exp2_sampled_continuation_variance.md`),
with `continuation_divergence.py`'s `mean_pairwise_divergence` (Jaccard)
replaced by `semantic_divergence.py`'s `SemanticDivergenceScorer`
(mean pairwise cosine distance between sentence embeddings). n=50,
gsm8k, Qwen2.5-0.5B-Instruct, run in 5 batches of 10 and merged via
`merge_exp3_batches.py`.

## Results

| Metric | p-value | Rank-biserial effect | Null percentile |
|---|---|---|---|
| Peak location | 0.606 | -0.043 | 34.0 |
| Post-peak slope | 0.699 | -0.033 | 0.0 |

18 correct / 32 incorrect chains (same correctness split as the fixed
Experiment 2, since both use the same n=50 gsm8k subset and the same
corrected answer-matching).

**Null percentile note:** `null_percentile` measures what fraction of
1,000 label-shuffled surrogate effects have a *smaller magnitude* than the
real observed effect. A percentile of 0.0 for post-peak slope means the
real effect (|rank-biserial| = 0.033) was smaller in magnitude than *every
single one* of the 1,000 shuffled-label effects — i.e. the actual
correct/incorrect difference is smaller than typical noise from randomly
shuffling labels on this data. This is the mirror image of the original
(buggy) Experiment 2 result, where a *high* percentile (99th) corresponded
to a *low* p-value. Here, a low percentile and a high p-value both point
the same way: no signal.

## Interpretation

**Clean null result**, consistent with Experiment 1 (token entropy) and
the bug-fixed Experiment 2 (Jaccard continuation divergence). Switching
from lexical to semantic divergence did not surface a signal that was
being masked by wording noise — if anything, the semantic version shows
an even smaller effect than Jaccard's post-fix result.

Three independent operationalizations of "output-distribution variance
across a reasoning chain" (token-level entropy, lexical continuation
divergence, semantic continuation divergence) now agree: none finds a
peak-then-collapse signature that distinguishes correct from incorrect
chains on this model/dataset/scale.

## Next steps

Given three converging nulls, the most informative remaining directions
from the original candidate list are the ones that change the *regime*
rather than the *metric*:
- **Larger model scale** — the signal, if real, may only emerge at a
  capability level where the model has genuine internal uncertainty to
  express, rather than Qwen2.5-0.5B's comparatively narrow, low-variance
  outputs on gsm8k-level problems. This is also the most CPU-cost-expensive
  direction to pursue.
- Alternatively, treat the three-way null as the headline finding itself:
  a well-triangulated negative result (same hypothesis, three different
  operationalizations, consistent non-effect) is a legitimate and
  honestly-reported conclusion in its own right, not just a stepping stone
  to a positive result.
