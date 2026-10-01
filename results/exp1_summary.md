# Experiment 1: Preliminary Findings

*Last updated: 2026-10-01 — pilot run, n=50 chains, model=Qwen/Qwen2.5-0.5B-Instruct*

## Setup used for this run

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: GSM8K test split, first 50 problems
- Variance metric: mean per-step token entropy (see `experiments/exp1_variance_trajectories.md`
  for full methodology)
- Trajectories aligned to 20-point normalized progress axis

## Results

- Correct chains: 17
- Incorrect chains: 33
- Peak location — correct vs incorrect: rank-biserial effect = -0.123, p = 0.479 (Mann-Whitney U)
- Post-peak slope — correct vs incorrect: rank-biserial effect = -0.116, p = 0.510
- Null-percentile check (label shuffle, 1000 iterations): peak location 55.1%, post-peak slope 49.6%

See `results/exp1_trajectories.png` for the mean trajectory plot.

## Interpretation

This pilot found **no evidence** of the hypothesized early-peak-then-collapse signature
distinguishing correct from incorrect reasoning chains. Both p-values are well above
conventional significance thresholds, and critically, the real effect sizes land
almost exactly at the center of their respective label-shuffle null distributions
(55.1st and 49.6th percentile) — essentially indistinguishable from what pure chance
would produce. This is a clean null result, not a weak positive one: the shuffle
control was built specifically to prevent over-reading a marginal p-value as signal,
and here it does its job — there is nothing in this data that clears it.

This does not rule out the underlying hypothesis. It rules out detecting it via
**mean per-step token entropy**, aligned by simple delimiter-based step segmentation,
on a 0.5B-parameter instruction-tuned model, on GSM8K, at n=50. Several of those
choices are candidates for revision before concluding anything more general — see
Next steps.

## Limitations of this pilot

- Small sample size (n=50, 17 correct) — a larger run would tighten confidence
  intervals but is unlikely to flip a result this centered on the null distribution
- Step segmentation uses a simple delimiter heuristic (sentence/line boundaries),
  not a semantically-aware reasoning-step detector
- Single model, single dataset — no evidence yet this generalizes, or fails to
  generalize, across model families or task types
- Token entropy is one proxy for "variance"; activation-space or sampled-continuation
  variance (mentioned in the methodology doc) haven't been tested yet
- A 0.5B model may simply be too small/noisy for any internal-uncertainty signal to
  be cleanly separable from generation noise — the hypothesis may hold better, or
  only, at larger model scales

## Next steps

- [ ] Try the sampled-continuation variance metric (v1 in the methodology doc) instead
      of token entropy — may be more sensitive to real uncertainty than single-sample entropy
- [ ] Repeat at a larger model scale (e.g. a 3-8B instruction-tuned model) to test
      whether the null result is specific to small models
- [ ] Improve step segmentation beyond the delimiter heuristic
- [ ] If results remain null after the above, treat that as a genuine finding worth
      writing up on its own: that this particular signal does not appear to exist in
      the form tested, which is useful negative information for the field