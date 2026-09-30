# Experiment 1: Preliminary Findings

*Last updated: [DATE] — pilot run, n=[N] chains, model=[MODEL NAME]*

## Setup used for this run

- Model: [e.g. Qwen2.5-0.5B-Instruct]
- Dataset: GSM8K test split, first [N] problems
- Variance metric: mean per-step token entropy (see `experiments/exp1_variance_trajectories.md`
  for full methodology)
- Trajectories aligned to 20-point normalized progress axis

## Results

- Correct chains: [N_CORRECT]
- Incorrect chains: [N_INCORRECT]
- Peak location — correct vs incorrect: [VALUES], p=[P_VALUE] (Mann-Whitney U)
- Post-peak slope — correct vs incorrect: [VALUES], p=[P_VALUE]
- Null-percentile check (label shuffle, 1000 iterations): [PERCENTILE]%

See `results/exp1_trajectories.png` for the mean trajectory plot.

## Interpretation

[Write 2-4 sentences: does the data support, partially support, or fail to support
the early-peak-then-collapse hypothesis? Be explicit about effect size and whether
it clears the shuffle-null control — a p-value alone is not evidence here.]

Example framing if the result is preliminary/mixed (edit to match your actual data):
> Preliminary results on a small pilot (n=[N]) suggest [a modest / no clear / a
> promising] difference in [peak location / post-peak slope] between correct and
> incorrect chains. The effect [does / does not] clear the label-shuffle null at the
> 95th percentile, so this should be treated as [suggestive but inconclusive /
> not yet distinguishable from noise]. A larger sample and a stronger model are
> needed before drawing firm conclusions.

## Limitations of this pilot

- Small sample size — effect sizes at this n are noisy; treat as directional only
- Step segmentation uses a simple delimiter heuristic (sentence/line boundaries),
  not a semantically-aware reasoning-step detector
- Single model, single dataset — no evidence yet this generalizes across model
  families or task types
- Token entropy is one proxy for "variance"; activation-space or sampled-continuation
  variance (mentioned in the methodology doc) haven't been tested yet

## Next steps

- [ ] Scale to full n=200 pilot if not already done
- [ ] Try a second model family to check generalization
- [ ] Try a second task (e.g. multi-step tool-use agent trace, not just math word problems)
- [ ] Implement a better step-segmentation method
