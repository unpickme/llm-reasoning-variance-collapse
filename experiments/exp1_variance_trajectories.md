# Experiment 1: Variance Trajectories in Correct vs. Incorrect Reasoning Chains

## Goal

Test whether incorrect reasoning chains show a distinguishable early-peak-then-collapse
pattern in output-distribution variance, relative to correct chains.

## Setup

- **Model:** GPT-2 (124M) or Pythia-410M to start — small enough to run on CPU/single
  GPU, well-documented, fully open. Swappable via `--model` flag; larger models
  (Llama-3-8B-Instruct etc.) planned as a follow-up once the pipeline is validated.
- **Dataset:** GSM8K (grade-school math word problems) — chosen because each problem
  has a verifiable final numeric answer, and solutions are naturally multi-step.
- **Chain generation:** Standard chain-of-thought prompting, greedy or low-temperature
  sampling, one reasoning chain per problem.
- **Sample size:** n=200 problems for the pilot run (enough for an initial significance
  check; scale up if the pilot shows a signal worth chasing).

## Metrics computed per generation step

1. **Token entropy:** Shannon entropy of the softmax distribution over the vocabulary
   at each generated token position.
2. **Logit margin:** difference between the top-1 and top-2 token probabilities
   (a low margin = high local uncertainty).
3. **Step-level aggregation:** since a "reasoning step" spans multiple tokens, we
   aggregate token-level metrics within each step (mean and max) to get one variance
   value per step.

## Trajectory alignment

Chains vary in number of steps. We normalize each chain's step index to [0, 1]
(progress through the chain) and interpolate the variance metric onto a fixed grid
(e.g. 20 points) so trajectories are comparable across chains of different lengths.

## Hypothesis test

- **H1 (signal exists):** incorrect chains show a peak in aggregated variance earlier
  in the normalized trajectory, followed by a steeper decline, compared to correct
  chains.
- **H0 (no signal):** peak location, peak height, and post-peak slope are not
  significantly different between correct and incorrect chains.

Planned test: compare peak-location and post-peak-slope distributions between groups
with a Mann-Whitney U test (non-parametric, doesn't assume normal trajectories).
Report effect size (rank-biserial correlation), not just p-value.

## Controls against false positives

- **Label shuffle control:** randomly reassign correct/incorrect labels and re-run
  the same test 1000x to build a null distribution. If the real split's effect size
  doesn't clear the shuffled null distribution's 95th percentile, treat the result as
  not significant regardless of the raw p-value.
- **Length control:** check whether the effect (if any) is actually just a proxy for
  chain length (longer chains might be incorrect more often *and* show different
  variance shapes for unrelated reasons). Report the effect after controlling for
  chain length.

## Deliverables

- `results/exp1_summary.md` — written summary of findings (updated after each run,
  including negative/null results)
- `results/exp1_trajectories.png` — mean trajectory plot, correct vs incorrect, with
  confidence bands
- Raw per-chain data cached in `data/` for reproducibility (not committed — see
  `.gitignore`)
