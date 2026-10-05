# LLM Reasoning Variance Collapse

**Research question:** Can the *shape* of a language model's per-step output-distribution
variance across a multi-step reasoning chain serve as an early warning signal for
reasoning failure — detectable before the failure is visible in the final answer?

## Motivation

Most work on LLM reasoning reliability looks at the *final* output: is the answer
correct, is the model calibrated, does it hallucinate. Far less work looks at the
*trajectory* of the model's internal uncertainty as it reasons step by step.

This project investigates a specific structural hypothesis: that failing reasoning
chains show a characteristic **early peak in output-distribution variance, followed
by a collapse**, before the chain arrives at an incorrect final answer — while
successful chains do not show this pattern. If real, this would give a
purely-internal, label-free signal that a reasoning chain is heading toward failure,
usable for early intervention (e.g. triggering re-sampling, a verifier pass, or
human review) in agentic and multi-step LLM systems.

This is exploratory empirical research, not a proof. The goal of this repository is
to test the hypothesis honestly, report what the data shows (including null or mixed
results), and build reusable tooling for others to test it on their own models and
tasks.

## Methodology: how "variance" is operationalized

"Output-distribution variance" is deliberately underspecified in casual conversation
about LLM uncertainty, so this project is explicit about which operationalization is
used at each stage:

- **v0 (current, implemented):** *token-level entropy* — the Shannon entropy of the
  model's softmax distribution over the vocabulary at each generated token, averaged
  within each reasoning step. This is the cheapest and most directly comparable
  signal across models, and is the metric used in the results below. A companion
  metric, *logit margin* (top-1 vs top-2 probability gap), is also computed per step
  and available in the data, though not yet the primary reported signal.
- **Planned v1:** *sampled-continuation variance* — generate k completions from the
  same partial reasoning chain and measure how much they diverge (e.g. embedding
  distance between completions), which captures uncertainty the single-sample
  entropy metric can miss.
- **Planned v2:** *activation-space variance* — variance in hidden-state
  representations across steps, which may pick up structural instability that isn't
  visible at the output-token level at all.

Each reasoning chain's per-step metric is aligned onto a normalized progress axis
(0 → 1) so that chains of different lengths can be averaged and compared directly —
see `src/variance_metrics.py::align_trajectory`.

## Method (v0)

1. Run a small open-source LLM on a reasoning benchmark (GSM8K, arithmetic subset),
   generating step-by-step chain-of-thought completions.
2. At each generation step, record the token-level output distribution and compute
   two variance proxies: Shannon entropy and logit-margin (top-1 vs top-2 probability
   gap).
3. Align each chain's per-step trajectory to a normalized "reasoning progress" axis
   (0 → 1) so chains of different lengths are comparable.
4. Split chains by outcome: correct final answer vs. incorrect final answer.
5. Test for a statistically distinguishable early-peak-then-collapse signature
   between the two groups (peak location, peak height, post-peak slope).
6. Report effect sizes, significance, and — critically — negative controls
   (e.g. shuffled labels) to guard against spurious pattern-fitting.

See `experiments/exp1_variance_trajectories.md` for the full experimental design and
`src/` for the implementation.

**Experiment 1 result (n=50, token entropy):** null result — no evidence this
signature exists in mean per-step token entropy at this scale. See
`results/exp1_summary.md` for full details and the label-shuffle control that
backs this up.

**Experiment 2 (in progress):** tests an alternative operationalization —
divergence between independently-sampled continuations branched from fixed
checkpoints along each chain, rather than single-sample token entropy. See
`experiments/exp2_sampled_continuation_variance.md`.

## Repository structure

```
llm-reasoning-variance-collapse/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── generate_chains.py        # generate reasoning chains + capture token distributions
│   ├── variance_metrics.py       # entropy / logit-margin / trajectory alignment
│   ├── continuation_divergence.py # exp2: divergence between sampled continuations
│   ├── detect_signature.py       # peak/collapse detection + statistical tests
│   ├── run_experiment.py         # exp1 orchestration
│   ├── run_experiment2.py        # exp2 orchestration
│   ├── plot_results.py           # exp1 plotting
│   └── plot_results2.py          # exp2 plotting
├── experiments/
│   ├── exp1_variance_trajectories.md
│   └── exp2_sampled_continuation_variance.md
├── tests/
│   ├── test_variance_metrics.py
│   └── test_continuation_divergence.py
├── data/          # cached datasets / generated chains (gitignored, see below)
└── results/       # plots, tables, run logs (gitignored except summaries)
```

## Status

Three experiments conducted so far, all null results.

- **Experiment 1** (token entropy, n=50): peak-location and post-peak-slope
  effects sit at ~50th percentile of the shuffle-null distribution — no signal.
- **Experiment 2** (sampled-continuation Jaccard divergence, n=50): an
  apparent post-peak-slope signal (p=0.0078, 99th percentile) was traced to
  an answer-matching bug contaminating the correct/incorrect split. After
  fixing the bug and re-running, the effect disappeared (p=0.49, 38.7th
  percentile). See `results/exp2_summary.md`.
- **Experiment 3** (sampled-continuation semantic-embedding divergence,
  n=50): replacing Jaccard with sentence-embedding cosine distance, to rule
  out lexical-wording noise as a confound, still finds no signal (p=0.61 and
  p=0.70). See `results/exp3_summary.md`.

The hypothesis — that output-distribution variance shows an early-peak-then-
collapse signature predicting reasoning failure — is not supported across
three independent operationalizations (token entropy, lexical continuation
divergence, semantic continuation divergence) on this model/dataset/scale.
Next directions are logged in `results/exp3_summary.md`.

## Running it

```bash
pip install -r requirements.txt
**Status:** Both experiments conducted so far are null results.

- **Experiment 1** (token entropy, n=50): peak-location and post-peak-slope
  effects sit at ~50th percentile of the shuffle-null distribution — no signal.
- **Experiment 2** (sampled-continuation divergence, n=50): an apparent
  post-peak-slope signal (p=0.0078, 99th percentile) was traced to an
  answer-matching bug contaminating the correct/incorrect split. After fixing
  the bug and re-running, the effect disappeared (p=0.49, 38.7th percentile).
  See `results/exp2_summary.md` for the full before/after comparison.

The hypothesis — that output-distribution variance shows an early-peak-then-
collapse signature predicting reasoning failure — is not supported by either
experiment conducted so far. Next directions are logged in
`results/exp2_summary.md`.python src/run_experiment.py --model gpt2 --n-samples 200 --dataset gsm8k
```

See `experiments/exp1_variance_trajectories.md` for parameters and expected runtime.

## Relationship to other work

This project sits alongside a broader interest of mine in coordination and cascade
dynamics in multi-agent systems generally — the LLM case here is treated as its own
self-contained empirical question, with its own methodology and evaluation, rather
than an application of any prior unpublished framework.

## License

MIT — see `LICENSE`.
