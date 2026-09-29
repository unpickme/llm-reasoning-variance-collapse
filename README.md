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

## Repository structure

```
llm-reasoning-variance-collapse/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── generate_chains.py     # generate reasoning chains + capture token distributions
│   ├── variance_metrics.py    # entropy / logit-margin / trajectory alignment
│   ├── detect_signature.py    # peak/collapse detection + statistical tests
│   └── run_experiment.py      # orchestrates end-to-end run
├── experiments/
│   └── exp1_variance_trajectories.md
├── tests/
│   └── test_variance_metrics.py
├── data/          # cached datasets / generated chains (gitignored, see below)
└── results/       # plots, tables, run logs (gitignored except summaries)
```

## Status

🚧 Early stage — hypothesis under active testing. Results, positive or negative,
will be posted here as they come in rather than only if they confirm the hypothesis.

## Running it

```bash
pip install -r requirements.txt
python src/run_experiment.py --model gpt2 --n-samples 200 --dataset gsm8k
```

See `experiments/exp1_variance_trajectories.md` for parameters and expected runtime.

## Relationship to other work

This project sits alongside a broader interest of mine in coordination and cascade
dynamics in multi-agent systems generally — the LLM case here is treated as its own
self-contained empirical question, with its own methodology and evaluation, rather
than an application of any prior unpublished framework.

## License

MIT — see `LICENSE`.
