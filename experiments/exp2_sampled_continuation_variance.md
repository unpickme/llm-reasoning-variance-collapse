# Experiment 2: Sampled-Continuation Variance

## Motivation

Experiment 1 found no evidence of the hypothesized signature using mean
per-step token entropy (see `results/exp1_summary.md`). One candidate
explanation: single-sample token entropy only captures uncertainty about
the *immediate next token*, which may be a noisy or shallow proxy for the
model's deeper uncertainty about how the reasoning will unfold. This
experiment tests an alternative operationalization: **how much do
independent continuations from the same point actually diverge**, rather
than how uncertain the model looks one token at a time.

This also removes a methodological weakness from Experiment 1: instead of
relying on a delimiter-based heuristic to segment reasoning into "steps,"
checkpoints here are defined directly as fixed fractions of each chain's
own length, so trajectories are aligned across chains by construction.

## Method

1. Generate one full reasoning chain per problem, as in Experiment 1
   (same model, same prompt, same correctness check against GSM8K gold answers).
2. At `n_checkpoints` fixed fractions along that chain (default: 8, i.e.
   12.5%, 25%, ..., 100%), take the token prefix up to that point.
3. From each checkpoint prefix, independently sample `k` short continuations
   (default: 5 continuations, 20 tokens each).
4. Measure the mean pairwise divergence among those k continuations using
   Jaccard distance over token sets (see `src/continuation_divergence.py`
   for why this metric was chosen over a semantic-embedding approach for v0).
5. This produces one divergence trajectory per chain, directly comparable
   across chains since checkpoints are fraction-based.
6. Run the same peak-location / post-peak-slope comparison between correct
   and incorrect chains as Experiment 1, including the label-shuffle null
   control (`src/detect_signature.py` — reused unchanged from Experiment 1).

## Compute cost vs. Experiment 1

This experiment is notably more expensive per problem: each problem now
requires 1 main generation + (`n_checkpoints` × `k` × `continuation_length`)
additional tokens of generation. With defaults (8 checkpoints × 5
continuations × 20 tokens), that's up to 800 extra generated tokens per
problem on top of the main chain. Start with a small `--n-samples` (5-10)
to sanity check before scaling up.

## Known limitations going in

- Jaccard token-overlap is a surface-level divergence measure — it can
  under- or over-count true semantic divergence (see module docstring in
  `continuation_divergence.py`)
- Checkpoint-prefix generation does not carry over the main chain's
  temperature/sampling trajectory exactly — continuations are freshly
  sampled from that point, which is intentional (that's the point of the
  branching) but means the main chain and the checkpoint continuations are
  not the "same" generation process end to end
- Still a single model, single dataset — same generalization caveat as
  Experiment 1

## Running it

```bash
python src/run_experiment2.py --model Qwen/Qwen2.5-0.5B-Instruct --n-samples 10
python src/plot_results2.py
```

Results are saved to `results/exp2_raw_chains.json` and
`results/exp2_comparison.json`. Findings go in `results/exp2_summary.md`
(template to be added once a real run is complete, following the same
format as `results/exp1_summary.md`).
