## Setup used for this run

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Dataset: GSM8K test split, problems 0-49
- Variance metric: mean pairwise Jaccard divergence between 5 sampled continuations
  at each of 8 checkpoint fractions along each chain (see
  `experiments/exp2_sampled_continuation_variance.md`)
- Run in 5 non-overlapping batches of 10 for practical reasons (session length);
  merged via `src/merge_exp2_batches.py` before analysis — equivalent to a single
  n=50 run, see that script for the duplicate/overlap check used to verify this.

## Results

**Update (3 October 2026):** The answer-matching bug described below has been
fixed in `run_experiment2.py` (and `run_experiment.py`) — correctness is now
checked by parsing both the extracted and gold answers as floats and comparing
numerically, rather than via exact string equality. Experiment 2 is being
re-run (n=50) with the fix in place; the results below predate that fix and
should be treated as provisional until the re-run completes.
- Correct chains: 18
- Incorrect chains: 32
- Peak location — correct vs incorrect: rank-biserial effect = -0.210, p = 0.083,
  null-percentile = 89.2% (does not clear the 95% bar)
- Post-peak slope — correct vs incorrect: rank-biserial effect = -0.321, p = 0.0078,
  **null-percentile = 99.0%** (clears the 95% bar)

See `results/exp2_trajectories.png` for the mean trajectory plot.
**Update (3 October 2026):** The answer-matching bug described above has been
fixed — correctness is now checked by parsing both the extracted and gold
answers as floats and comparing numerically, rather than via exact string
equality (`run_experiment.py` and `run_experiment2.py`). Experiment 2 was
re-run in full (n=50) with the fix in place.

**Result: the apparent signal did not survive the fix.**

| Metric | Before fix (buggy split) | After fix (correct split) |
|---|---|---|
| Post-peak slope | p=0.0078, 99th percentile of shuffle-null | p=0.492, 38.7th percentile of shuffle-null |
| Peak location | p=0.083, 89.2nd percentile of shuffle-null | p=0.651, 36.5th percentile of shuffle-null |

With the correctness labels fixed (18 correct / 32 incorrect, vs. a different
split under the bug), both metrics now sit comfortably within the shuffle-null
distribution. The original post-peak-slope result was an artifact of the
contaminated correct/incorrect split, not a real effect. **Experiment 2 is
now a clean null, consistent with Experiment 1.**

This is being recorded as a null result rather than reframed or minimized:
the bug produced a false-positive-looking signal, and fixing it made that
signal disappear. See "Next steps" below for where the investigation goes
from here (semantic-embedding divergence, larger model scale).

## Interpretation

Unlike Experiment 1's clean null result, this pilot shows a **partial signal**:
post-peak slope differs between correct and incorrect chains in a way that clears
our pre-registered 95th-percentile shuffle-null bar, while peak location does not.
This is more interesting than a flat null, but should not yet be treated as a
confirmed finding — see the known data-quality issue below, which affects the
correct/incorrect labels this result depends on.

## Known issue affecting this result (important)

**Answer-matching bug:** correctness is currently checked via exact string equality
between the extracted answer and the gold answer (e.g. `"26.00" == "26"` → `False`,
even though these are numerically identical). This was noticed after this run
completed. Some chains labeled "incorrect" in this dataset may actually have
produced the correct numeric answer in a different string format. Since
correct/incorrect is the entire basis of this comparison, this is a real
confound, not a cosmetic issue.

**Before treating the post-peak-slope result as a real finding, this should be
fixed (compare as parsed floats, not strings) and the experiment re-run.** The
current result should be read as "worth following up," not "confirmed."

## Limitations of this pilot

- Answer-matching bug described above — the most important limitation
- Small sample size (n=50, 18 correct) — same caveat as Experiment 1
- Jaccard token-overlap divergence is a coarse, surface-level proxy for semantic
  divergence (see `src/continuation_divergence.py` docstring)
- Single model, single dataset
- Checkpoint continuations are freshly sampled branches, not a continuation of the
  main chain's own sampling trajectory — see design doc for why this is intentional
  but still a simplification

## Next steps

- [ ] Fix the answer-matching bug (parse both sides as floats before comparing)
      and re-run before drawing firm conclusions
- [ ] If the post-peak-slope effect survives the bug fix, investigate why slope
      specifically (not peak location) carries the signal
- [ ] Try a semantic-embedding divergence metric instead of Jaccard distance
- [ ] Repeat at larger model scale

