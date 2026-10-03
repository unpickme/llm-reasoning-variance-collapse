"""
run_experiment2.py

Experiment 2: sampled-continuation variance.

For each problem:
1. Generate one main reasoning chain (as in Experiment 1) to determine
   correctness and establish the chain's token sequence.
2. At several fixed checkpoint FRACTIONS along that chain (e.g. 10%, 20%,
   ... 100% of the way through), branch off k independently-sampled short
   continuations from that exact point.
3. Measure how much those k continuations diverge from each other
   (continuation_divergence.py) — this is the variance signal at that
   checkpoint.
4. The resulting per-checkpoint divergence values form a trajectory already
   aligned across chains (since checkpoints are defined as fractions, not
   absolute positions), sidestepping Experiment 1's step-segmentation
   heuristic entirely.
5. Run the same peak/collapse signature comparison (detect_signature.py)
   between correct and incorrect chains.

Usage:
    python src/run_experiment2.py --model Qwen/Qwen2.5-0.5B-Instruct --n-samples 30
"""

import argparse
import json
import math
import re
from pathlib import Path

import torch

from generate_chains import ChainGenerator
from continuation_divergence import mean_pairwise_divergence
from detect_signature import extract_signature_features, compare_groups

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

COT_PROMPT_TEMPLATE = (
    "Solve the following problem step by step. Keep your reasoning brief and in "
    "plain text — no LaTeX, no markdown formatting, no section headers. "
    "End your solution with a new line that says exactly: 'Final Answer: <number>'.\n\n"
    "Problem: {question}\n\nSolution:"
)


def extract_final_answer(text: str) -> str:
    match = re.search(r"Final Answer:?\**\s*\$?([\-0-9][\d,\.]*)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip().rstrip(".").replace(",", "")
    numbers = re.findall(r"\$?(-?\d[\d,]*\.?\d*)", text)
    if numbers:
        return numbers[-1].strip().rstrip(".").replace(",", "")
    return ""


def answers_match(predicted: str, gold: str) -> bool:
    """Compare a predicted answer to the gold answer.

    Bug fix: the original code compared these as raw strings, so numerically
    correct answers like "26.00" or "26.0" were marked wrong against a gold
    answer of "26". Here we parse both sides as floats and compare
    numerically when possible, falling back to exact string equality only
    if either side can't be parsed as a number (e.g. extraction failed and
    predicted == "").
    """
    if predicted == gold:
        return True
    try:
        return math.isclose(
            float(predicted.replace(",", "")), float(gold.replace(",", "")),
            rel_tol=1e-9, abs_tol=1e-9,
        )
    except (ValueError, TypeError, AttributeError):
        return False


def load_gsm8k_subset(n_samples: int, start_index: int = 0):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    end_index = min(start_index + n_samples, len(ds))
    ds = ds.select(range(start_index, end_index))
    examples = []
    for i, row in zip(range(start_index, end_index), ds):
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        examples.append({"id": str(i), "question": row["question"], "gold_answer": gold})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--start-index", type=int, default=0,
                         help="Dataset index to start from — use this to run non-overlapping batches.")
    parser.add_argument("--n-checkpoints", type=int, default=8)
    parser.add_argument("--k-continuations", type=int, default=5)
    parser.add_argument("--continuation-length", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading {args.n_samples} examples from gsm8k, starting at index {args.start_index}...")
    examples = load_gsm8k_subset(args.n_samples, start_index=args.start_index)
    batch_tag = f"_batch{args.start_index}-{args.start_index + len(examples) - 1}"

    print(f"Loading model {args.model}...")
    generator = ChainGenerator(model_name=args.model)

    correct_features, incorrect_features = [], []
    all_chains_summary = []

    for i, ex in enumerate(examples):
        print(f"Problem {i + 1}/{len(examples)} (id={ex['id']})...", flush=True)
        prompt = COT_PROMPT_TEMPLATE.format(question=ex["question"])

        # Step 1: main chain, same as Experiment 1
        main_chain = generator.generate_chain(prompt, problem_id=ex["id"], max_new_tokens=args.max_new_tokens)
        final_answer = extract_final_answer(main_chain.generated_text)
        is_correct = answers_match(final_answer, ex["gold_answer"])
        print(f"  -> main answer={final_answer!r} gold={ex['gold_answer']!r} correct={is_correct}", flush=True)

        n_steps = len(main_chain.steps)
        if n_steps < args.n_checkpoints:
            print(f"  -> chain too short ({n_steps} tokens) for {args.n_checkpoints} checkpoints, skipping")
            continue

        # Reconstruct the full prompt token ids (needed to build checkpoint prefixes)
        formatted_prompt = generator._format_prompt(prompt)
        prompt_ids = generator.tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(generator.device)
        full_token_ids = [s.token_id for s in main_chain.steps]

        # Step 2 + 3: branch at each checkpoint fraction, measure divergence
        divergence_trajectory = []
        for cp in range(1, args.n_checkpoints + 1):
            fraction = cp / args.n_checkpoints
            idx = max(1, int(fraction * n_steps))
            prefix_ids = torch.cat(
                [prompt_ids, torch.tensor([full_token_ids[:idx]], device=generator.device)], dim=-1
            )

            continuations = [
                generator.generate_text_continuation(
                    prefix_ids, max_new_tokens=args.continuation_length
                )
                for _ in range(args.k_continuations)
            ]
            divergence = mean_pairwise_divergence(continuations)
            divergence_trajectory.append(divergence)

        # Step 4: features from this already-aligned trajectory (fixed length = n_checkpoints)
        import numpy as np
        trajectory_arr = np.array(divergence_trajectory)
        features = extract_signature_features(trajectory_arr)
        (correct_features if is_correct else incorrect_features).append(features)

        all_chains_summary.append({
            "problem_id": ex["id"],
            "is_correct": is_correct,
            "final_answer": final_answer,
            "gold_answer": ex["gold_answer"],
            "generated_text": main_chain.generated_text,
            "divergence_trajectory": divergence_trajectory,
        })

    print(f"Correct chains: {len(correct_features)} | Incorrect chains: {len(incorrect_features)}")

    if len(correct_features) >= 5 and len(incorrect_features) >= 5:
        comparison = compare_groups(correct_features, incorrect_features)
    else:
        comparison = {"warning": "Not enough chains in one group for a meaningful test."}

    raw_path = RESULTS_DIR / f"exp2_raw_chains{batch_tag}.json"
    comparison_path = RESULTS_DIR / f"exp2_comparison{batch_tag}.json"

    with open(raw_path, "w") as f:
        json.dump(all_chains_summary, f, indent=2)

    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Saved results to {raw_path} and {comparison_path}")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
