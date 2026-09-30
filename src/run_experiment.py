"""
run_experiment.py

End-to-end pilot: load a GSM8K subset, generate reasoning chains, compute
variance trajectories, split by correctness, and run the signature
comparison. Saves results to results/.

Usage:
    python src/run_experiment.py --model gpt2 --n-samples 200
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from generate_chains import ChainGenerator
from variance_metrics import split_tokens_into_steps, step_level_metrics, align_trajectory
from detect_signature import extract_signature_features, compare_groups

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COT_PROMPT_TEMPLATE = (
    "Solve the following problem step by step, showing your reasoning, "
    "and end with 'Final Answer: <number>'.\n\nProblem: {question}\n\nSolution:"
)


def extract_final_answer(text: str) -> str:
    match = re.search(r"Final Answer:\s*([\-0-9\.,]+)", text)
    return match.group(1).strip().rstrip(".") if match else ""


def load_gsm8k_subset(n_samples: int):
    """
    Loads a subset of GSM8K via the `datasets` library. Requires network
    access at run time (not included in this container by default).
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.select(range(min(n_samples, len(ds))))
    examples = []
    for i, row in enumerate(ds):
        gold = row["answer"].split("####")[-1].strip()
        examples.append({"id": str(i), "question": row["question"], "gold_answer": gold})
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--dataset", default="gsm8k")
    parser.add_argument("--n-grid-points", type=int, default=20)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Loading {args.n_samples} examples from {args.dataset}...")
    examples = load_gsm8k_subset(args.n_samples)

    print(f"Loading model {args.model}...")
    generator = ChainGenerator(model_name=args.model)

    correct_features, incorrect_features = [], []
    all_chains_summary = []

    for ex in examples:
        prompt = COT_PROMPT_TEMPLATE.format(question=ex["question"])
        chain = generator.generate_chain(prompt, problem_id=ex["id"])

        chain.final_answer = extract_final_answer(chain.generated_text)
        chain.is_correct = chain.final_answer == ex["gold_answer"]

        token_strs = [s.token_str for s in chain.steps]
        step_groups = split_tokens_into_steps(token_strs)

        step_metric_values = []
        for group in step_groups:
            probs = [chain.steps[i].probs for i in group]
            metrics = step_level_metrics(probs)
            step_metric_values.append(metrics["mean_entropy"])

        trajectory = align_trajectory(step_metric_values, n_points=args.n_grid_points)
        features = extract_signature_features(trajectory)

        (correct_features if chain.is_correct else incorrect_features).append(features)

        all_chains_summary.append({
            "problem_id": ex["id"],
            "is_correct": chain.is_correct,
            "final_answer": chain.final_answer,
            "gold_answer": ex["gold_answer"],
            "trajectory": trajectory.tolist(),
        })

    print(f"Correct chains: {len(correct_features)} | Incorrect chains: {len(incorrect_features)}")

    if len(correct_features) >= 5 and len(incorrect_features) >= 5:
        comparison = compare_groups(correct_features, incorrect_features)
    else:
        comparison = {"warning": "Not enough chains in one group for a meaningful test."}

    with open(RESULTS_DIR / "exp1_raw_chains.json", "w") as f:
        json.dump(all_chains_summary, f, indent=2)

    with open(RESULTS_DIR / "exp1_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("Saved results to results/exp1_raw_chains.json and results/exp1_comparison.json")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
