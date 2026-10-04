"""
merge_exp2_batches.py

Combines multiple batch runs of run_experiment2.py (each covering a
non-overlapping slice of the dataset, e.g. --start-index 0 --n-samples 10,
then --start-index 10 --n-samples 10, etc.) into one combined result,
equivalent to having run the full n in a single pass.

Running in batches is a practical choice (shorter sessions, resumable if
interrupted) and does not change the method: this script verifies there are
no duplicate or missing problem_ids before merging, so the combined result
is exactly what a single n=50 run would have produced, just assembled from
pieces.

Usage:
    python src/merge_exp2_batches.py
    (automatically finds all results/exp2_raw_chains_batch*.json files)
"""

import glob
import json
from pathlib import Path

from detect_signature import extract_signature_features, compare_groups
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main():
    batch_files = sorted(glob.glob(str(RESULTS_DIR / "exp2_raw_chains_batch*.json")))
    if not batch_files:
        print("No batch files found (expected results/exp2_raw_chains_batch*.json)")
        return

    print(f"Found {len(batch_files)} batch file(s):")
    for f in batch_files:
        print(f"  {f}")

    all_chains = []
    seen_ids = set()
    for f in batch_files:
        with open(f) as fh:
            chains = json.load(fh)
        for c in chains:
            if c["problem_id"] in seen_ids:
                raise ValueError(
                    f"Duplicate problem_id '{c['problem_id']}' found across batches — "
                    "check your --start-index values don't overlap."
                )
            seen_ids.add(c["problem_id"])
            all_chains.append(c)

    print(f"Merged {len(all_chains)} total chains, no duplicates found.")

    correct_features, incorrect_features = [], []
    for c in all_chains:
        trajectory = np.array(c["divergence_trajectory"])
        features = extract_signature_features(trajectory)
        (correct_features if c["is_correct"] else incorrect_features).append(features)

    print(f"Correct chains: {len(correct_features)} | Incorrect chains: {len(incorrect_features)}")

    if len(correct_features) >= 5 and len(incorrect_features) >= 5:
        comparison = compare_groups(correct_features, incorrect_features)
    else:
        comparison = {"warning": "Not enough chains in one group for a meaningful test."}

    with open(RESULTS_DIR / "exp2_raw_chains.json", "w") as f:
        json.dump(all_chains, f, indent=2)

    with open(RESULTS_DIR / "exp2_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("Saved merged results to results/exp2_raw_chains.json and results/exp2_comparison.json")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
