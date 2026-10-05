"""
semantic_divergence.py

v1 upgrade to continuation_divergence.py's Jaccard metric, as flagged in that
module's own docstring: measures how much k independently-sampled
continuations diverge from one another using sentence-embedding cosine
distance instead of surface token overlap.

Why this matters: Jaccard distance treats two continuations that say the same
thing in different words (e.g. "the total is 12" vs. "that gives us 12") as
highly divergent, since they share almost no tokens. That's a plausible
source of noise in Experiment 2's signal — real uncertainty (the model
genuinely unsure what comes next) gets mixed together with superficial
wording variation (the model sure of the content, just phrasing it
differently each sample). Embedding-based divergence should strip out most
of that wording noise and isolate divergence in meaning.

Uses sentence-transformers/all-MiniLM-L6-v2: small (~80MB), fast on CPU,
a standard baseline for semantic similarity. Loaded once per script run and
reused across all checkpoints/problems — reloading it per-call would
dominate runtime.

Everything else about Experiment 3 (chain generation, checkpoint branching,
peak/slope feature extraction, shuffle-null significance testing) is
unchanged from Experiment 2; only this metric differs. See run_experiment3.py.
"""

from itertools import combinations
from typing import List

import numpy as np


class SemanticDivergenceScorer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model {model_name}...")
        self.model = SentenceTransformer(model_name)

    def mean_pairwise_divergence(self, continuations: List[str]) -> float:
        """
        Mean cosine distance (1 - cosine similarity) across all pairs of k
        continuations' sentence embeddings. Same interpretation as the
        Jaccard version in continuation_divergence.py: higher = continuations
        disagree more = higher local uncertainty at the checkpoint they were
        branched from. Kept as the same metric name/shape (list of strings in,
        single float out) so run_experiment3.py is otherwise identical to
        run_experiment2.py.
        """
        if len(continuations) < 2:
            return float("nan")

        # Empty-string continuations (e.g. the model immediately produced an
        # end-of-sequence token) still embed to *some* vector, so unlike
        # jaccard_distance's empty-set special case, no extra handling is
        # needed here.
        embeddings = self.model.encode(continuations, normalize_embeddings=True)

        pairs = list(combinations(range(len(continuations)), 2))
        distances = [1.0 - float(np.dot(embeddings[i], embeddings[j])) for i, j in pairs]
        return sum(distances) / len(distances)
