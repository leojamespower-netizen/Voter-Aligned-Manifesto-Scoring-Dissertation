"""Secondary stability metric: text similarity, reported alongside."""

import json
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTIONS = REPO_ROOT / "outputs" / "medoid_selections.json"

_MODEL = None  # lazily loaded sentence-transformer


# load the embedding model once
def _embedder():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Primary stability metric needs sentence-transformers: "
                "pip install sentence-transformers. Use metric='rouge_l' "
                "for the secondary metric only."
            ) from exc
        # Registered model id - pin it, it affects the similarity values.
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


# pairwise cosine similarity between embeddings of `texts`
def cosine_matrix(texts):
    import numpy as np
    emb = _embedder().encode(texts, normalize_embeddings=True)
    sim = np.asarray(emb) @ np.asarray(emb).T
    return sim.tolist()


# pairwise ROUGE-L F1
def rouge_l_matrix(texts):
    def lcs(a, b):
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
            prev = cur
        return prev[-1]

    toks = [t.lower().split() for t in texts]
    n = len(texts)
    m = [[1.0] * n for _ in range(n)]
    for i, j in combinations(range(n), 2):
        l = lcs(toks[i], toks[j])
        p = l / len(toks[i]) if toks[i] else 0.0
        r = l / len(toks[j]) if toks[j] else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        m[i][j] = m[j][i] = f
    return m


# stability of one cell's replicate runs
def cell_stability(texts, metric='cosine'):
    if len(texts) < 2:
        raise ValueError("stability needs at least 2 replicate runs")
    m = cosine_matrix(texts) if metric == "cosine" else rouge_l_matrix(texts)
    n = len(texts)
    offdiag = [m[i][j] for i, j in combinations(range(n), 2)]
    row_means = [mean(m[i][j] for j in range(n) if j != i) for i in range(n)]
    best = max(range(n), key=lambda i: (row_means[i], -i))
    return {
        "metric": metric,
        "n_runs": n,
        "stability": mean(offdiag),
        "sd": pstdev(offdiag) if len(offdiag) > 1 else 0.0,
        "matrix": m,
        "row_means": row_means,
        "medoid_index": best,
    }


# select and RECORD the medoid response for one cell
def select_medoid(cell_key, responses, text_field='text'):
    texts = [r[text_field] for r in responses]
    primary = cell_stability(texts, "cosine")
    secondary = cell_stability(texts, "rouge_l")

    record = {
        "cell": cell_key,
        "selected_run_index": primary["medoid_index"],
        "selected_cache_key": responses[primary["medoid_index"]]["cache_key"],
        "stability_cosine": primary["stability"],
        "stability_cosine_sd": primary["sd"],
        "stability_rouge_l": secondary["stability"],
        "metrics_agree_on_medoid":
            primary["medoid_index"] == secondary["medoid_index"],
        "n_runs": primary["n_runs"],
    }
    _append_selection(record)
    return {**record, "response": responses[primary["medoid_index"]]}


def _append_selection(record):
    SELECTIONS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(SELECTIONS.read_text()) if SELECTIONS.exists() else {}
    data[record["cell"]] = record
    SELECTIONS.write_text(json.dumps(data, indent=2))


# aggregate stability for one variant across cells
def variant_stability(cells, metric='cosine'):
    per_cell = {k: cell_stability(v, metric)["stability"]
                for k, v in cells.items()}
    vals = list(per_cell.values())
    return {
        "mean_stability": mean(vals),
        "sd_across_cells": pstdev(vals) if len(vals) > 1 else 0.0,
        "min_cell": min(per_cell, key=per_cell.get),
        "max_cell": max(per_cell, key=per_cell.get),
        "per_cell": per_cell,
    }
