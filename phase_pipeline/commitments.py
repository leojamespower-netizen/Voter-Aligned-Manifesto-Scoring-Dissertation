"""Primary stability metric: does the same policy content survive?"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTIONS = REPO_ROOT / "outputs" / "commitment_selections.json"

# registered constants
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # pinned; threshold is tied to it
MATCH_THRESHOLD = 0.75                 # cosine above which two commitments match
CONSENSUS_MIN = 3                      # replicates a commitment must appear in
SENSITIVITY_GRID = (0.65, 0.70, 0.75, 0.80, 0.85)

_MODEL = None


# embed short commitment clauses
def _embed(texts, embedder=None):
    if embedder is not None:
        return embedder(texts)
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL.encode(texts, normalize_embeddings=True)

# extraction

# one cached extraction call
def extract_commitments(summary, model, cache_key):
    from .llm_client import call_llm
    from .prompts import EXTRACTION_PROMPT
    resp = call_llm(EXTRACTION_PROMPT.format(summary=summary),
                    model=model, cache_key=cache_key, subdir="extractions")
    return parse_lines(resp["text"])


# strip bullets and numbering the model may add despite instructions
def parse_lines(raw):
    out = []
    for line in raw.strip().splitlines():
        line = line.strip().lstrip("-*\u2022").strip()
        if line[:1].isdigit():
            for sep in (".", ")"):
                head, _, tail = line.partition(sep)
                if head.isdigit():
                    line = tail.strip()
                    break
        if line and line.upper() != "NONE":
            out.append(line)
    return out

# clustering

# cluster commitments across replicates; return one entry per cluster
def cluster_commitments(replicates, threshold=MATCH_THRESHOLD, embedder=None):
    flat, origin = [], []
    for r, lst in enumerate(replicates):
        for c in lst:
            flat.append(c)
            origin.append(r)
    if not flat:
        return []

    order = sorted(range(len(flat)), key=lambda i: (flat[i].lower(), origin[i]))
    emb = _embed([flat[i] for i in order], embedder)

    canonical = []          # indices into `order`
    assignment = [-1] * len(order)
    for pos in range(len(order)):
        for c_idx, c_pos in enumerate(canonical):
            if float(emb[pos] @ emb[c_pos]) >= threshold:
                assignment[pos] = c_idx
                break
        else:
            assignment[pos] = len(canonical)
            canonical.append(pos)

    clusters = {}
    for pos, c_idx in enumerate(assignment):
        i = order[pos]
        entry = clusters.setdefault(c_idx, {
            "canonical": flat[order[canonical[c_idx]]],
            "replicates": set(), "variants": set()})
        entry["replicates"].add(origin[i])
        entry["variants"].add(flat[i])

    n = len(replicates)
    out = [{"commitment": v["canonical"],
            "survival": len(v["replicates"]),
            "survival_rate": len(v["replicates"]) / n,
            "replicates": sorted(v["replicates"]),
            "phrasings": sorted(v["variants"])}
           for v in clusters.values()]
    return sorted(out, key=lambda d: (-d["survival"], d["commitment"]))

# stability and selection

# stability, consensus set, and selected replicate for one cell
def cell_stability(replicates, threshold=MATCH_THRESHOLD,
                   consensus_min=CONSENSUS_MIN, embedder=None):
    n = len(replicates)
    if n < 2:
        raise ValueError("stability needs at least 2 replicates")
    clusters = cluster_commitments(replicates, threshold, embedder)
    if not clusters:
        raise ValueError("no commitments extracted from any replicate")

    consensus = [c for c in clusters if c["survival"] >= consensus_min]
    consensus_names = {c["commitment"] for c in consensus}

    # Coverage: how much of the consensus set each replicate contains.
    coverage = []
    for r in range(n):
        got = sum(1 for c in consensus if r in c["replicates"])
        coverage.append(got / len(consensus) if consensus else 0.0)
    selected = max(range(n), key=lambda r: (coverage[r], -r))

    return {
        "stability": sum(c["survival_rate"] for c in clusters) / len(clusters),
        "n_clusters": len(clusters),
        "n_consensus": len(consensus),
        "consensus_set": sorted(consensus_names),
        "coverage_per_replicate": coverage,
        "selected_index": selected,
        "n_commitments_per_replicate": [len(r) for r in replicates],
        "clusters": clusters,
        "threshold": threshold,
        "consensus_min": consensus_min,
        "embedding_model": EMBEDDING_MODEL,
    }


# select and RECORD the representative replicate for one cell
def select_replicate(cell_key, responses, commitment_lists,
                     threshold=MATCH_THRESHOLD, embedder=None):
    stats = cell_stability(commitment_lists, threshold, embedder=embedder)
    idx = stats["selected_index"]
    record = {
        "cell": cell_key,
        "selected_index": idx,
        "selected_cache_key": responses[idx]["cache_key"],
        "stability": stats["stability"],
        "n_clusters": stats["n_clusters"],
        "n_consensus": stats["n_consensus"],
        "coverage": stats["coverage_per_replicate"][idx],
        "threshold": threshold,
        "embedding_model": EMBEDDING_MODEL,
    }
    SELECTIONS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(SELECTIONS.read_text()) if SELECTIONS.exists() else {}
    data[cell_key] = record
    SELECTIONS.write_text(json.dumps(data, indent=2))
    return {**record, "response": responses[idx], "detail": stats}

# sensitivity

# re-run the whole measure across a grid of match thresholds
def threshold_sensitivity(replicates, grid=SENSITIVITY_GRID,
                          consensus_min=CONSENSUS_MIN, embedder=None):
    rows = {}
    for th in grid:
        s = cell_stability(replicates, th, consensus_min, embedder)
        rows[th] = {"stability": s["stability"],
                    "n_clusters": s["n_clusters"],
                    "n_consensus": s["n_consensus"],
                    "selected_index": s["selected_index"]}
    selections = {v["selected_index"] for v in rows.values()}
    return {
        "per_threshold": rows,
        "selection_stable": len(selections) == 1,
        "selections": sorted(selections),
        "registered_threshold": MATCH_THRESHOLD,
    }


# sample commitment pairs spanning the similarity range, for hand-judging
def calibration_pairs(replicates, n=50, embedder=None):
    flat = sorted({c for lst in replicates for c in lst})
    if len(flat) < 2:
        return []
    emb = _embed(flat, embedder)
    pairs = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            pairs.append((flat[i], flat[j], float(emb[i] @ emb[j])))
    pairs.sort(key=lambda p: -p[2])
    if len(pairs) <= n:
        return pairs
    step = len(pairs) / n            # spread across the range, not just the top
    return [pairs[int(k * step)] for k in range(n)]
