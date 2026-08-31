"""Manifesto Project data: category distributions and coverage checks.

Per-document annotated corpus exports, one quasi-sentence per row with its
cmp_code. Rename each download to {election}_{party_key}.csv and place it in
data/cmp/, matching the manifesto convention.
"""

from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CMP_DIR = REPO_ROOT / "data" / "cmp"

# the seven CMP policy domains, by leading digit of the category code
DOMAINS = {
    "1": "external relations",
    "2": "freedom and democracy",
    "3": "political system",
    "4": "economy",
    "5": "welfare and quality of life",
    "6": "fabric of society",
    "7": "social groups",
}


def cmp_path(election, party):
    """Locate one document's corpus export.

    Files are named {election}_{party_key}.csv, matching data/manifestos/.
    Rename the Manifesto Project download rather than keeping its MARPOR
    filename: the codes are easy to get wrong (51210 looks like a plausible
    UK minor party and is in fact Sinn Fein) and nothing here needs them.

    Args:
        election (int): election year.
        party (str): a party_key from vote_shares.csv.

    Returns:
        Path: the expected file, whether or not it exists.
    """
    return CMP_DIR / f"{election}_{party}.csv"


def load_cmp_text(election, party):
    """Concatenated quasi-sentences for one manifesto.

    This is a registered Phase 3 input condition - CMP-coded text as an
    alternative to LLM summaries - not an extraction fallback. The fallback
    role is retired: all forty manifestos extracted successfully and the
    quality gate never fired.

    Note the corpus is segmented and coder-selected, so it is shorter than
    the source document. Check the row count against a large party before
    treating it as full text.

    Args:
        election (int): election year.
        party (str): a party_key from vote_shares.csv.

    Returns:
        str: the quasi-sentences, one per line.
    """
    import pandas as pd

    path = cmp_path(election, party)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Download the corpus export from the Manifesto "
            f"Project and rename it {path.name}.")
    rows = pd.read_csv(path)
    return "\n".join(str(t) for t in rows["text"] if str(t) != "nan")


def load_cmp_category_distribution(election, party):
    """Proportion of quasi-sentences per CMP category.

    Args:
        election (int): election year.
        party (str): a party_key from vote_shares.csv.

    Returns:
        dict: {category code: proportion}, plus the same aggregated to the
        seven policy domains and the number of coded sentences.
    """
    import pandas as pd

    path = cmp_path(election, party)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing. Download the corpus export from the Manifesto "
            f"Project and rename it {path.name}.")

    rows = pd.read_csv(path)
    codes = [str(c) for c in rows["cmp_code"] if str(c) not in ("nan", "H")]
    if not codes:
        raise ValueError(f"{path} has no coded sentences")

    by_category = defaultdict(int)
    by_domain = defaultdict(int)
    for code in codes:
        by_category[code] += 1
        by_domain[DOMAINS.get(code[0], "uncoded")] += 1

    n = len(codes)
    return {
        "categories": {k: v / n for k, v in by_category.items()},
        "domains": {k: v / n for k, v in by_domain.items()},
        "n_coded": n,
        "n_rows": len(rows),
    }
