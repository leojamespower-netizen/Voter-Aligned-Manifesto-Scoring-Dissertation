"""Actual vote shares, the primary validation criterion."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VOTE_SHARES_CSV = REPO_ROOT / "data" / "vote_shares.csv"

ELECTIONS = [1997, 2001, 2005, 2010, 2015, 2017, 2019, 2024]

# (election, party_key) pairs with no CHES classification available.
UNCLASSIFIED = {
    (1997, "referendum"),
    (2001, "ukip"),
}


def _load(path=VOTE_SHARES_CSV):
    import pandas as pd
    return pd.read_csv(path)


# {party_key: vote_share} for one election
def vote_shares(election, path=VOTE_SHARES_CSV):
    rows = _load(path)
    rows = rows[rows["election"] == election]
    if rows.empty:
        raise ValueError(f"no vote shares recorded for {election}")
    return dict(zip(rows["party_key"], rows["vote_share"].astype(float)))


# descriptive only; seats reflect geography as well as support
def seats(election, path=VOTE_SHARES_CSV):
    rows = _load(path)
    rows = rows[rows["election"] == election]
    return dict(zip(rows["party_key"], rows["seats"].astype(int)))


# {party_key: ches_party_name}. Unmatched parties omitted
def parties(election, path=VOTE_SHARES_CSV):
    rows = _load(path)
    rows = rows[(rows["election"] == election) & rows["ches_party"].notna()]
    return dict(zip(rows["party_key"], rows["ches_party"]))


def display_names(path=VOTE_SHARES_CSV):
    rows = _load(path)
    return dict(zip(rows["party_key"], rows["party_name"]))


def coverage_table(path=VOTE_SHARES_CSV):
    rows = _load(path)
    out = []
    for e in ELECTIONS:
        r = rows[rows["election"] == e]
        keys = list(r["party_key"])
        out.append({
            "election": e,
            "n_parties": len(keys),
            "parties": keys,
            "share_covered": round(float(r["vote_share"].sum()), 2),
            "unclassified": [k for k in keys if (e, k) in UNCLASSIFIED],
        })
    return out
