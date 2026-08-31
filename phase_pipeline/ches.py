"""Chapel Hill Expert Survey: party families and expert positions."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHES_CSV = REPO_ROOT / "data" / "ches" / "1999-2024_CHES_dataset_meansV2.csv"
UK_COUNTRY_CODE = 11

# CHES family codes (standard coding).
FAMILY_LABELS = {
    1: "radical right",
    2: "conservative",
    3: "liberal",
    4: "christian democratic",
    5: "socialist / social democratic",
    6: "radical left",
    7: "green",
    8: "regionalist",
    9: "no family",
    10: "confessional",
    11: "agrarian / centre",
}

# Coarser grouping for directional error, where n per family must be > 1 for
# a family mean to mean anything. REGISTERED - fix before results.
FAMILY_TO_BLOC = {
    6: "left", 7: "left", 5: "centre-left",
    3: "centre", 11: "centre",
    2: "centre-right", 4: "centre-right",
    1: "right",
    8: "regionalist",   # excluded from the left-right directional test
    9: "unclassified",
}

# REGISTERED assignment of CHES round to election. Fixed before any results
# were seen. Six of eight elections have a round that directly references
# them; 2015 and 2017 do not, and carry the assignments noted below.
ROUND_FOR_ELECTION = {
    1997: 1999,   # direct
    2001: 2002,   # direct
    2005: 2006,   # direct
    2010: 2010,   # direct
    2015: 2014,   # INDIRECT - nearest preceding round, references 2010
    2017: 2019,   # INDIRECT - nearest round; see limitation below
    2019: 2019,   # direct
    2024: 2024,   # direct
}

# Elections whose classification is indirect. Reported as such; any
# directional-error result should be checked with and without them.
INDIRECT_ASSIGNMENT = {
    2015: ("2014 round, which references the 2010 election. Positions are one "
           "cycle stale but formed before the election being classified."),
    2017: ("2019 round, which references the 2019 election. REGISTERED "
           "LIMITATION: the only backward assignment in the set - expert "
           "judgement was formed AFTER the election being classified. The "
           "magnitude is measurable and is not small: Labour's lrgen moves "
           "3.6 (2014 round) -> 1.9 (2019 round), a shift of 1.7 points on a "
           "0-10 scale, the largest movement by any UK party across adjacent "
           "rounds in this dataset. Conservative positions are stable across "
           "the same interval (7.0 -> 7.1), so the risk is concentrated in "
           "Labour's placement. Chosen over the 2014 round, which references "
           "2010 and would be two cycles stale; neither option is clean. "
           "Directional-error results should be reported with and without "
           "2017."),
}

# NOTE: the 1999/2002/2006/2010 rounds also post-date the elections they
# reference, but by CHES design - experts are asked about positions AT that
# election, so the reference is intended and the placement is contemporaneous.


# uK rows only, as a DataFrame
def load_uk(path=CHES_CSV):
    import pandas as pd
    df = pd.read_csv(path)
    return df[df["country"] == UK_COUNTRY_CODE].copy()


# {internal_party_key: bloc} for one election
def party_families(election, parties, path=CHES_CSV):
    uk = load_uk(path)
    rnd = ROUND_FOR_ELECTION[election]
    rows = uk[uk["year"] == rnd]
    out = {}
    for key, ches_name in parties.items():
        match = rows[rows["party"] == ches_name]
        if match.empty:
            out[key] = "unclassified"
            continue
        out[key] = FAMILY_TO_BLOC.get(int(match.iloc[0]["family"]), "unclassified")
    return out


# expert placements on one dimension: 'galtan', 'lrgen', or 'lrecon'
def expert_positions(election, parties, dimension='galtan', path=CHES_CSV):
    uk = load_uk(path)
    rows = uk[uk["year"] == ROUND_FOR_ELECTION[election]]
    out = {}
    for key, ches_name in parties.items():
        match = rows[rows["party"] == ches_name]
        if not match.empty and match.iloc[0][dimension] == match.iloc[0][dimension]:
            out[key] = float(match.iloc[0][dimension])
    return out


# correlate axis-arm Bradley-Terry scores against CHES expert galtan
def axis_benchmark(bt_scores, election, parties, path=CHES_CSV):
    from scipy.stats import spearmanr
    expert = expert_positions(election, parties, "galtan", path)
    common = sorted(set(bt_scores) & set(expert))
    if len(common) < 3:
        return {"n": len(common), "rho": None,
                "note": "too few parties matched for a correlation"}
    rho = spearmanr([bt_scores[p] for p in common],
                    [expert[p] for p in common]).statistic
    return {"n": len(common), "rho": float(rho),
            "expert_positions": {p: expert[p] for p in common},
            "round_used": ROUND_FOR_ELECTION[election]}


# limitation text for an indirectly assigned election, or None
def assignment_note(election):
    return INDIRECT_ASSIGNMENT.get(election)


# round assignment and directness for every election in scope
def coverage_table():
    return [{"election": e, "round": r,
             "direct": e not in INDIRECT_ASSIGNMENT,
             "note": INDIRECT_ASSIGNMENT.get(e)}
            for e, r in sorted(ROUND_FOR_ELECTION.items())]
