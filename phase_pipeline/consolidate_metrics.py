"""Goal is to consolidate the various performance metrics from the phase2 and 3 reports.
Writes one JSON file holding every table. Nothing is from recomputed from verdicts, apart from the alpha table, 
which re-ranks the stored verdicts"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
import choix
from scipy.stats import pearsonr, spearmanr

from .profiles import _extract_json
try:
    from .vote_shares import polling_averages
except ImportError:  # polling benchmark not wired in yet; the table is simply omitted
    def polling_averages(election):
        return None

UNITS = ("care", "loyalty", "authority", "sanctity", "equality", "proportionality", "gal", "tan")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# one row per cell, with the design dimensions split out of the cell name
def cell_rows(report):
    rows = []
    for name, c in report["cells"].items():
        variant, arm, source = name.split("/")
        rows.append({
            "variant": variant, "arm": arm, "source": source,
            "rho": c["spearman"]["rho"],
            "null_percentile": c["null_percentile"],
            "winner": int(bool(c["binary_winner"])),
            "positional_error": c["positional_error"],
            "rejected": round(c["parse_failure_rate"] * report["plan"]["calls_per_cell"]),
            "ranking": " > ".join(c["ranking"]),
        })
    return rows

# signed rank error per party, from the stored ranking: predicted rank minus actual rank
# (+ means placed too low, - too high), for each cell and averaged over the run
def signed_error_rows(report):
    shares = report["vote_shares"]
    actual = sorted(shares, key=shares.get, reverse=True)
    per_cell = []
    for name, c in report["cells"].items():
        pred = c["ranking"]
        per_cell.append({"cell": name, **{p: pred.index(p) - actual.index(p) for p in actual}})
    per_party = [{"party": p, "actual_rank": actual.index(p) + 1,
                  "mean_signed_error": mean(r[p] for r in per_cell),
                  "cells_too_high": sum(r[p] < 0 for r in per_cell),
                  "cells_exact": sum(r[p] == 0 for r in per_cell),
                  "cells_too_low": sum(r[p] > 0 for r in per_cell)} for p in actual]
    return per_cell, per_party


# mean of the performance columns over whichever dimensions are named
def marginal(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for g, rs in sorted(groups.items()):
        out.append({**dict(zip(keys, g)), "n_cells": len(rs),
                    "mean_rho": mean(r["rho"] for r in rs),
                    "cells_rho_positive": sum(r["rho"] > 0 for r in rs),
                    "winner_rate": mean(r["winner"] for r in rs),
                    "mean_positional_error": mean(r["positional_error"] for r in rs),
                    "rejected": sum(r["rejected"] for r in rs)})
    return out


# rho and winner for every alpha the report stored, averaged over cells
def alpha_rows(report):
    shares = report["vote_shares"]
    parties = sorted(shares)
    by_alpha = defaultdict(lambda: {"rho": [], "winner": [], "error": []})
    for c in report["cells"].values():
        for alpha, scores in c["alpha_sensitivity"].items():
            rho, _ = spearmanr([scores[p] for p in parties], [shares[p] for p in parties])
            by_alpha[alpha]["rho"].append(float(rho))
            by_alpha[alpha]["error"].append(mean_error_points(implied_shares(scores, parties), shares, parties))
            by_alpha[alpha]["winner"].append(max(scores, key=scores.get) == max(shares, key=shares.get))
    return [{"alpha": float(a), "mean_rho": mean(v["rho"]), "cells_rho_positive": sum(r > 0 for r in v["rho"]),
             "mean_error_points": mean(v["error"]), "best_cell_error_points": min(v["error"]),
             "winner_rate": mean(v["winner"])} for a, v in sorted(by_alpha.items(), key=lambda kv: float(kv[0]))]


# nulls in the profile Phase 3 used for each arm/source: count of units without a numeric weight
def profile_nulls(phase2):
    nulls = {}
    for arm, sources in phase2["selections"].items():
        for source, sel in sources.items():
            obj = _extract_json(sel["response"]["text"])
            items = next((v for v in obj.values() if isinstance(v, list)), []) if obj else []
            weighted = sum(isinstance(x.get("weight"), (int, float)) and x.get("status", "SCORED") == "SCORED"
                           for x in items)
            nulls[(arm, source)] = len(items) - weighted
    return nulls


# per profile: nulls in it, and the performance of the cells built on it
def nulls_vs_performance(rows, nulls):
    out = []
    for (arm, source), n in sorted(nulls.items()):
        rs = [r for r in rows if r["arm"] == arm and r["source"] == source]
        if rs:
            out.append({"arm": arm, "source": source, "nulls": n, "n_cells": len(rs),
                        "mean_rho": mean(r["rho"] for r in rs), "winner_rate": mean(r["winner"] for r in rs)})
    if len(out) > 2 and len({r["nulls"] for r in out}) > 1:
        r, p = pearsonr([o["nulls"] for o in out], [o["mean_rho"] for o in out])
        out.append({"arm": "pearson r (nulls, mean_rho)", "source": "", "nulls": round(r, 3), "n_cells": len(out),
                    "mean_rho": round(p, 3), "winner_rate": ""})  # last row: r in nulls column, p in mean_rho column
    return out

# The fitted model's choice probability for each party via choix. Constructed via
# a rescaling of the scores onto the units of vote share, for interpretability
# and for the comparison with the polls.
def implied_shares(scores, parties):
    probs = choix.probabilities(range(len(parties)), [scores[p] for p in parties])
    return {p: 100 * float(probs[i]) for i, p in enumerate(parties)}

# shares of a subset, rescaled to sum to 100, so both sides are on the same base
def renormalise(shares, parties):
    total = sum(shares[p] for p in parties)
    return {p: 100 * shares[p] / total for p in parties}

def mean_error_points(implied, shares, parties):
    return mean(abs(implied[p] - shares[p]) for p in parties)


# the raw Bradley-Terry scores, how far apart the
# parties are, how clear the top is, and whether magnitude tracks vote share
def score_rows(report):
    shares = report["vote_shares"]; parties = sorted(shares)
    rows = []
    for name, c in report["cells"].items():
        variant, arm, source = name.split("/")
        sc = c["scores"]; ordered = sorted(sc.values(), reverse=True)
        rows.append({"variant": variant, "arm": arm, "source": source,
                     "spread": ordered[0] - ordered[-1],        # top minus bottom
                     "top_margin": ordered[0] - ordered[1],     # first over second
                     "mean_error_points": mean_error_points(implied_shares(sc, parties), shares, parties),  # points per party, raw
                     **{f"implied_{p}": implied_shares(sc, parties)[p] for p in parties},
                     **{f"score_{p}": sc[p] for p in parties}})
    return rows

def score_marginal(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    return [{**dict(zip(keys, g)), "n_cells": len(rs),
             "mean_spread": mean(r["spread"] for r in rs),
             "mean_top_margin": mean(r["top_margin"] for r in rs),
             "mean_error_points": mean(r["mean_error_points"] for r in rs),
             "best_cell_error_points": min(r["mean_error_points"] for r in rs)}
            for g, rs in sorted(groups.items())]

# how much each design dimension moves performance: the spread between its best
# and worst level, so "does the arm matter more than the variant" is answerable
def contrasts(rows, score_rows_, keys):
    err = {(r["variant"], r["arm"], r["source"]): r["mean_error_points"] for r in score_rows_}
    out = []
    for key in keys:
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        means = {g: mean(r["rho"] for r in rs) for g, rs in groups.items()}
        errs = {g: mean(err[(r["variant"], r["arm"], r["source"])] for r in rs) for g, rs in groups.items()}
        best, worst = max(means, key=means.get), min(means, key=means.get)
        out.append({"dimension": key, "n_levels": len(means),
                    "best_level": best, "best_mean_rho": means[best],
                    "worst_level": worst, "worst_mean_rho": means[worst],
                    "rho_spread": means[best] - means[worst],
                    "error_points_spread": max(errs.values()) - min(errs.values())})
    return out


# Phase 1 replication stability per summary variant: text similarity across the
# five runs, and how far apart the parties within a variant were
def summary_stability_rows(phase1):
    out = []
    for variant, block in phase1.get("stability", {}).items():
        per = block.get("per_cell", {})
        out.append({"variant": variant, "mean_stability": block.get("mean_stability"),
                    "sd_across_parties": block.get("sd_across_cells"),
                    "least_stable": block.get("min_cell"), "most_stable": block.get("max_cell"),
                    "n_parties": len(per)})
    return sorted(out, key=lambda r: r["variant"])


# Measures Phase 2 stability per profile cell, according to both metrics and whether they agree
def profile_stability_rows(phase2):
    out = []
    for arm, sources in phase2.get("selections", {}).items():
        for source, sel in sources.items():
            out.append({"arm": arm, "source": source,
                        "stability_cosine": sel.get("stability_cosine"),
                        "stability_cosine_sd": sel.get("stability_cosine_sd"),
                        "stability_rouge_l": sel.get("stability_rouge_l"),
                        "metrics_agree_on_medoid": sel.get("metrics_agree_on_medoid"),
                        "selected_run": sel.get("selected_run_index"), "n_runs": sel.get("n_runs")})
    return sorted(out, key=lambda r: (r["arm"], r["source"]))
    


# Phase 1 commitment stability is the primary stability measure. The
# ledger accumulates over every election and model, so it is filtered by both.
def commitment_stability_rows(ledger, election, model):
    prefix, suffix = f"summary:{election}_", f"_{model}"
    by_variant = defaultdict(list)
    for key, rec in ledger.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        variant = key[len(prefix):-len(suffix)].split("_", 1)[1]  # key is <election>_<party>_<variant>_<model>
        by_variant[variant].append(rec)
    return [{"variant": v, "n_parties": len(rs),
             "mean_commitment_stability": mean(r["stability"] for r in rs),
             "mean_clusters": mean(r["n_clusters"] for r in rs),
             "mean_consensus_set": mean(r["n_consensus"] for r in rs),
             "mean_coverage_of_selected": mean(r["coverage"] for r in rs),
             "match_threshold": rs[0]["threshold"]}
            for v, rs in sorted(by_variant.items())]
   


# whether the model could name the parties from the summaries it was given
def probe_rows(probes):
    leak, dual = probes.get("leakage", {}), probes.get("dual", {})
    return [{"accuracy": leak.get("accuracy"), "chance": leak.get("chance"),
             "high_confidence_correct_rate": leak.get("high_confidence_correct_rate"),
             "threshold": leak.get("threshold"), "anonymisation_failed": leak.get("anonymisation_failed"),
             "n": leak.get("n"),
             **{f"accuracy_{k}": v for k, v in dual.items() if isinstance(v, (int, float))}}]


# per party: rank error and error in points, per cell and averaged to pick up parties consistently favoured.
def party_rows(report, score_rows_):
    shares = report["vote_shares"]
    actual = sorted(shares, key=shares.get, reverse=True)
    out = []
    for i, party in enumerate(actual):
        ranks = [c["ranking"].index(party) - i for c in report["cells"].values()]
        implied = [r[f"implied_{party}"] for r in score_rows_]
        out.append({"party": party, "actual_rank": i + 1, "actual_share": shares[party],
                    "mean_rank_error": mean(ranks),          # + means placed too low
                    "mean_implied_share": mean(implied),
                    "mean_share_error": mean(v - shares[party] for v in implied),  # + means overstated
                    "cells_overstated": sum(v > shares[party] for v in implied),
                    "n_cells": len(implied)})
    return out


# where each cell's rho sits in its own permutation null: the check that a ranking
# beats chance, which rho alone cannot say on five parties
def null_rows(rows):
    return [{"threshold": t, "cells_at_or_above": sum(r["null_percentile"] >= t for r in rows),
             "n_cells": len(rows)} for t in (0.5, 0.9, 0.95, 0.99)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase3", type=Path)
    ap.add_argument("--phase2", type=Path, default=None)
    ap.add_argument("--phase1", type=Path, default=None)
    ap.add_argument("--probes", type=Path, default=None)
    ap.add_argument("--commitments", type=Path, default=None,
                    help="outputs/commitment_selections.json, the accumulating Phase 1 ledger")
    ap.add_argument("--out", type=Path, default=Path("outputs/tables"))
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    report = load(args.phase3)
    rows = cell_rows(report)
    scores = score_rows(report)
    tables = {
        "label": args.label,
        "model": report["model"],
        "election": report["election"],
        "cells": rows,                                      # every variant/arm/source cell
        "by_arm_source": marginal(rows, ["arm", "source"]),  # profile/prompt combination
        "by_arm": marginal(rows, ["arm"]),                   # prompt combination
        "by_source": marginal(rows, ["source"]),             # profile combination
        "by_variant": marginal(rows, ["variant"]),
        "alpha": alpha_rows(report),
        "scores_by_cell": score_rows(report), # raw Bradley-Terry scores and their spread
        "scores_by_arm": score_marginal(score_rows(report), ["arm"]),
        "scores_by_source": score_marginal(score_rows(report), ["source"]),
        "signed_error_by_party": signed_error_rows(report)[1], # mean over all 50 cells
        "contrasts": contrasts(rows, scores, ["arm", "source", "variant"]),  # how much each dimension moves performance
        "party_error": party_rows(report, scores),        # per party, in ranks and in points
        "null_percentiles": null_rows(rows),              # cells beating their own permutation null
        "signed_error_by_cell": signed_error_rows(report)[0],
        "rejected": [{"cell": f"{r['variant']}/{r['arm']}/{r['source']}", "rejected": r["rejected"]}
                     for r in rows if r["rejected"]] + [{"cell": "total", "rejected": sum(r["rejected"] for r in rows)}],
    }
    polling = polling_averages(report["election"])
    if polling:
        shares = report["vote_shares"]
        parties = sorted(shares)
        polled = sorted(polling)  # parties the polls report separately; the rest sat in "Others"
        partial = len(polled) < len(parties)
        tables["polling_benchmark"] = {
            "polled_parties": polled,
            "partial": partial,
            "polling_ranking": " > ".join(sorted(polling, key=polling.get, reverse=True)),
        }
        if not partial:  # a rank correlation on three of five parties is not comparable with the pipeline's
            rho_polling = float(spearmanr([polling[p] for p in parties], [shares[p] for p in parties]).statistic)
            tables["polling_benchmark"].update({
                "rho_polling": rho_polling,
                "mean_rho_pipeline": mean(r["rho"] for r in rows),
                "best_cell_rho_pipeline": max(r["rho"] for r in rows),
                "cells_matching_or_beating_polls": sum(r["rho"] >= rho_polling for r in rows),
            })
        
        error_points_polling = mean_error_points(renormalise(polling, polled), renormalise(shares, polled), polled)
        cell_error = [mean_error_points(renormalise({p: r[f"implied_{p}"] for p in polled}, polled),
                                        renormalise(shares, polled), polled)
                      for r in score_rows(report)]
        tables["polling_benchmark"].update({
            "error_points_polling": error_points_polling,
            "mean_error_points_pipeline": mean(cell_error),
            "best_cell_error_points_pipeline": min(cell_error),
            "cells_beating_polls_error_points": sum(m <= error_points_polling for m in cell_error),
            "polling_shares": {p: polling[p] for p in polled},
            "actual_shares": {p: shares[p] for p in polled},
        })
    if args.phase2:
        phase2 = load(args.phase2)
        tables["nulls_vs_performance"] = nulls_vs_performance(rows, profile_nulls(load(args.phase2)))
        tables["profile_stability"] = profile_stability_rows(phase2)

    if args.phase1:
        tables["summary_stability"] = summary_stability_rows(load(args.phase1))

    if args.probes:
        tables["probe"] = probe_rows(load(args.probes))

    if args.commitments:
        tables["commitment_stability"] = commitment_stability_rows(
            load(args.commitments), report["election"], report["model"])
        
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"tables_{args.label or report['model']}.json"
    path.write_text(json.dumps(tables, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()