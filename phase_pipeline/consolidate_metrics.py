"""Goal of this addition is to consolidate the various performance metrics from the phase2 and 3 reports.
Writes one JSON file holding every table. Nothing is from recomputed from verdicts, apart from the alpha table, 
which re-ranks the stored verdicts"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

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
    by_alpha = defaultdict(lambda: {"rho": [], "winner": []})
    for c in report["cells"].values():
        for alpha, scores in c["alpha_sensitivity"].items():
            rho, _ = spearmanr([scores[p] for p in parties], [shares[p] for p in parties])
            by_alpha[alpha]["rho"].append(float(rho))
            by_alpha[alpha]["winner"].append(max(scores, key=scores.get) == max(shares, key=shares.get))
    return [{"alpha": float(a), "mean_rho": mean(v["rho"]), "cells_rho_positive": sum(r > 0 for r in v["rho"]),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase3", type=Path)
    ap.add_argument("--phase2", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/tables"))
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    report = load(args.phase3)
    rows = cell_rows(report)
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
        "signed_error_by_party": signed_error_rows(report)[1],   # mean over all 50 cells
        "signed_error_by_cell": signed_error_rows(report)[0],
        "rejected": [{"cell": f"{r['variant']}/{r['arm']}/{r['source']}", "rejected": r["rejected"]}
                     for r in rows if r["rejected"]] + [{"cell": "total", "rejected": sum(r["rejected"] for r in rows)}],
    }
    polling = polling_averages(report["election"])
    if polling:
        shares = report["vote_shares"]; parties = sorted(shares)
        rho_polling = float(spearmanr([polling[p] for p in parties], [shares[p] for p in parties]).statistic)
        tables["polling_benchmark"] = {
            "rho_polling": rho_polling,                      # polls versus the result, same criterion as the pipeline
            "polling_ranking": " > ".join(sorted(polling, key=polling.get, reverse=True)),
            "mean_rho_pipeline": mean(r["rho"] for r in rows),
            "best_cell_rho_pipeline": max(r["rho"] for r in rows),
            "cells_matching_or_beating_polls": sum(r["rho"] >= rho_polling for r in rows),
        }
    if args.phase2:
        tables["nulls_vs_performance"] = nulls_vs_performance(rows, profile_nulls(load(args.phase2)))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"tables_{args.label or report['model']}.json"
    path.write_text(json.dumps(tables, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()