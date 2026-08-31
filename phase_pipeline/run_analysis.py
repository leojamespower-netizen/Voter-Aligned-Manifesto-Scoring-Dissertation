"""Analysis across cells and elections, from the Phase 3 reports.

    python -m phase_pipeline.run_analysis
    python -m phase_pipeline.run_analysis --elections 2015 2019 2024

Phase 3 scores one design cell at a time and writes it. The measures here
need more than one cell: framing sensitivity compares verdicts between
conditions, directional error pools party families across elections, and the
invariance floor is what every other rate is judged against. Nothing new is
called - these are the validate functions Phase 3 cannot reach.

Makes no API calls.
"""

import argparse
import json
import sys
from pathlib import Path

from . import ches, validate
from .report import write_report
from .vote_shares import parties as ches_names, vote_shares

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PHASE3 = REPO_ROOT / "outputs" / "phase3"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "analysis"


def load_reports(phase3_dir, elections, source):
    """Read every Phase 3 report for the requested elections.

    Args:
        phase3_dir (Path): directory of phase3_{election}_{source}.json.
        elections (list): election years, or None for whatever is present.
        source (str): "summary" or "cmp".

    Returns:
        dict: {election: report}.
    """
    reports = {}
    pattern = f"phase3_*_{source}.json"
    for path in sorted(phase3_dir.glob(pattern)):
        report = json.loads(path.read_text(encoding="utf-8"))
        election = report["election"]
        if elections and election not in elections:
            continue
        reports[election] = report
    if not reports:
        raise FileNotFoundError(
            f"No reports matching {pattern} in {phase3_dir}. Run run_phase3 "
            f"first.")
    return reports


def pool_verdicts(reports):
    """Every verdict from every report, in one list."""
    return [v for r in reports.values() for v in r.get("verdicts", [])]


def best_cell(report):
    """The cell with the highest rank correlation.

    Cross-election measures need one score set per election, and the design
    grid gives many. The best cell is used, and which one it was is recorded
    so the choice is visible rather than buried.
    """
    cells = report["cells"]
    if not cells:
        return None, None
    key = max(cells, key=lambda k: cells[k]["spearman"]["rho"])
    return key, cells[key]


def baseline_cell(report):
    """The no-profile cell, for profile_value_added."""
    for key, cell in report["cells"].items():
        if key.endswith("/noprofile"):
            return cell
    return None


def noise_floors(verdicts):
    """The rates every manipulation is judged against."""
    return {
        "invariance": validate.invariance_violation(verdicts),
        "framing": validate.framing_sensitivity(verdicts),
        "calibration": validate.calibration_error(verdicts),
    }


def per_party(reports):
    """Signed rank error per party, and pooled by ideological family."""
    bt = {e: best_cell(r)[1]["scores"] for e, r in reports.items()
          if best_cell(r)[1]}
    shares = {e: vote_shares(e) for e in bt}

    families = {}
    for election in bt:
        try:
            families.update(
                ches.party_families(election, ches_names(election)))
        except Exception as err:                      # noqa: BLE001
            print(f"  no CHES families for {election}: {err}", file=sys.stderr)

    out = {"party_error": validate.party_error_profile(bt, shares)}
    if families:
        out["directional_error"] = validate.directional_error(
            bt, shares, families)
        out["families"] = families
    return out


def profile_contribution(reports):
    """Whether the Phase 2 profile changed the ranking at all."""
    out = {}
    for election, report in reports.items():
        baseline = baseline_cell(report)
        _, best = best_cell(report)
        if baseline is None or best is None:
            continue
        out[election] = validate.profile_value_added(
            best["scores"], baseline["scores"], vote_shares(election))
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase3", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", choices=("summary", "cmp"),
                        default="summary")
    parser.add_argument("--elections", type=int, nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        reports = load_reports(args.phase3, args.elections, args.source)
    except FileNotFoundError as err:
        sys.exit(str(err))

    elections = sorted(reports)
    verdicts = pool_verdicts(reports)
    print(f"Analysing {len(elections)} elections "
          f"({', '.join(str(e) for e in elections)}), "
          f"{len(verdicts):,} verdicts")

    if not verdicts:
        sys.exit("Reports contain no verdicts. They may predate the change "
                 "that started recording them; rerun run_phase3.")

    analysis = {
        "elections": elections,
        "source": args.source,
        "n_verdicts": len(verdicts),
        "best_cell": {e: best_cell(r)[0] for e, r in reports.items()},
        "noise_floors": noise_floors(verdicts),
        "profile_value_added": profile_contribution(reports),
    }
    analysis.update(per_party(reports))

    floor = analysis["noise_floors"]["invariance"]["floor"]
    ordering = analysis["noise_floors"]["framing"]["ordering_holds"]
    print(f"  invariance floor:   {floor:.1%}" if floor is not None
          else "  invariance floor:   n/a")
    print(f"  registered ordering holds: {ordering}")

    path = write_report(args.output, f"analysis_{args.source}", analysis)
    print(f"\nWrote {path}")
    print("Not computed here: benchmark_against_polling needs polling "
          "averages, adversarial_ablation needs an adversarial run.")


if __name__ == "__main__":
    main()
