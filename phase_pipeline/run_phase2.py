"""Phase 2: survey records in, selected voter profiles out, one election.

    python -m phase_pipeline.run_phase2 2024 --dry-run
"""

import argparse
import sys
from pathlib import Path

from . import profiles
from .report import write_report
from .prompts import PROFILE_PROMPTS, SOURCE_CONDITIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECORDS = REPO_ROOT / "data" / "voter_profiles"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "phase2"

N_RUNS = 5
ARMS = tuple(PROFILE_PROMPTS)


def load_records(records_dir, election):
    """Read the two survey records for one election.

    Args:
        records_dir (Path): directory written by build_election_data.
        election (int): election year.

    Returns:
        tuple: the Ipsos text and the BES text.

    Raises:
        FileNotFoundError: if either record is missing.
    """
    texts = {}
    for source in ("ipsos", "bes"):
        path = records_dir / f"{election}_{source}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing. Run build_election_data first.")
        texts[source] = path.read_text(encoding="utf-8")
    return texts["ipsos"], texts["bes"]


# call count before anything is spent
def plan(model):
    cells = len(ARMS) * len(SOURCE_CONDITIONS)
    return {
        "model": model,
        "arms": len(ARMS),
        "sources": len(SOURCE_CONDITIONS),
        "runs_per_cell": N_RUNS,
        "cells": cells,
        "total_calls": cells * N_RUNS,
    }


def run_election(election, ipsos, bes, model):
    """Generate and select a profile for every cell.

    Args:
        election (int): election year.
        ipsos (str): the Ipsos record.
        bes (str): the BES record.
        model (str): model identifier passed to llm_client.

    Returns:
        dict: {arm: {source: selection}}, and a list of cells that failed
        to parse.
    """
    selections, failures = {}, []

    for arm in ARMS:
        selections[arm] = {}
        for source in SOURCE_CONDITIONS:
            responses = profiles.generate_profiles(
                election=str(election), arm=arm, source=source, model=model,
                ipsos=ipsos, bes=bes, n_runs=N_RUNS)

            selection = profiles.select_profile(
                election=str(election), arm=arm, source=source, model=model,
                responses=responses)

            # a cell whose selected profile will not parse cannot feed Phase 3
            parsed = profiles.parse_profile(selection["response"], arm)
            if parsed["parse_error"]:
                failures.append(f"{arm}/{source}")

            selections[arm][source] = {
                "selection": selection,
                "parsed": parsed,
            }
            print(f"  {arm}/{source}: {N_RUNS} runs, "
                  f"{'PARSE FAILED' if parsed['parse_error'] else 'parsed'}")

    return selections, failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("election", type=int)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve records and count calls, make none")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        ipsos, bes = load_records(args.records, args.election)
    except FileNotFoundError as err:
        sys.exit(str(err))

    counts = plan(args.model)
    print(f"Election {args.election}: {counts['arms']} arms x "
          f"{counts['sources']} sources x {counts['runs_per_cell']} runs")
    print(f"      {counts['cells']} cells, {counts['total_calls']} calls "
          f"(cold cache)")

    if args.dry_run:
        print("\nDry run: no calls made.")
        print(f"  ipsos record: {len(ipsos.split()):,} words")
        print(f"  bes record:   {len(bes.split()):,} words")
        return

    print()
    selections, failures = run_election(args.election, ipsos, bes, args.model)

    for cell in failures:
        print(f"  PARSE FAILED {cell}", file=sys.stderr)

    report = {
        "election": args.election,
        "model": args.model,
        "plan": counts,
        "parse_failures": failures,
        "selections": {arm: {src: cell["selection"]
                             for src, cell in sources.items()}
                       for arm, sources in selections.items()},
    }
    path = write_report(args.output, f"phase2_{args.election}", report)
    print(f"\nWrote {path}")
    if failures:
        sys.exit(f"{len(failures)} cells failed to parse; Phase 3 needs all "
                 f"{counts['cells']}.")


if __name__ == "__main__":
    main()
