"""Identification probes: can the model name the election from a summary?

    python -m phase_pipeline.run_probes 2024 --dry-run

Run this BEFORE building any anonymisation, on non-anonymised summaries.
If the model cannot identify the election from those, it will not identify
it from anonymised ones either, and the anonymisation arm is unnecessary.

Two probes per summary. minimal_check asks what the text is with no
scaffolding, which measures spontaneous recognition. biased_check supplies
the frame and a party list, which is an upper bound rather than a
measurement of the same thing.
"""

import argparse
import sys
from pathlib import Path

from . import probe
from .report import write_report
from .run_phase3 import load_phase1
from .vote_shares import display_names, vote_shares

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PHASE1 = REPO_ROOT / "outputs" / "phase1"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "probes"

# the probe reads the control-group summaries, not every variant
DEFAULT_VARIANTS = ("minimal", "neutral")


def plan(summaries, variants, model):
    """Call count before anything is spent. Two probes per summary."""
    n = sum(len(summaries.get(v, {})) for v in variants)
    return {"model": model, "summaries": n, "total_calls": n * 2}


def run_election(election, summaries, variants, model):
    """Both probes on every summary in the requested variants."""
    parties = list(vote_shares(election))
    names = display_names()
    results = []

    for variant in variants:
        for party, text in summaries.get(variant, {}).items():
            minimal = probe.minimal_check(
                election=str(election), party=party, text=text,
                model=model, variant=variant)
            results.append(minimal)

            biased = probe.biased_check(
                election=str(election), party=party, text=text,
                model=model, parties=[names.get(p, p) for p in parties],
                variant=variant, aliases=[names.get(party, party)])
            results.append(biased)

            print(f"  {variant}/{party}: minimal "
                  f"{'correct' if minimal['correct'] else 'no'}, biased "
                  f"{'correct' if biased['correct'] else 'no'}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("election", type=int)
    parser.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--temperature", type=float, default=None, help="override the registed temperature for this run")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature is not None:
        from . import llm_client
        llm_client.TEMPERATURE_OVERRIDE = args.temperature
    try:
        summaries = load_phase1(args.phase1, args.election, args.variants)
    except FileNotFoundError as err:
        sys.exit(str(err))

    counts = plan(summaries, args.variants, args.model)
    n_parties = len(vote_shares(args.election))
    print(f"Election {args.election}: {counts['summaries']} summaries x "
          f"2 probes = {counts['total_calls']} calls")
    print(f"      chance on the biased probe is "
          f"{1 / n_parties:.0%} with {n_parties} parties")

    if args.dry_run:
        print("\nDry run: no calls made.")
        for variant in args.variants:
            print(f"  {variant}: {len(summaries.get(variant, {}))} summaries")
        return

    print()
    results = run_election(args.election, summaries, args.variants,
                           args.model)

    leakage = probe.leakage_report(results, n_parties)
    dual = probe.dual_report(results, n_parties)

    print(f"\naccuracy: {leakage['accuracy']:.1%} against "
          f"{leakage['chance']:.0%} chance")
    print(f"minimal {dual['minimal']:.1%} | biased {dual['biased']:.1%}"
          if dual["minimal"] is not None else "")
    print(f"threshold {leakage['threshold']:.0%} -> anonymisation "
          f"{'REQUIRED' if leakage['anonymisation_failed'] else 'not indicated'}")

    report = {
        "election": args.election,
        "model": args.model,
        "plan": counts,
        "leakage": leakage,
        "dual": dual,
        "results": results,
    }
    path = write_report(args.output, f"probes_{args.election}", report)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
