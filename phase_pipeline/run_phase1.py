"""Phase 1: manifesto text in, selected summaries out, one election.

    python -m phase_pipeline.run_phase1 2024 --dry-run
"""

import argparse
import sys
from pathlib import Path

from . import summarise
from .report import write_report
from .prompts import MAIN_VARIANTS, VARIANTS
from .vote_shares import display_names, vote_shares

REPO_ROOT = Path(__file__).resolve().parent.parent


GATE_EXEMPT = {
    (1997, "referendum"),   # Statement of Aims; no manifesto, no CMP entry
    (2001, "lab"),          # "Investment and reform"; no complete source
}
DEFAULT_MANIFESTOS = REPO_ROOT / "data" / "manifestos"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "phase1"


def load_manifestos(directory, election):
    """Read every party's manifesto for one election.

    A missing or gate-failing document is reported rather than raised, so a
    partial run proceeds instead of stopping at the first gap.

    Args:
        directory (Path): directory of {election}_{party}.txt files.
        election (int): election year.

    Returns:
        tuple: {party: text}, and [(party, reason, detail)] for the rest.
    """
    texts, problems = {}, []
    for party_key in vote_shares(election):
        path = directory / f"{election}_{party_key}.txt"
        if not path.is_file():
            problems.append((party_key, "no file", f"{election}_{party_key}.txt"))
            continue
        text = path.read_text(encoding="utf-8")
        exempt = (election, party_key) in GATE_EXEMPT
        if not summarise.extraction_ok(text) and not exempt:
            problems.append((party_key, "failed extraction gate",
                             f"{len(text):,} chars"))
            continue
        if exempt:
            print(f"  {party_key}: gate exempt "
                  f"({len(text.split()):,} words)")
        texts[party_key] = text
    return texts, problems


# call count before anything is spent
def plan(texts, model):
    per_manifesto = len(variants) * summarise.N_RUNS
    summaries = len(texts) * per_manifesto
    return {
        "model": model,
        "manifestos": len(texts),
        "variants": len(variants),
        "runs_per_variant": summarise.N_RUNS,
        "summary_calls": summaries,
        "commitment_calls": summaries,
        "total_calls": summaries * 2,
    }


# returns selection plus the five replicate texts the ranking needs
def run_election(election, texts, model, replacements=None):
    names = display_names()
    variants = variants or list(VARIANTS)
    replacements = replacements or {}
    selections = {}

    for party_key, text in texts.items():
        print(f"  {party_key}: summarising "
              f"({len(VARIANTS)} variants x {summarise.N_RUNS} runs)")
        responses = summarise.summarise_manifesto(
            election=str(election),
            party=party_key,
            manifesto_text=text,
            model=model,
            party_name=names.get(party_key, party_key),
            replacement_list=replacements.get(party_key), variants=variants
        )

        # summarise_manifesto returns one flat list; split it back into cells
        # in the order VARIANTS was iterated.
        selections[party_key] = {}
        for i, variant in enumerate(variants):
            start = i * summarise.N_RUNS
            cell = responses[start:start + summarise.N_RUNS]
            selections[party_key][variant] = {
                "selection": summarise.select_summary(
                    election=str(election), party=party_key, variant=variant,
                    model=model, responses=cell),
                "texts": [r["text"] for r in cell],
            }
        print(f"  {party_key}: selected {len(selections[party_key])} replicates")

    return selections


def score_stability(selections, election, model):
    by_variant = {}
    for party_key, variants in selections.items():
        for variant, cell in variants.items():
            cell_key = f"{election}_{party_key}_{model}"
            by_variant.setdefault(variant, {})[cell_key] = cell["texts"]

    stability = summarise.stability_scores(by_variant)
    control = summarise.select_control_group(stability)
    return {"stability": stability, "control_group": control}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("election", type=int,
                        help="election year, e.g. 2024")
    parser.add_argument("--manifestos", type=Path, default=DEFAULT_MANIFESTOS,
                        help="directory of {election}_{party}.txt files")
    parser.add_argument("--model", default="gpt-5",
                        help="model identifier passed to llm_client")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=None, help="override the registered temperature for this run")
    parser.add_argument("--variants", nargs="+", default=list(MAIN_VARIANTS), choices=list(VARIANTS),
                        help="summary variants to run (default: the main-series grid)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve files and count calls, make none")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature is not None:
        from . import llm_client
        llm_client.TEMPERATURE_OVERRIDE = args.temperature

    if not args.manifestos.is_dir():
        sys.exit(f"Not a directory: {args.manifestos}")

    expected = list(vote_shares(args.election))
    print(f"Election {args.election}: {len(expected)} parties "
          f"({', '.join(expected)})")

    texts, problems = load_manifestos(args.manifestos, args.election)
    for party_key, reason, detail in problems:
        print(f"  MISSING {party_key}: {reason} ({detail})", file=sys.stderr)

    if not texts:
        sys.exit("No usable manifestos; nothing to run.")

    counts = plan(texts, args.model, args.variants)
    print(f"\nPlan: {counts['manifestos']} manifestos x "
          f"{counts['variants']} variants x {counts['runs_per_variant']} runs")
    print(f"      {counts['summary_calls']} summary calls + "
          f"{counts['commitment_calls']} commitment calls "
          f"= {counts['total_calls']} total (cold cache)")

    if args.dry_run:
        print("\nDry run: no calls made.")
        for party_key, text in texts.items():
            print(f"  {party_key}: {len(text.split()):,} words, gate passed")
        return

    print()
    selections = run_election(args.election, texts, args.model, variants=args.variants)
    scored = score_stability(selections, args.election, args.model)

    top, minimal = scored["control_group"]
    print(f"\nControl group: {top} + {minimal}")
    print(f"Carried to Phase 3: {', '.join(summarise.carried_forward())}")

    report = {
        "election": args.election,
        "model": args.model,
        "plan": counts,
        "missing": [{"party": p, "reason": r, "detail": d}
                    for p, r, d in problems],
        "stability": scored["stability"],
        "control_group": list(scored["control_group"]),
        "selections": {party: {v: cell["selection"] for v, cell in variants.items()}
                       for party, variants in selections.items()},
    }
    path = write_report(args.output, f"phase1_{args.election}", report)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
