"""Phase 3: takes manifesto summaries and profiles in and pumps party rankings and validation out.

    python -m phase_pipeline.run_phase3 2024 --dry-run

Every pair of parties is judged in both orderings under pseudonyms, once per
design cell. Bradley-Terry converts the verdicts into a score per party, which is
validated against that election's vote shares, and polling.

--variants, --arms and --sources default to the main-series grid registered in
prompts.py. Naming them explicitly recovers the full grid the 2024 calibration
cycle ran.
"""

import argparse
import json
import sys
from pathlib import Path

from . import bradley_terry, compare, profiles, validate
from .report import write_report
from .prompts import CARRIED_FORWARD, COMPARE_PROMPTS, MAIN_ARMS, MAIN_SOURCES, MAIN_VARIANTS, NEEDS_PROFILE
from .vote_shares import vote_shares

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PHASE1 = REPO_ROOT / "outputs" / "phase1"
DEFAULT_PHASE2 = REPO_ROOT / "outputs" / "phase2"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "phase3"

PROMPT_TYPES = tuple(COMPARE_PROMPTS)



def load_phase1(phase1_dir, election, variants):
    """Reads the summary for every party and variant.

    Args:
        phase1_dir (Path): directory of phase1_{election}.json reports.
        election (int): election year.
        variants (tuple): summary variants to carry forward.

    Returns:
        dict: {variant: {party: summary text}}.
    """
    path = phase1_dir / f"phase1_{election}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing. Run run_phase1 first.")

    report = json.loads(path.read_text(encoding="utf-8"))
    summaries = {v: {} for v in variants}
    for party, cells in report["selections"].items():
        for variant, selection in cells.items():
            if variant in summaries:
                summaries[variant][party] = selection["response"]["text"]
    return summaries


def load_phase2(phase2_dir, election):
    """Reads the profile for every arm and source.

    Returns:
        dict: {arm: {source: parsed profile}}.
    """
    path = phase2_dir / f"phase2_{election}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing. Run run_phase2 first.")

    report = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for arm, sources in report["selections"].items():
        out[arm] = {}
        for source, selection in sources.items():
            parsed = profiles.parse_profile(selection["response"], arm)
            out[arm][source] = parsed
    return out


def cells(variants, arms, sources):
    """Every combination to run.

    The baseline prompt takes no profile, so it runs once per variant rather
    than once per source.
    """
    out = []
    for variant in variants:
        for prompt_type in arms:
            if prompt_type in NEEDS_PROFILE:
                out.extend((variant, prompt_type, s) for s in sources)
            else:
                out.append((variant, prompt_type, None))
    return out


# call count before anything is spent
def plan(parties, design, model):
    pairs = len(compare.all_pairs(parties))
    return {
        "model": model,
        "parties": len(parties),
        "pairs": pairs,
        "cells": len(design),
        "calls_per_cell": pairs * 2,          # both orderings
        "total_calls": len(design) * pairs * 2,
    }


def run_cell(election, parties, summaries, profile, variant, prompt_type,
             source, model, run_index):
    """Runs every pair for one design cell, in both orderings.

    Returns:
        list: verdict dicts, two per pair.
    """
    condition_tag = f"{variant}_{prompt_type}_{source or 'noprofile'}"
    mapping = compare.run_mapping(parties, run_index)
    profile_block = profiles.render_profile_block(profile) if profile else ""

    verdicts = []
    for party_a, party_b in compare.all_pairs(parties):
        responses = compare.run_pair(
            election=str(election), party_a=party_a, party_b=party_b,
            text_a=summaries[party_a], text_b=summaries[party_b],
            prompt_type=prompt_type, condition_tag=condition_tag,
            scorer_model=model, run_index=run_index, mapping=mapping,
            profile_block=profile_block)
        for r in responses:
            verdict = compare.parse_verdict(r)
            # the design cell, so cross-condition analysis can group on it
            verdict["meta"]["variant"] = variant
            verdict["meta"]["source"] = source or "noprofile"
            verdicts.append(verdict)
    return verdicts


# bradley-Terry scores and validation for one cell
def score_cell(verdicts, parties, shares):
    scores = bradley_terry.estimate_scores(verdicts, parties)
    null = validate.permutation_null(shares)
    rho = validate.spearman_validation(scores, shares)

    return {
        "scores": scores,
        "ranking": bradley_terry.rank_parties(scores),
        "spearman": rho,
        "null_percentile": validate.rho_percentile(rho["rho"], null),
        "binary_winner": validate.binary_winner(scores, shares),
        "positional_error": validate.positional_error(verdicts),
        "parse_failure_rate": validate.parse_failure_rate(verdicts),
        "slot_balance": compare.slot_balance(verdicts),
        "alpha_sensitivity": bradley_terry.alpha_sensitivity(verdicts, parties),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("election", type=int)
    parser.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--variants", nargs="+", default=list(MAIN_VARIANTS), choices=CARRIED_FORWARD,
                        help="summary variants to run (default: the main-series grid)")
    parser.add_argument("--arms", nargs="+", default=["baseline", *MAIN_ARMS], choices=list(PROMPT_TYPES),
                        help="comparison prompt types to run")
    parser.add_argument("--sources", nargs="+", default=list(MAIN_SOURCES), choices=["ipsos", "bes", "both"],
                        help="profile source conditions to run")
    parser.add_argument("--run-index", type=int, default=0,
                        help="seeds the pseudonym mapping")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature is not None:
        from . import llm_client 
        llm_client.TEMPERATURE_OVERRIDE = args.temperature


    shares = vote_shares(args.election)
    parties = list(shares)

    try:
        profs = load_phase2(args.phase2, args.election)
        summaries = load_phase1(args.phase1, args.election, args.variants)
    except FileNotFoundError as err:
        sys.exit(str(err))

    design = cells(args.variants, args.arms, args.sources)
    counts = plan(parties, design, args.model)

    print(f"Election {args.election}: {counts['parties']} parties, "
          f"{counts['pairs']} pairs")
    print(f"      {counts['cells']} cells x {counts['calls_per_cell']} calls "
          f"= {counts['total_calls']} total (cold cache)")

    if args.dry_run:
        print("\nDry run: no calls made.")
        for variant in args.variants:
            have = len(summaries.get(variant, {}))
            print(f"  {variant}: {have}/{len(parties)} summaries")
        for arm in args.arms:
            if arm in NEEDS_PROFILE:
                ready = sum(1 for s in args.sources
                            if (profs.get(arm, {}).get(s) or {}).get(
                                "parse_error") is False)
                print(f"  {arm}: {ready}/{len(args.sources)} profiles parsed")
        return

    print()
    results, all_verdicts = {}, []
    for variant, prompt_type, source in design:
        key = f"{variant}/{prompt_type}/{source or 'noprofile'}"
        profile = profs.get(prompt_type, {}).get(source) if source else None
        if prompt_type in NEEDS_PROFILE and (
                profile is None or profile["parse_error"]):
            print(f"  {key}: no parsed profile, skipped", file=sys.stderr)
            continue

        verdicts = run_cell(args.election, parties, summaries[variant],
                            profile, variant, prompt_type, source,
                            args.model, args.run_index)
        results[key] = score_cell(verdicts, parties, shares)
        # kept so run_analysis can pool them; reasoning text is dropped
        all_verdicts.extend({
            "winner_party": v["winner_party"],
            "confidence": v.get("confidence"),
            "parse_error": v.get("parse_error", False),
            "meta": v["meta"],
        } for v in verdicts)
        rho = results[key]["spearman"]["rho"]
        pct = results[key]["null_percentile"]
        print(f"  {key}: rho {rho:+.2f} (null percentile {pct:.0f})")

    report = {
        "election": args.election,
        "model": args.model,
        "text_source": "summary",
        "plan": counts,
        "vote_shares": shares,
        "cells": results,
        "verdicts": all_verdicts,
    }
    path = write_report(args.output, f"phase3_{args.election}_summary", report)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
