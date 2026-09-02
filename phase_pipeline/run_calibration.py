"""Calibration cell: the noise floor, from repeated runs of one fixed cell.

    python -m phase_pipeline.run_calibration --dry-run

Registered specification: 2024, explicit MFT prompt, neutral summaries,
Ipsos source, blind labels, five full passes. Each pass is 10 pairs x 2
orderings, so 100 calls. Nothing varies between passes, so any disagreement
is model stochasticity.

Produces two reference statistics, for two different purposes:

    verdict disagreement rate   for any rate comparison - calibration error,
                                invariance violation, framing flip rates
    Bradley-Terry score spread  for score- and ranking-level comparisons

Report both; do not use the second where the first is meant.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from . import bradley_terry, compare, profiles, validate
from .run_phase3 import load_phase1, load_phase2
from .vote_shares import vote_shares
from .prompts import N_RUNS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PHASE1 = REPO_ROOT / "outputs" / "phase1"
DEFAULT_PHASE2 = REPO_ROOT / "outputs" / "phase2"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "calibration"

# the registered cell
ELECTION = 2024
VARIANT = "neutral"
PROMPT_TYPE = "explicit_mft"
SOURCE = "ipsos"


def run_pass(election, parties, summaries, profile_block, model, repeat_index):
    """One full pass over every pair, in both orderings.

    Args:
        election (int): election year.
        parties (list): the five party keys.
        summaries (dict): {party: summary text}.
        profile_block (str): the rendered voter profile.
        model (str): model identifier.
        repeat_index (int): which pass this is; recorded in the metadata so
            _flip_rate can group by it.

    Returns:
        list: verdict dicts, two per pair.
    """
    verdicts = []
    for party_a, party_b in compare.all_pairs(parties):
        responses = compare.run_pair(
            election=str(election), party_a=party_a, party_b=party_b,
            text_a=summaries[party_a], text_b=summaries[party_b],
            prompt_type=PROMPT_TYPE,
            condition_tag=f"calibration_r{repeat_index}",
            scorer_model=model, run_index=repeat_index,
            profile_block=profile_block)
        for response in responses:
            verdict = compare.parse_verdict(response)
            verdict["meta"]["calibration_cell"] = True
            verdict["meta"]["repeat_index"] = repeat_index
            verdicts.append(verdict)
    return verdicts


def reference_statistics(passes, parties, shares):
    """The two numbers everything else is judged against.

    Args:
        passes (list): one list of verdicts per repeat.
        parties (list): the five party keys.
        shares (dict): actual vote shares, for the rank correlation.

    Returns:
        dict: the verdict disagreement rate and the score spread.
    """
    flat = [v for p in passes for v in p]

    # rate: how often the same comparison gets a different winner between
    # passes, when nothing has changed
    disagreement = validate._flip_rate(flat, "repeat_index")

    # scores: how far the Bradley-Terry estimate moves between passes
    per_pass = [bradley_terry.estimate_scores(p, parties) for p in passes]
    spread = {}
    for party in parties:
        values = [s[party] for s in per_pass]
        spread[party] = {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "range": max(values) - min(values),
        }

    rhos = [validate.spearman_validation(s, shares)["rho"] for s in per_pass]

    return {
        "verdict_disagreement": disagreement,
        "score_spread": spread,
        "max_score_sd": max(v["sd"] for v in spread.values()),
        "rho_per_pass": rhos,
        "rho_sd": statistics.stdev(rhos) if len(rhos) > 1 else 0.0,
        "confidence_distribution": validate.confidence_distribution(flat),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase1", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase2", type=Path, default=DEFAULT_PHASE2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--temperature", type=float, default=None, help="override the registered temperature for this run")
    parser.add_argument("--repeats", type=int, default=N_RUNS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature is not None:
        from . import llm_client
        llm_client.TEMPERATURE_OVERRIDE = args.temperature
    shares = vote_shares(ELECTION)
    parties = list(shares)

    try:
        summaries = load_phase1(args.phase1, ELECTION, (VARIANT,))[VARIANT]
        profile = load_phase2(args.phase2, ELECTION)[PROMPT_TYPE][SOURCE]
    except (FileNotFoundError, KeyError) as err:
        sys.exit(f"Cell inputs missing: {err}")

    calls = args.repeats * len(compare.all_pairs(parties)) * 2
    print(f"Calibration cell: {ELECTION}, {PROMPT_TYPE}, {VARIANT} "
          f"summaries, {SOURCE} source")
    print(f"      {args.repeats} passes x {len(compare.all_pairs(parties))} "
          f"pairs x 2 orderings = {calls} calls")

    if args.dry_run:
        print("\nDry run: no calls made.")
        print(f"  summaries: {len(summaries)}/{len(parties)}")
        print(f"  profile:   {'parsed' if not profile['parse_error'] else 'PARSE FAILED'}")
        return

    if profile["parse_error"]:
        sys.exit("The cell's profile did not parse; fix Phase 2 first.")

    profile_block = profiles.render_profile_block(profile)
    passes = []
    for i in range(args.repeats):
        passes.append(run_pass(ELECTION, parties, summaries, profile_block,
                               args.model, i))
        print(f"  pass {i + 1}/{args.repeats} done")

    stats = reference_statistics(passes, parties, shares)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "calibration_cell.json"
    path.write_text(json.dumps({
        "cell": {"election": ELECTION, "prompt_type": PROMPT_TYPE,
                 "variant": VARIANT, "source": SOURCE,
                 "model": args.model, "repeats": args.repeats},
        "reference_statistics": stats,
    }, indent=2, default=str), encoding="utf-8")

    rate = stats["verdict_disagreement"]["rate"]
    print(f"\nVerdict disagreement rate: "
          f"{rate:.1%}" if rate is not None else "\nVerdict disagreement: n/a")
    print(f"Largest score SD:          {stats['max_score_sd']:.3f}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
