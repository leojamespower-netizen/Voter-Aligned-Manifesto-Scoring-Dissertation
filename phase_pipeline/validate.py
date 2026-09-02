"""Validation: correlations, noise floors, and the four error types."""

from collections import defaultdict

from scipy.stats import spearmanr

# every design dimension a verdict carries
_DESIGN_KEYS = ("variant", "prompt_type", "source", "labelled", "run_index")

# spearman rho between BT scores and vote shares for one election
def spearman_validation(scores, vote_shares):
    parties = sorted(set(scores) & set(vote_shares))
    rho, p = spearmanr([scores[p_] for p_ in parties],
                       [vote_shares[p_] for p_ in parties])
    return {"rho": float(rho), "pvalue": float(p), "n": len(parties)}


# secondary criterion: does the top-scoring party win the most votes?
def binary_winner(scores, vote_shares):
    return max(scores, key=scores.get) == max(vote_shares, key=vote_shares.get)

# error types

# verdict reversal rate across orderings, against the 48.4% benchmark
def positional_error(verdicts):
    cells = defaultdict(list)
    for v in verdicts:
        if v.get("parse_error"):
            continue
        m = v["meta"]
        # Cell keys must match the metadata written by compare.run_pair().
        cell = (m["election"], tuple(m["pair"]), m["condition"],
                m.get("prompt_type"), m.get("labelled"), m["scorer"],
                m.get("run_index"))
        cells[cell].append(v["winner_party"])
    pairs = [ws for ws in cells.values() if len(ws) == 2]
    if not pairs:
        raise ValueError("No complete ordering pairs found.")
    reversals = sum(1 for a, b in pairs if a != b)
    return reversals / len(pairs)


# hold every dimension constant except the excluded one, or verdicts from
# different arms and sources collapse into one group and the rate is nothing
def _comparison_key(meta, exclude=None):
    key = [meta["election"], tuple(sorted(meta["pair"])), meta["scorer"]]
    for k in _DESIGN_KEYS:
        if k != exclude:
            key.append(meta.get(k))
    return tuple(key)


def _flip_rate(verdicts, varying):
    """Disagreement rate across one condition dimension.

    Groups verdicts by the comparison being made, then within each group asks
    how often the winner changes as `varying` changes. Groups with only one
    value of `varying` contribute nothing.

    Args:
        verdicts (list): parsed verdicts.
        varying (str): the meta key allowed to differ within a group.

    Returns:
        dict: the flip rate, the groups it was computed over, and the
        distinct values of `varying` that were present.
    """
    groups = defaultdict(dict)
    values = set()
    for v in verdicts:
        if v.get("parse_error"):
            continue
        m = v["meta"]
        value = m.get(varying)
        if value is None:
            continue
        values.add(value)
        # one winner per (comparison, value); later runs overwrite, so
        # positional pairs are collapsed first by taking the slot-A ordering
        if m.get("slot_A") == m["pair"][0]:
            groups[_comparison_key(m, exclude=varying)][value] = v["winner_party"]

    comparable = [g for g in groups.values() if len(g) > 1]
    if not comparable:
        return {"rate": None, "n_comparisons": 0, "values": sorted(values)}

    flipped = sum(1 for g in comparable if len(set(g.values())) > 1)
    return {
        "rate": flipped / len(comparable),
        "n_comparisons": len(comparable),
        "values": sorted(values),
    }


def invariance_violation(verdicts):
    """Disagreement where the meaning is identical, so divergence is noise.

    Three comparisons qualify: the positional swap (same text, different
    slot), distorted against minimal summaries (same semantics, articles
    deleted), and repeated runs of the calibration cell (nothing varies).

    This is the only place the published prompt-sensitivity benchmarks apply,
    because Sclar et al. and Salinas and Morstatter both measure
    meaning-preserving perturbation.

    Args:
        verdicts (list): parsed verdicts.

    Returns:
        dict: a rate for each of the three comparisons, and the highest of
        them, which is the floor any manipulation must clear.
    """
    try:
        positional = positional_error(verdicts)
    except ValueError:
        # no pair was run in both orderings, so this floor is unavailable
        positional = None
    surface = _flip_rate([v for v in verdicts
                          if v.get("meta", {}).get("variant")
                          in ("minimal", "distorted")], "variant")
    repeats = _flip_rate([v for v in verdicts
                          if v.get("meta", {}).get("calibration_cell")],
                         "repeat_index")

    rates = [r for r in (positional, surface["rate"], repeats["rate"])
             if r is not None]
    return {
        "positional_swap": positional,
        "distorted_vs_minimal": surface,
        "calibration_repeats": repeats,
        "floor": max(rates) if rates else None,
    }


def framing_sensitivity(verdicts):
    """Divergence across conditions that differ in meaning.

    These should diverge: that is the manipulation working. What matters is
    the size relative to the invariance floor, not the rate itself.

    neutral against framing_preserving is NOT an invariance control - one
    instructs suppression of ideological framing and the other instructs
    preservation, which is the core framing manipulation. The invariance
    control among summary variants is distorted against minimal.

    Args:
        verdicts (list): parsed verdicts.

    Returns:
        dict: a rate per manipulation, plus the registered ordering check.
    """
    framing_axis = _flip_rate(
        [v for v in verdicts if v.get("meta", {}).get("variant")
         in ("minimal", "neutral", "framing_preserving")], "variant")
    party_cue = _flip_rate(
        [v for v in verdicts if v.get("meta", {}).get("variant")
         in ("framed", "anonymised")], "variant")
    framework_arm = _flip_rate(verdicts, "prompt_type")
    source = _flip_rate(verdicts, "source")

    floor = invariance_violation(verdicts)["floor"]
    ordering = None
    if None not in (floor, framing_axis["rate"], party_cue["rate"]):
        # registered prediction: invariance < framing axis < party cue
        ordering = floor < framing_axis["rate"] < party_cue["rate"]

    return {
        "framing_axis": framing_axis,
        "party_cue": party_cue,
        "framework_arm": framework_arm,
        "source_condition": source,
        "invariance_floor": floor,
        "ordering_holds": ordering,
    }


def confidence_distribution(verdicts):
    """How often each confidence label was used.

    Registered alongside calibration_error: if the model returns one label
    almost uniformly, the calibration comparison is underpowered and that
    should be reported rather than a rate computed from a handful of cases.
    """
    counts = defaultdict(int)
    for v in verdicts:
        if not v.get("parse_error"):
            counts[str(v.get("confidence", "")).strip().lower()] += 1
    total = sum(counts.values())
    return {
        "counts": dict(counts),
        "proportions": {k: n / total for k, n in counts.items()} if total else {},
        "n": total,
    }


def calibration_error(verdicts):
    """Whether stated confidence tracks positional consistency.

    Splits verdicts by confidence label and computes the positional reversal
    rate within each. The registered expectation is that high-confidence
    verdicts are no more consistent than low-confidence ones, judged against
    the calibration cell's verdict disagreement rate - a rate, in the same
    units as the quantity being compared.

    Args:
        verdicts (list): parsed verdicts.

    Returns:
        dict: reversal rate per confidence label, the gap between the highest
        and lowest label, and the label distribution.
    """
    by_label = defaultdict(list)
    for v in verdicts:
        if v.get("parse_error"):
            continue
        by_label[str(v.get("confidence", "")).strip().lower()].append(v)

    rates = {}
    for label, subset in by_label.items():
        try:
            rates[label] = positional_error(subset)
        except ValueError:
            rates[label] = None

    usable = {k: r for k, r in rates.items() if r is not None}
    gap = (max(usable.values()) - min(usable.values())
           if len(usable) > 1 else None)

    return {
        "reversal_by_confidence": rates,
        "gap": gap,
        "distribution": confidence_distribution(verdicts),
    }


# reporting requirements

# proportion of verdicts excluded from estimation
def parse_failure_rate(verdicts):
    if not verdicts:
        return 0.0
    return sum(1 for v in verdicts if v.get("parse_error")) / len(verdicts)


# bundle Bradley-Terry scores with the numbers that make them readable
def scores_with_noise_context(scores, verdicts):
    return {
        "scores": scores,
        "swap_inconsistency": positional_error(verdicts),
        "parse_failure_rate": parse_failure_rate(verdicts),
        "n_verdicts": len(verdicts),
    }


# is the profile load-bearing?
def adversarial_ablation(scores_real, scores_adversarial):
    rank_real = sorted(scores_real, key=scores_real.get, reverse=True)
    rank_adv = sorted(scores_adversarial, key=scores_adversarial.get,
                      reverse=True)
    return {
        "ranking_real": rank_real,
        "ranking_adversarial": rank_adv,
        "ranking_changed": rank_real != rank_adv,
        "profile_is_load_bearing": rank_real != rank_adv,
    }

# Null baselines and benchmarks

def permutation_null(vote_shares, n_iter=10000, seed=20260101):
    """Distribution of rho from random ranking.

    With five parties, random clears rho 0.80 about 5% of the time."""
    import random
    from scipy.stats import spearmanr

    parties = sorted(vote_shares)
    truth = [vote_shares[p] for p in parties]
    rng = random.Random(seed)

    draws = []
    for _ in range(n_iter):
        shuffled = parties[:]
        rng.shuffle(shuffled)
        fake = [shuffled.index(p) for p in parties]  # arbitrary random scores
        draws.append(spearmanr(fake, truth).statistic)

    draws.sort()
    def q(p):
        return draws[min(int(p * len(draws)), len(draws) - 1)]
    return {
        "n_iter": n_iter, "n_parties": len(parties),
        "mean": sum(draws) / len(draws),
        "p50": q(0.50), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99),
        "critical_rho_95": q(0.95),
        "_draws": draws,
    }


# where an observed rho sits in the random-ranking null distribution
def rho_percentile(rho, null):
    draws = null["_draws"]
    below = sum(1 for d in draws if d < rho)
    return below / len(draws)


def benchmark_against_polling(bt_scores, polling, vote_shares):
    """Pipeline versus pre-election polling on the same target.

    The gap between the two is what gets reported."""
    from scipy.stats import spearmanr
    parties = sorted(set(bt_scores) & set(polling) & set(vote_shares))
    truth = [vote_shares[p] for p in parties]
    rho_pipeline = spearmanr([bt_scores[p] for p in parties], truth).statistic
    rho_polling = spearmanr([polling[p] for p in parties], truth).statistic
    return {
        "rho_pipeline": float(rho_pipeline),
        "rho_polling": float(rho_polling),
        "gap": float(rho_polling - rho_pipeline),
        "pipeline_matches_or_beats": rho_pipeline >= rho_polling,
        "n_parties": len(parties),
    }


def profile_value_added(bt_with_profile, bt_baseline, vote_shares):
    """Does the Phase 2 profile contribute anything?

    Run Phase 3 without it and compare. If scores barely move, the
    profile is decorative."""
    from scipy.stats import spearmanr
    parties = sorted(set(bt_with_profile) & set(bt_baseline) & set(vote_shares))
    truth = [vote_shares[p] for p in parties]
    rho_profile = spearmanr([bt_with_profile[p] for p in parties], truth).statistic
    rho_base = spearmanr([bt_baseline[p] for p in parties], truth).statistic
    return {
        "rho_with_profile": float(rho_profile),
        "rho_no_profile": float(rho_base),
        "improvement": float(rho_profile - rho_base),
        "profile_helps": rho_profile > rho_base,
    }

# Per-party accuracy and its stability across elections

# signed rank error per party per election, plus per-party summaries
def party_error_profile(bt_by_election, shares_by_election):
    from statistics import mean, pstdev

    per_party = {}
    for elec, scores in bt_by_election.items():
        shares = shares_by_election.get(elec, {})
        parties = sorted(set(scores) & set(shares))
        pred = sorted(parties, key=lambda p: scores[p], reverse=True)
        actual = sorted(parties, key=lambda p: shares[p], reverse=True)
        for p in parties:
            per_party.setdefault(p, {})[elec] = pred.index(p) - actual.index(p)

    summary = {}
    for p, errs in per_party.items():
        vals = list(errs.values())
        summary[p] = {
            "mean_signed_error": mean(vals),
            "sd_across_elections": pstdev(vals) if len(vals) > 1 else 0.0,
            "n_elections": len(vals),
            "per_election": errs,
        }
    return summary


def directional_error(bt_by_election, shares_by_election, party_families):
    """Mean signed rank error per party family.

    Positive means the pipeline ranked a party worse than voters did.
    Tests the left-leaning bias in Rutinowski et al. (2024)."""
    from statistics import mean, pstdev

    profile = party_error_profile(bt_by_election, shares_by_election)
    by_family = {}
    for party, stats in profile.items():
        fam = party_families.get(party)
        if fam:
            by_family.setdefault(fam, []).append(stats["mean_signed_error"])

    return {
        "by_family": {
            f: {"mean_signed_error": mean(v),
                "sd": pstdev(v) if len(v) > 1 else 0.0,
                "n_parties": len(v)}
            for f, v in by_family.items()
        },
        "by_party": profile,
    }


# election-wide consistency, and how it compares to polling's
def consistency_report(bt_by_election, shares_by_election,
                       polling_by_election=None):
    from statistics import mean, pstdev
    from scipy.stats import spearmanr

    def rho(scores, shares):
        parties = sorted(set(scores) & set(shares))
        return float(spearmanr([scores[p] for p in parties],
                               [shares[p] for p in parties]).statistic)

    per_elec = {e: rho(s, shares_by_election[e])
                for e, s in bt_by_election.items()
                if e in shares_by_election}
    vals = list(per_elec.values())
    out = {
        "rho_per_election": per_elec,
        "mean_rho": mean(vals),
        "sd_rho": pstdev(vals) if len(vals) > 1 else 0.0,
        "worst_election": min(per_elec, key=per_elec.get),
        "best_election": max(per_elec, key=per_elec.get),
    }
    if polling_by_election:
        poll_rho = {e: rho(p, shares_by_election[e])
                    for e, p in polling_by_election.items()
                    if e in shares_by_election}
        pv = list(poll_rho.values())
        out["polling_mean_rho"] = mean(pv)
        out["polling_sd_rho"] = pstdev(pv) if len(pv) > 1 else 0.0
        out["variability_ratio"] = (out["sd_rho"] / out["polling_sd_rho"]
                                    if out["polling_sd_rho"] else None)
    return out
