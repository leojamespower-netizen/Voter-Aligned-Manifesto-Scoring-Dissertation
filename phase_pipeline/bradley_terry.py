#Pairwise verdicts to a party ranking.

import choix


# bradley-Terry scores for one cell, via regularised lsr_pairwise with pre-agreed 0.1 alpha
def estimate_scores(verdicts, parties, alpha=0.1):
    idx = {p:k for k, p in enumerate(parties)}
    comparisons = []  # choix format:(winner_index, loser_index)
    skipped = 0
    for v in verdicts:
        if v.get("parse_error") or v["winner_party"] not in idx:
            skipped += 1
            continue
        a, b = v["meta"]["pair"]
        winner = v["winner_party"]
        loser = b if winner == a else a
        comparisons.append((idx[winner], idx[loser]))

    if not comparisons:
        raise ValueError("No usable verdicts supplied.")
    if skipped:
        print(f"[bradley_terry] skipped {skipped} unusable verdicts "
              f"of {len(verdicts)}")

    scores = choix.lsr_pairwise(len(parties), comparisons, alpha=alpha)
    return {p:float(scores[idx[p]]) for p in parties}


# parties ordered best-to-worst by Bradley-Terry score
def rank_parties(scores):
    return sorted(scores, key=scores.get, reverse=True)


# re-estimate across alphas; report whether the ranking is stable
def alpha_sensitivity(verdicts, parties, alphas=(0.01, 0.05, 0.1, 0.5, 1.0)):
    return {a: estimate_scores(verdicts, parties, alpha=a) for a in alphas}
