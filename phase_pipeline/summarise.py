"""Phase 1: manifesto summarisation across six registered variants."""

from .llm_client import call_llm

from .prompts import N_RUNS  # registered count


from .prompts import (ANONYMISED_INPUT, CARRIED_FORWARD, NEEDS_PARTY,
                      VARIANTS)
from .commitments import extract_commitments, select_replicate
from .stability import variant_stability  # text-similarity SECONDARY


# strip party names/leader names from manifesto text for the anonymised variant
def anonymise(text, party_names):
    for name in party_names:
        text = text.replace(name, "[PARTY]")
    return text


# run all six variants x N_RUNS for one manifesto with one model
def summarise_manifesto(election, party, manifesto_text, model, party_name='',
                        replacement_list=None):
    responses = []
    for variant, template in VARIANTS.items():
        text = manifesto_text
        if variant in ANONYMISED_INPUT:
            text = anonymise(manifesto_text, replacement_list or [])
        prompt = (template.format(manifesto=text, party=party_name)
                  if variant in NEEDS_PARTY
                  else template.format(manifesto=text))
        for run in range(1, N_RUNS + 1):
            key = f"{election}_{party}_{variant}_{model}_run{run}"
            responses.append(
                call_llm(prompt, model=model, cache_key=key, subdir="summaries")
            )
    return responses


# ranks prompt variants by output stability
def stability_scores(summaries_by_cell, metric='cosine'):
    return {v: variant_stability(cells, metric)
            for v, cells in summaries_by_cell.items()}


def select_control_group(stability):
    """Most stable variant plus minimal; if the most stable is minimal, use
    the second most stable."""
    ranked = sorted(stability, key=lambda v: stability[v]["mean_stability"],
                    reverse=True)
    top = ranked[0] if ranked[0] != "minimal" else ranked[1]
    return top, "minimal"


def select_summary(election, party, variant, model, responses):
    """Select the representative replicate for one cell.

    Uses the commitment-level metric, not text similarity: Phase 3
    consumes policy content, not phrasing."""
    cell = f"summary:{election}_{party}_{variant}_{model}"
    lists = [extract_commitments(
                r["text"], model,
                cache_key=f"extract_{election}_{party}_{variant}_{model}_run{i+1}")
             for i, r in enumerate(responses)]
    return select_replicate(cell, responses, lists)


# variants that proceed to Phase 3
def carried_forward():
    return CARRIED_FORWARD


# quality gate for scanned-PDF detection
def extraction_ok(text, min_chars=5000):
    if len(text) < min_chars:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text)
    return alpha / max(len(text), 1) > 0.85
