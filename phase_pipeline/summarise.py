"""Phase 1: manifesto summarisation across six registered variants."""

from .llm_client import call_llm
from . import llm_client
import json

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

# A response that is too short for its source, or mostly copied from it, is
# a fragment, not a summary. Claude produced these for the
# 2005 Green manifesto, which ends in a credits page.
MIN_SUMMARY_WORDS = 150
MAX_COPIED_SHARE = 0.5


def looks_like_summary(summary, manifesto, min_words=MIN_SUMMARY_WORDS):
    return len(manifesto.split()) <= 1000 or len(summary.split()) >= min_words

# a rejected response is evidence about the model, as such it is retained in storage.
def keep_rejected(response, name):
    folder = llm_client.CACHE_DIR / "summaries_rejected"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")

# run all six variants x N_RUNS for one manifesto with one model
def summarise_manifesto(election, party, manifesto_text, model, party_name='',
                        replacement_list=None, variants=None):
    responses = []
    variants = variants or list(VARIANTS)
    for variant in variants:
        template = VARIANTS[variant]
        text = manifesto_text
        if variant in ANONYMISED_INPUT:
            text = anonymise(manifesto_text, replacement_list or [])
        prompt = (template.format(manifesto=text, party=party_name)
                  if variant in NEEDS_PARTY
                  else template.format(manifesto=text))
        for run in range(1, N_RUNS + 1):
            key = f"{election}_{party}_{variant}_{model}_run{run}"
            resp = call_llm(prompt, model=model, cache_key=key, subdir="summaries")
            for attempt in range(3):  # not a summary: keep it as evidence and ask again
                if looks_like_summary(resp["text"], text):
                    break
                print(f"  {key}: {len(resp['text'].split())} words, not a summary; re-requesting ({attempt + 1}/3)")
                keep_rejected(resp, f"{key}_rejected{attempt + 1}")
                resp = call_llm(prompt, model=model, cache_key=key, subdir="summaries", force_refresh=True)
            else:
                raise ValueError(f"{key}: no valid summary in 3 attempts; check the manifesto text")
            resp["rejected_attempts"] = attempt
            responses.append(resp)
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
