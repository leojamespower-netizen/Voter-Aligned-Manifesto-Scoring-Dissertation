"""Phase 3: pairwise manifesto comparison, each pair in both orderings."""

import itertools
import random

from . import prompts as P
from .llm_client import call_llm
from .profiles import _extract_json

MASTER_SEED = 20260101
# Pseudonym letters. The pilot asked for the winner out of "A" or "B".
# The problem with this is that two texts were often labelled TEXT_A and TEXT_B, in "B" and "A" respectively.
# So the LLM was not sure which A and B we were referring to, thereby leading to artificially compromised comparisons.
LABEL_LETTERS = "CDEFGHJK"
LABEL_SCHEME = "byname"  # in the comparison cache key, so pilot verdicts are never reused

# all unique unordered pairs (10 pairs for five parties)
def all_pairs(parties):
    return list(itertools.combinations(parties, 2))


# pseudonym mapping for one run
def run_mapping(parties, run_index, master_seed=MASTER_SEED):
    rng = random.Random(master_seed + run_index)
    labels = [f"TEXT_{c}" for c in LABEL_LETTERS][:len(parties)]
    rng.shuffle(labels)
    return dict(zip(parties, labels))


# render one comparison prompt
def build_compare_prompt(prompt_type, text_a, text_b, label_a, label_b,
                         profile_block='', source_description='',
                         voter_priority_data=''):
    tpl = P.COMPARE_PROMPTS[prompt_type]
    return tpl.format(
        mft_definitions=P.MFT_DEFINITIONS,
        bidirectionality_examples=P.BIDIRECTIONALITY_EXAMPLES,
        axis_definitions=P.AXIS_DEFINITIONS,
        profile=profile_block,
        source_description=source_description,
        voter_priority_data=voter_priority_data,
        label_a=label_a, label_b=label_b,
        text_a=text_a, text_b=text_b,
        ref_a=label_a, ref_b=label_b,  # the question names the texts by label
    )


# run ONE pair in BOTH orderings
def run_pair(election, party_a, party_b, text_a, text_b, prompt_type,
             condition_tag, scorer_model, labelled=False, run_index=0,
             mapping=None, profile_block='', source_description='',
             voter_priority_data=''):
    mapping = mapping or run_mapping([party_a, party_b], run_index)
    out = []
    orderings = {
        "orderA": (party_a, party_b, text_a, text_b),
        "orderB": (party_b, party_a, text_b, text_a),
    }
    for order, (slot_a, slot_b, ta, tb) in orderings.items():
        if labelled:
            label_a, label_b = slot_a, slot_b
        else:
            label_a, label_b = mapping[slot_a], mapping[slot_b]

        prompt = build_compare_prompt(
            prompt_type, ta, tb, label_a, label_b,
            profile_block=profile_block,
            source_description=source_description,
            voter_priority_data=voter_priority_data,
        )
        key = (f"{election}_{party_a}-v-{party_b}_{order}_{prompt_type}"
               f"_{condition_tag}_{'labelled' if labelled else 'blind'}"
               f"_{scorer_model}_{P.PROFILE_DESIGN}_{LABEL_SCHEME}_run{run_index}")  # design and label tags: see prompts.PROFILE_DESIGN, LABEL_SCHEME
        resp = call_llm(prompt, model=scorer_model, cache_key=key,
                        subdir="comparisons")
        # Written at call-construction time; the model never sees this.
        resp["meta"] = {
            "election": election, "pair": (party_a, party_b), "order": order,
            "slot_A": slot_a, "slot_B": slot_b,
            "label_A": label_a, "label_B": label_b,
            "prompt_type": prompt_type, "condition": condition_tag,
            "labelled": labelled, "scorer": scorer_model,
            "run_index": run_index,
        }
        out.append(resp)
    return out


# extract {winner_party, confidence, reasoning, evidence} from a response
def parse_verdict(response):
    meta = response.get("meta", {})
    obj = _extract_json(response.get("text", ""))
    if obj is None:
        return _fail(meta)

    named = str(obj.get("winner", "")).strip().upper()  # the label the model returned
    winner = (meta.get("slot_A") if named == meta.get("label_A")
              else meta.get("slot_B") if named == meta.get("label_B") else None)
    if winner is None:
        return _fail(meta)

    return {
        "winner_party": winner,
        "confidence": obj.get("confidence"),
        "reasoning": obj.get("reasoning"),
        "evidence": obj.get("evidence"),
        "parse_error": False,
        "meta": meta,
    }


def _fail(meta):
    return {"winner_party": None, "confidence": None, "reasoning": None,
            "evidence": None, "parse_error": True, "meta": meta}

# slot balance

# verify slot-A occupancy is balanced across parties
def slot_balance(responses, tolerance=0.05):
    from collections import Counter
    counts = Counter(r["meta"]["slot_A"] for r in responses)
    parties = {p for r in responses for p in r["meta"]["pair"]}
    expected = len(responses) / len(parties)
    imbalance = {p: (counts.get(p, 0) - expected) / expected for p in parties}
    return {
        "counts": dict(counts),
        "expected": expected,
        "max_deviation": max(abs(v) for v in imbalance.values()),
        "balanced": all(abs(v) <= tolerance for v in imbalance.values()),
        "deviations": imbalance,
    }
