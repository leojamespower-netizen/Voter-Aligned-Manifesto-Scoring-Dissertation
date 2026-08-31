"""Identification probes: can the model name the election from a block?

Run first, on non-anonymised summaries, to decide whether anonymisation
is worth building. Forced probes have chance 20% with five parties.
"""

from .llm_client import call_llm
from .profiles import _extract_json

# Set BEFORE seeing the rate (open item in the pre-analysis plan).
FAILURE_THRESHOLD = 0.45

# OPEN family: no candidate list, no domain framing, no election
# Matched to Phase 3 conditions. Do not add scaffolding here: naming the
# country, the document type, or the parties converts this into the forced
# probe and inflates the measured rate.

MINIMAL_PROMPT = """What is the following text, and where does it come from?

<text>
{text}
</text>

Respond with a single JSON object and nothing else, with these keys in this order:

{{
  "evidence": "<what in the text informed your answer>",
  "confidence": "high" | "medium" | "low",
  "identification": "<what you think this is and who produced it, as specifically as you can>"
}}"""

BIASED_PROMPT = """Below is a policy summary from a UK general election manifesto. Identify \
the party that produced it. You must choose one.

<text>
{text}
</text>

Respond with a single JSON object and nothing else, with these keys in this order:

{{
  "evidence": "<what in the text informed your choice>",
  "confidence": "high" | "medium" | "low",
  "party": one of {options}
}}"""

def biased_check(election, party, text, model, parties, variant='neutral'):
    prompt = BIASED_PROMPT.format(text=text, options=list(parties))
    key = f"probe_biased_{election}_{party}_{variant}_{model}"
    resp = call_llm(prompt, model=model, cache_key=key, subdir="probes")
    obj = _extract_json(resp.get("text", "")) or {}
    guess = str(obj.get("party", "")).strip().lower()

    correct = guess == party.lower()
    if not correct and aliases:
        correct = any(guess == str(a).strip().lower() for a in aliases)
    return {
        "family": "biased", "kind": "singleton", "election": election,
        "truth": party,
        "guess": guess, "correct": correct,
        "stated_confidence": obj.get("confidence"),  # diagnostic only
        "variant": variant, "model": model,
    }


# summarise probe results against chance and the registered threshold
def leakage_report(results, n_parties=5):
    accuracy = (sum(r["correct"] for r in results) / len(results)
                if results else None)

    # high-confidence-and-correct is the stronger contamination signal
    hc = [r for r in results
          if r["correct"]
          and str(r.get("stated_confidence", "")).lower() == "high"]

    return {
        "accuracy": accuracy,
        "chance": 1.0 / n_parties,
        "high_confidence_correct_rate": (len(hc) / len(results)
                                         if results else None),
        "threshold": FAILURE_THRESHOLD,
        "anonymisation_failed": (accuracy is not None
                                 and accuracy > FAILURE_THRESHOLD),
        "n": len(results),
    }

# OPEN family - spontaneous recognition, no scaffolding

# Registered grading rule for free-text identifications.

ALIASES = {
    # Extend per election. Lowercase, substring-matched.
    "lab": ("labour",),
    "con": ("conservative", "tory", "tories"),
    "ld": ("liberal democrat", "lib dem", "libdem"),
    "grn": ("green party", "greens"),
    "ref": ("reform uk", "reform party"),
}

FAMILY_TERMS = (
    "manifesto", "left", "right", "centre", "center", "progressive",
    "conservative-leaning", "party political", "election",
)


# apply the registered grading rule to one free-text identification
def grade_open(identification, truth, aliases=None):
    aliases = aliases or ALIASES
    text = (identification or "").lower()
    if not text.strip():
        return "NONE"
    if any(a in text for a in aliases.get(truth, (truth,))):
        return "EXACT"
    for party, alts in aliases.items():
        if party != truth and any(a in text for a in alts):
            return "WRONG"
    if any(term in text for term in FAMILY_TERMS):
        return "PARTIAL"
    return "NONE"


def minimal_check(election, party, text, model, variant='neutral'):
    prompt = MINIMAL_PROMPT.format(text=text)
    key = f"probe_minimal_{election}_{party}_{variant}_{model}"
    resp = call_llm(prompt, model=model, cache_key=key, subdir="probes")
    obj = _extract_json(resp.get("text", "")) or {}
    ident = obj.get("identification", "")
    grade = grade_open(ident, party)
    return {
        "family": "open", "kind": "singleton", "election": election,
        "truth": party, "identification": ident, "grade": grade,
        "correct": grade == "EXACT",
        "stated_confidence": obj.get("confidence"),  # diagnostic only
        "evidence": obj.get("evidence"),  # tells you WHAT leaked
        "variant": variant, "model": model,
    }


# report the open and forced families side by side
def dual_report(results, n_parties=5):
    def rate(family):
        rs = [r for r in results if r.get("family") == family]
        return (sum(r["correct"] for r in rs) / len(rs)) if rs else None

    minimal_rate, biased_rate = rate("minimal"), rate("biased")
    partials = [r for r in results
                if r.get("family") == "minimal"
                and "PARTIAL" in (r["grade"] if isinstance(r["grade"], tuple)
                                  else (r["grade"],))]

    return {
        "minimal": minimal_rate,
        "biased": biased_rate,
        "chance_biased": 1.0 / n_parties,
        "gap": (None if None in (minimal_rate, biased_rate)
                else biased_rate - minimal_rate),
        "partial_rate": len(partials) / len(results) if results else None,
        "threshold": FAILURE_THRESHOLD,
        # the decision rests on the minimal probe: it matches Phase 3
        "anonymisation_indicated": (minimal_rate is not None
                                    and minimal_rate > FAILURE_THRESHOLD),
    }
