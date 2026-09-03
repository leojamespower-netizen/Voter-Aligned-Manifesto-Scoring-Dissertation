"""Phase 2: voter profile construction from survey blocks."""

import json

from . import prompts as P
from .prompts import N_RUNS as N_PROFILE_RUNS  # registered replication count
from .llm_client import call_llm
from .stability import select_medoid


# return (source_description, voter_priority_data) for a condition
def build_source_block(source, ipsos, bes):
    ipsos_tag = (
        '<source type="survey" instrument="Ipsos Issues Index" '
        'measurement="unprompted salience; multi-response" '
        'limitation="records salience only; direction not recoverable">\n'
        f"{ipsos}\n</source>"
    )
    bes_tag = (
        '<source type="survey" instrument="BES pre-election wave" '
        'measurement="structured attitudinal items; weighted" '
        'weighting="applied">\n'
        f"{bes}\n</source>"
    )
    if source == "ipsos":
        return "an unprompted issue-salience survey", ipsos_tag
    if source == "bes":
        return "a structured attitudinal survey", bes_tag
    if source == "both":
        return ("two surveys of the British electorate",
                f"{ipsos_tag}\n\n{bes_tag}")
    raise ValueError(f"unknown source condition: {source}")


def build_profile_prompt(arm, source, ipsos, bes):
    desc, data = build_source_block(source, ipsos, bes)
    return P.PROFILE_PROMPTS[arm].format(
        mft_definitions=P.MFT_DEFINITIONS,
        bidirectionality_examples=P.BIDIRECTIONALITY_EXAMPLES,
        axis_definitions=P.AXIS_DEFINITIONS,
        source_description=desc,
        voter_priority_data=data,
    )


# run one (arm, source, model) cell n_runs times
def generate_profiles(election, arm, source, model, ipsos, bes,
                      n_runs=N_PROFILE_RUNS):
    prompt = build_profile_prompt(arm, source, ipsos, bes)
    out = []
    for run in range(1, n_runs + 1):
        key = f"{election}_{arm}_{source}_{model}_{P.PROFILE_DESIGN}_run{run}"  # design tag keeps pilot and main-series files apart
        resp = call_llm(prompt, model=model, cache_key=key, subdir="profiles")
        resp["meta"] = {"election": election, "arm": arm, "source": source,
                        "model": model, "run": run}
        out.append(resp)
    return out


# medoid-select the representative profile for one cell and record it
def select_profile(election, arm, source, model, responses):
    cell = f"profile:{election}_{arm}_{source}_{model}_{P.PROFILE_DESIGN}"
    return select_medoid(cell, responses)

# parsing

# extract the profile from a raw response
def parse_profile(response, arm):
    unit_key = P.PROFILE_UNIT[arm]
    unit_name = "foundation" if unit_key == "foundations" else "pole"
    expected = set(P.FOUNDATIONS if unit_key == "foundations" else P.POLES)

    obj = _extract_json(response.get("text", ""))
    if obj is None or unit_key not in obj:
        return {"parse_error": True, "weights": {}, "weight_sum": None,
                "justification": None,
                "meta": response.get("meta")}

    weights = {}
    for row in obj[unit_key]:
        name = str(row.get(unit_name, "")).strip().lower()
        if name in expected and isinstance(row.get("weight"), (int, float)):
            weights[name] = float(row["weight"])

    return {
        "parse_error": len(weights) != len(expected), # every unit now needs a weight
        "weights": weights,
        "weight_sum": sum(weights.values()) if weights else None,  # recorded, not corrected
        "justification": obj.get("justification"),
        "meta": response.get("meta"),
    }


# render a parsed profile for insertion into a Phase 3 prompt
def render_profile_block(parsed):
    lines = [f"  {unit}: {w:.2f}" for unit, w in parsed["weights"].items()]
    return "<profile>\n" + "\n".join(lines) + "\n</profile>"


# first balanced JSON object in `text`, tolerating fences and prose
def _extract_json(text):
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
