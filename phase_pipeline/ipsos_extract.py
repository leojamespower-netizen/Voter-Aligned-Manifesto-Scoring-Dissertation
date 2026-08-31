"""Ipsos Issues Index reading to the text Phase 2 reads.

Not in the eight-slot BES structure: a single-question survey rendered in
a skeleton built for a different instrument would make the two sources
look more alike than they are.
"""

import ast
import re
from pathlib import Path

from .bes_extract import assert_blinded

# fieldwork month, election month, gap in weeks; only the gap is rendered
# because a month names the election and a relative distance does not
READINGS = {
    1997: ("April 1997", "May 1997", 4),
    2001: ("April 2001", "June 2001", 9),
    2005: ("March 2005", "May 2005", 6),
    2010: ("April 2010", "May 2010", 4),
    2015: ("April 2015", "May 2015", 4),
    2017: ("May 2017", "June 2017", 4),
    2019: ("November 2019", "December 2019", 4),
    2024: ("June 2024", "July 2024", 4),
}

# exact label match, never substring
IPSOS_NON_ANSWERS = {
    "Don't know",
}

# read the dict literal only; the file's normalisation step is not run
def parse_file(path):
    text = Path(path).read_text(encoding="utf-8")
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        raise ValueError(f"no category dictionary found in {path}")
    issues = ast.literal_eval(match.group(0))
    if not issues:
        raise ValueError(f"category dictionary in {path} is empty")
    return issues


def prepare_ipsos(cycle, data_dir):
    """Read one Ipsos reading into the text a prompt receives.

    Args:
        cycle (int): election year.
        data_dir (Path): directory holding the Ipsos .txt files.

    Returns:
        tuple: the text, and the metadata withheld from the model.
    """
    path = Path(data_dir) / f"IPSOS_INDEX_{cycle}.txt"
    raw = parse_file(path)

    # "Other" is retained: a respondent who named something the frame could
    # not hold still named something.
    issues = {k: v for k, v in raw.items()
              if k not in IPSOS_NON_ANSWERS and v > 0}
    ordered = sorted(issues.items(), key=lambda kv: -kv[1])

    fieldwork, election_month, weeks_before = READINGS[cycle]

    width = max(len(label) for label, _ in ordered) + 2
    lines = [
        "Issue salience, from a monthly national survey of British adults.",
        "",
        "Question: what is the most important issue facing Britain, and what "
        "other issues are important? Unprompted, so respondents supply their "
        "own answers, and multi-response, so one person may name several. "
        "Percentages are the share of respondents mentioning each issue and "
        f"therefore do not sum to 100. All {len(issues)} published categories "
        "are listed, in the survey's own wording.",
        "",
    ]
    lines += [f"  {label:<{width}}{pct:5.1f}%" for label, pct in ordered]
    lines += [
        "",
        "This instrument records that an issue was raised, not what "
        "respondents wanted done about it. It carries no measure of "
        "direction: nothing here indicates which way opinion leaned on any "
        "of these issues.",
    ]

    text = "\n".join(lines)
    assert_blinded(text)

    metadata = {
        "cycle": cycle,
        "study": "Ipsos Issues Index",
        "instrument": "Ipsos Issues Index (monthly, continuous since 1974)",
        "fieldwork": f"{fieldwork}, preceding the {election_month} election",
        "n_categories": len(issues),
        "weeks_before_poll": weeks_before,
    }
    return text, metadata
