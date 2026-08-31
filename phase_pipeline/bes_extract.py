"""Converts the raw BES data from 1997 to 2010 into the predefined 8 tiers
with the original weighting of survey answers preserved."""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOT_MEASURED = "NOT MEASURED in this survey cycle."

SCALE_0_10 = (0, 10)          # 0-10 self-placement scales
FIVE_POINT = (1, 5)           # approve/disapprove and agree/disagree items
LFTRIGHA_RANGE = (1, 5)       # 1997 derived left-right, 9 is missing
TAXSOCA_RANGE = (1, 2)        # 1997 two-way choice, 8 and 9 are non-answers
AQ2_RANGE = (1, 14)           # 2001 frame, 88/98/99 are non-answers

# categories naming a party or leader, dropped so the record cannot identify
# the election (0.30% of 1997, 0.00% of 2010)
BLINDING_EXCLUSIONS = {
    1997: {192, 193, 194},
    2001: set(),
    2005: set(),
    2010: {47, 48},
}

# non-answers are excluded by numeric code, never by matching label text
# ("na" is a substring of "National Health Service")

# scale items carry a mean; category items carry percentages
class Reading:

    def __init__(self, description, n, mean=None, categories=None):
        self.description = description
        self.n = n
        self.mean = mean                                  # scale items only
        self.categories = categories if categories is not None else []


# metadata fields are recorded but never rendered (blinding)
class ElectionData:

    def __init__(self, cycle, study, instrument, fieldwork, mode,
                 sample_basis, weight_note, salience_question,
                 salience_frame, salience, n_salience, notes, left_right=None,
                 economic=None, social=None, europe=None):
        # metadata: recorded, never rendered
        self.cycle = cycle
        self.study = study
        self.instrument = instrument
        self.fieldwork = fieldwork

        # model-visible
        self.mode = mode
        self.sample_basis = sample_basis
        self.weight_note = weight_note
        self.salience_question = salience_question
        self.salience_frame = salience_frame
        self.salience = salience
        self.n_salience = n_salience
        self.notes = notes
        self.left_right = left_right
        self.economic = economic
        self.social = social
        self.europe = europe


def weighted_percentages(df, var, weight, labels=None, valid_range=None,
                         exclude_codes=None, top=None):
    """Weighted share of respondents per answer category.

    Args:
        df (DataFrame): the survey data.
        var (str): the variable holding the answer codes.
        weight (str): the survey weight variable.
        labels (dict): code to label, from the file or typed from the codebook.
        valid_range (tuple): lowest and highest substantive code.
        exclude_codes (set): non-answer codes to drop.
        top (int): keep only the n largest, for console inspection.

    Returns:
        tuple: [(label, percentage)] sorted high to low, and the valid n.
    """
    valid = df[(df[var] > 0) & (df[weight] > 0)].copy()
    if valid_range is not None:
        valid = valid[valid[var].between(*valid_range)]
    if exclude_codes:
        valid = valid[~valid[var].isin(exclude_codes)]

    if labels is not None:
        valid["_key"] = valid[var].map(lambda c: labels.get(float(c), str(c)))
    else:
        valid["_key"] = valid[var].astype(str)

    n = len(valid)
    shares = valid.groupby("_key")[weight].sum()
    shares = shares.div(shares.sum()).mul(100).sort_values(ascending=False)
    if top:
        shares = shares.head(top)
    return [(str(k), round(float(v), 1)) for k, v in shares.items()], n


def weighted_mean(df, var, weight, valid_range):
    """Weighted mean of a scale item.

    valid_range is required rather than defaulted because BES stores
    don't-know as a high code on the same variable (8, 9, 99, 9999), so a
    mean taken without a ceiling averages refusals in.

    Args:
        df (DataFrame): the survey data.
        var (str): the scale variable.
        weight (str): the survey weight variable.
        valid_range (tuple): lowest and highest substantive value.

    Returns:
        tuple: the weighted mean and the valid n.
    """
    valid = df[df[var].between(*valid_range) & (df[weight] > 0)]
    if valid.empty:
        return float("nan"), 0
    mean = (valid[var] * valid[weight]).sum() / valid[weight].sum()
    return round(float(mean), 2), len(valid)


def _read(path, columns):
    import pyreadstat
    return pyreadstat.read_dta(str(path), usecols=columns)


def prepare_1997(data_dir):
    """1997, SN 3890. Salience wave C, attitudes wave A ten months earlier.

    Not pre-weighted. taxsoca is a split ballot and a two-way choice."""
    df, meta = _read(data_dir / "panel97u.dta",
                     ["twois1cc", "lftrigha", "taxsoca", "wtera"])
    W = "wtera"

    salience, n_sal = weighted_percentages(
        df, "twois1cc", W, labels=meta.variable_value_labels["twois1cc"],
        # 997 uncodeable, 998 don't know, 999 refusal/no answer.
        exclude_codes={997, 998, 999}.union(BLINDING_EXCLUSIONS[1997]))

    # 1-5 derived scale. Code 9 means "missing values" and must be excluded,
    # hence the ceiling of 5.
    lr_mean, lr_n = weighted_mean(df, "lftrigha", W, LFTRIGHA_RANGE)

    # Two options only: 1 = reduce tax and services, 2 = increase both.
    # Codes 8 and 9 are can't-choose and not-answered.
    tax_cats, tax_n = weighted_percentages(
        df, "taxsoca", W, labels=meta.variable_value_labels["taxsoca"],
        valid_range=TAXSOCA_RANGE)

    return ElectionData(
        cycle=1997,
        study="SN 3890",
        instrument="British Election Study 1997 Campaign Panel",
        fieldwork=("Salience wave C, 17-30 Apr 1997, telephone; attitudes "
                   "wave A, 10 May - 31 Jul 1996, face-to-face"),
        mode=("Salience collected by telephone; attitudinal items collected "
              "face-to-face in an earlier wave of the same panel"),
        sample_basis="Probability sample",
        weight_note="an electors-only weight; the file is not pre-weighted",
        salience_question="The issue that most influenced the respondent's vote",
        salience_frame="39 categories, manually coded",
        salience=salience,
        n_salience=n_sal,
        left_right=Reading("1-5 derived scale (1 = left, 5 = right)",
                           lr_n, mean=lr_mean),
        economic=Reading("taxation versus social services",
                         tax_n, categories=tax_cats),
        social=None,
        europe=None,
        notes=("Salience measures the vote-deciding issue, not the most "
               "important issue facing the country. Attitudinal items "
               "predate polling day by approximately ten months. "
               "Tax-versus-services ran as a split ballot, hence the "
               "reduced n."),
    )


def prepare_2001(data_dir):
    """2001, SN 4620, pre-campaign wave.

    No value labels in the Stata file: every frame below is typed from
    4620userguide.pdf (aq2 p.33, aq14i p.39, aq24 p.42, aq25f p.43,
    aq26a p.44). A slip here is invisible to the code.

    aq26a matches 2005 pre_q118 and 2010 aaq104. aq24 matches 2010
    aaq103 but not 2005, which used a 0-10 scale."""
    df, meta = _read(data_dir / "prepostagg.dta",
                     ["aq2", "aq25f", "aq26a", "aq24", "aq14i", "awgtgb"])
    W = "awgtgb"

    # The file carries no value labels; this frame is transcribed from the
    # user guide, p.33. Codes 88, 98 and 99 are non-answers and are dropped
    # by the ceiling of 14 rather than by name.
    ISSUES = {
        1.0: "Britain's membership in the European Monetary Union",
        2.0: "Britain's relations with the European Union",
        3.0: "Law and order",
        4.0: "Educational standards",
        5.0: "Environment",
        6.0: "National Health Service",
        7.0: "Inflation, prices generally",
        8.0: "Public transport",
        9.0: "Taxation",
        10.0: "State of the economy",
        11.0: "Unemployment",
        12.0: "My standard of living",
        13.0: "Price of petrol",
        14.0: "Other",
    }
    salience, n_sal = weighted_percentages(
        # 88 no important issues, 98 don't know, 99 refused - all above the
        # ceiling of 14, so the ceiling alone removes them.
        df, "aq2", W, labels=ISSUES, valid_range=AQ2_RANGE)

    lr_mean, lr_n = weighted_mean(df, "aq25f", W, SCALE_0_10)
    econ_mean, econ_n = weighted_mean(df, "aq26a", W, SCALE_0_10)

    # Typed frames - the file supplies none.
    EU = {1.0: "Strongly approve", 2.0: "Approve",
          3.0: "Neither approve nor disapprove", 4.0: "Disapprove",
          5.0: "Strongly disapprove"}
    AGREE = {1.0: "Strongly agree", 2.0: "Agree",
             3.0: "Neither agree nor disagree", 4.0: "Disagree",
             5.0: "Strongly disagree"}

    eu_cats, eu_n = weighted_percentages(df, "aq24", W, labels=EU,
                                          valid_range=FIVE_POINT)
    soc_cats, soc_n = weighted_percentages(df, "aq14i", W, labels=AGREE,
                                           valid_range=FIVE_POINT)

    return ElectionData(
        cycle=2001,
        study="SN 4620",
        instrument="British Election Panel Study 2001, pre-campaign wave",
        fieldwork="Pre-campaign wave, 2001, in-person CAPI",
        mode="In-person computer-assisted interviewing",
        sample_basis="Probability sample",
        weight_note="a region x gender x age weight",
        salience_question="Single most important issue in the general election",
        salience_frame="14 categories, manually coded",
        salience=salience,
        n_salience=n_sal,
        left_right=Reading("0-10, left to right", lr_n, mean=lr_mean),
        economic=Reading("0-10, cut taxes and spend much less to raise taxes "
                         "and spend much more", econ_n, mean=econ_mean),
        social=Reading("violent criminals deserve to be deprived of some of "
                       "their human rights", soc_n, categories=soc_cats),
        europe=Reading("approve or disapprove of EU membership",
                       eu_n, categories=eu_cats),
        notes=("The authority-liberty slot uses a single agree/disagree "
               "item, not a self-placement scale, so it is not equivalent "
               "to a graded position measure. Salience asks for the most "
               "important issue in the election rather than facing the "
               "country."),
    )


# 2005, SN 6607, pre-campaign. Same file as 2010; no left-right item
def prepare_2005(data_dir):
    df, meta = _read(data_dir / "p200506080910_jtw8.dta",
                     ["premiszz", "pre_q118", "pre_q124", "pre_q101", "pre_w8"])
    W = "pre_w8"

    salience, n_sal = weighted_percentages(
        df, "premiszz", W, labels=meta.variable_value_labels["premiszz"],
        # 88 "None, DK", 99 "NA"
        exclude_codes={88, 99})

    # All three are 0-10 self-placements; 11 and above are don't-know codes.
    econ_mean, econ_n = weighted_mean(df, "pre_q118", W, SCALE_0_10)
    soc_mean, soc_n = weighted_mean(df, "pre_q124", W, SCALE_0_10)
    eu_mean, eu_n = weighted_mean(df, "pre_q101", W, SCALE_0_10)

    return ElectionData(
        cycle=2005,
        study="SN 6607",
        instrument="British Election Study Nine-Wave Panel Survey 2005-2010",
        fieldwork="Pre-campaign wave, 2005, internet (YouGov)",
        mode="Internet panel",
        sample_basis="Online panel",
        weight_note="the standard published panel weighting",
        salience_question="Most important issue facing the country",
        salience_frame="26 categories, manually coded",
        salience=salience,
        n_salience=n_sal,
        left_right=None,
        economic=Reading("0-10, cut taxes to increase spending",
                         econ_n, mean=econ_mean),
        social=Reading("0-10, crime versus rights of the accused",
                       soc_n, mean=soc_mean),
        europe=Reading("0-10, EU in or out", eu_n, mean=eu_mean),
        notes=("No left-right self-placement was asked in this wave. "
               "Non-probability internet panel; a face-to-face probability "
               "cross-section of the same electorate exists but was not "
               "used, so that this cycle and the following one share a "
               "single instrument."),
    )


def prepare_2010(data_dir):
    """2010, SN 6607, pre-campaign, aa* prefix.

    EU item changed format from 2005 and the authority-liberty item was
    dropped. Sample is a five-year panel survivor subset, no refresh."""
    df, meta = _read(data_dir / "p200506080910_jtw8.dta",
                     ["aaissue1", "aaq104", "aaq103", "w8"])
    W = "w8"

    salience, n_sal = weighted_percentages(
        df, "aaissue1", W, labels=meta.variable_value_labels["aaissue1"],
        # 98 "DK, not sure", 99 "Refused". 80 "Miscellaneous" is retained.
        # 47 "UKIP-neg", 48 "UKIP-pos" are blinding exclusions; both are
        # 0.00% of this cycle.
        exclude_codes={98, 99}.union(BLINDING_EXCLUSIONS[2010]))

    econ_mean, econ_n = weighted_mean(df, "aaq104", W, SCALE_0_10)

    # Five named options; 6 is don't-know, 8 skipped, 9 not asked.
    eu_cats, eu_n = weighted_percentages(
        df, "aaq103", W, labels=meta.variable_value_labels["aaq103"],
        valid_range=FIVE_POINT)

    return ElectionData(
        cycle=2010,
        study="SN 6607",
        instrument="British Election Study Nine-Wave Panel Survey 2005-2010",
        fieldwork="Pre-campaign wave, 2010, internet (YouGov)",
        mode="Internet panel",
        sample_basis=("Continuation panel; 3,402 of the 7,793 respondents "
                      "originally recruited, 44% retention over five years, "
                      "with no refreshment sample"),
        weight_note="the standard published pre-campaign weight",
        salience_question="Most important issue facing the country",
        salience_frame="58 categories, manually coded",
        salience=salience,
        n_salience=n_sal,
        left_right=None,
        economic=Reading("0-10, cut taxes to increase spending",
                         econ_n, mean=econ_mean),
        social=None,
        europe=Reading("approve or disapprove of EU membership",
                       eu_n, categories=eu_cats),
        notes=("No left-right and no authority-liberty item were asked in "
               "this wave. The European integration item uses a five-point "
               "approval format rather than a graded scale. Respondents are "
               "a five-year panel survivor subset, which skews towards the "
               "politically engaged in a way demographic weighting does not "
               "correct."),
    )

# import deferred: besip_extract imports ElectionData from here
def _besip(cycle):
    def reader(data_dir):
        from .besip_extract import prepare_besip
        return prepare_besip(cycle)
    return reader


# add an election by writing its reader above and listing it here
READERS = {
    1997: prepare_1997,
    2001: prepare_2001,
    2005: prepare_2005,
    2010: prepare_2010,
    2015: _besip(2015),
    2017: _besip(2017),
    2019: _besip(2019),
    2024: _besip(2024),
    # 2015, 2017, 2019, 2024: one reader for all four, once the BESIP
    #                         subset exists - they share an instrument
}


def _render_directional(header, reading):
    out = ["", header]
    if reading is None:
        out.append(NOT_MEASURED)
    elif reading.mean is not None:
        out.append(f"Weighted mean {reading.mean} on {reading.description}.")
    else:
        parts = "; ".join(f"{label}: {pct}%" for label, pct in reading.categories)
        out.append(f"{reading.description.capitalize()} - {parts}.")
    return out

# each of these identifies the election on its own
_LEAKS = [
    (re.compile(r"\b(?:18|19|20)\d{2}\b"), "a four-digit year"),
    (re.compile(r"\bSN\s*\d{3,5}\b", re.I), "a study number"),
    (re.compile(r"\b(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\b"), "a month name"),
    # full forms only, bare "Reform" and "Labour" appear in real categories
    # ("Government Reform", "labour market").
    (re.compile(r"\bUKIP\b|\bSNP\b|Labour Party|Conservative Party|"
                r"Liberal Democrat|Reform UK|Brexit Party|Green Party|"
                r"Tony Blair", re.I), "a party or leader name"),
]


def assert_blinded(text):
    """Raise if the rendered record names the election.

    Args:
        text (str): the rendered record.

    Raises:
        ValueError: if a year, study number, month or party name is present.
    """
    for pattern, what in _LEAKS:
        hit = pattern.search(text)
        if hit:
            raise ValueError(
                f"blinding violation: block contains {what} ({hit.group(0)!r}). "
                "Identifying detail belongs in metadata(), not in the block."
            )


# identifying detail, for cache keys and the audit trail
def metadata(data):
    return {
        "cycle": data.cycle,
        "study": data.study,
        "instrument": data.instrument,
        "fieldwork": data.fieldwork,
        "n_salience": data.n_salience,
    }


# reads only the model-visible half of ElectionData
def render(data, check=True):
    lines = [
        "SLOT 1  Instrument and mode",
        data.mode + ".",
        "",
        "SLOT 2  Sample",
        f"{data.sample_basis}. Weighted by {data.weight_note}.",
        "",
        "SLOT 3  Issue salience",
        f"{data.salience_question}, {data.salience_frame}:",
    ]
    for label, pct in data.salience:
        lines.append(f"  {label:<32}{pct:>5.1f}%")

    lines += _render_directional("SLOT 4  Left-right position", data.left_right)
    lines += _render_directional("SLOT 5  Economic position (tax and spending)",
                                 data.economic)
    lines += _render_directional("SLOT 6  Social position (authority and liberty)",
                                 data.social)
    lines += _render_directional("SLOT 7  European integration", data.europe)
    lines += ["", "SLOT 8  Instrument notes", data.notes]

    text = "\n".join(lines)
    if check:
        assert_blinded(text)
    return text


def build_election_data(cycle, data_dir):
    """Read one election and render it for the prompt.

    Args:
        cycle (int): election year.
        data_dir (Path): directory holding the BES .dta files.

    Returns:
        tuple: the blinded text, and the metadata withheld from the model.
    """
    if cycle not in READERS:
        raise ValueError(f"no reader written for {cycle}; "
                         f"have {sorted(READERS)}")
    block = READERS[cycle](Path(data_dir))
    return render(block), metadata(block)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="directory holding the BES .dta files")
    parser.add_argument("-c", "--cycle", type=int, action="append",
                        choices=sorted(READERS),
                        help="render one election (repeatable; default all)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_dir.is_dir():
        sys.exit(f"Not a directory: {args.data_dir}")

    for cycle in (args.cycle or sorted(READERS)):
        try:
            text, meta = build_election_data(cycle, args.data_dir)
        except FileNotFoundError as err:
            print(f"{cycle}: {err}", file=sys.stderr)
            continue
        print("=" * 68)
        print(f"  {cycle}  [metadata, NOT sent to the model]")
        for key, value in meta.items():
            print(f"    {key}: {value}")
        print("=" * 68)
        print(text)
        print()


if __name__ == "__main__":
    main()
