"""REGISTERED prompt specification (Phases 1-3)."""
# Registered replication count: every LLM cell (summary variant, profile
# cell, calibration pass) is run this many times. One value, imported
# everywhere, so the phases cannot drift apart.
N_RUNS = 5

VERDICT_SCHEMA = """Respond with a single JSON object and nothing else. Do not wrap it in \
markdown code fences and do not add commentary before or after it. The object \
must have exactly these keys, in this order:

{{
  "evidence": "<the specific commitments in each text you relied on>",
  "reasoning": "<two sentences connecting that evidence to the verdict>",
  "confidence": "high" | "medium" | "low",
  "winner": "A" | "B"
}}"""

# Profile design. The 2024 pilot let the model abstain per foundation
# (SCORED / AMBIGUOUS / INSUFFICIENT); one model abstained on most of them.
# The main series forces a full set of weights. The tag goes into the profile
# and comparison cache keys so the two designs never share a file.
PROFILE_DESIGN = "forced"

PROFILE_SCHEMA = """Your profile construction must follow the below rules:
   - All six foundations must appear exactly once.
   - Each foundation must be given a weight between 0 and 1.
   - The six weights must sum to 1.

Respond with a single JSON object and nothing else. Do not wrap it in markdown \
code fences and do not add commentary before or after it. The object must have \
exactly these keys, in this order:

{{
  "justification": "<two sentences on how you read the data>",
  "foundations": [
    {{"foundation": "care" | "loyalty" | "authority" | "sanctity" | "equality" | "proportionality",
     "basis": "<one line: which items in the data bear on this>",
     "weight": <0-1>}}
  ]
}}"""

AXIS_SCHEMA = """Your profile construction must follow the below rules:
   - Both poles must appear exactly once.
   - Each pole must be given a weight between 0 and 1.
   - The two weights must sum to 1.

Respond with a single JSON object and nothing else. Do not wrap it in markdown \
code fences and do not add commentary before or after it. The object must have \
exactly these keys, in this order:

{{
  "justification": "<two sentences on how you read the data>",
  "poles": [
    {{"pole": "gal" | "tan",
     "basis": "<one line: which items in the data bear on this>",
     "weight": <0-1>}}
  ]
}}"""

# Six-foundation formulation (Fairness split into Equality and
# Proportionality), per the revised Moral Foundations Theory.
FOUNDATIONS = ("care", "loyalty", "authority", "sanctity",
               "equality", "proportionality")
POLES = ("gal", "tan")


# Framework definition blocks (byte-identical across all conditions)
MFT_DEFINITIONS = """  - Care/Harm: related to our long evolution as mammals with attachment systems
    and an ability to feel (and dislike) the pain of others. It underlies the
    virtues of kindness, gentleness, and nurturance.

  - Loyalty/Betrayal: related to our long history as tribal creatures able to
    form shifting coalitions. It is active anytime people feel that it is "one
    for all and all for one." It underlies the virtues of patriotism and
    self-sacrifice for the group.

  - Authority/Subversion: shaped by our long primate history of hierarchical
    social interactions. It underlies virtues of leadership and followership,
    including deference to prestigious authority figures and respect for
    traditions.

  - Sanctity/Purity: shaped by the psychology of disgust and contamination. It
    underlies notions of striving to live in an elevated, less carnal, more
    noble, and more "natural" way (often present in religious narratives). This
    foundation underlies the widespread idea that the body is a temple that can
    be desecrated by immoral activities and contaminants (an idea not unique to
    religious traditions). It underlies the virtues of self-discipline,
    self-improvement, naturalness, and spirituality.

  - Equality: related to our intuitions about equal treatment and equal outcome
    for individuals.

  - Proportionality: related to our intuitions about individuals getting
    rewarded in proportion to their merit or contribution."""

# Single worked example of a bidirectional foundation
BIDIRECTIONALITY_EXAMPLES = """One person's sensitivity to the Care/Harm foundation may lead them to support \
high levels of immigration from poorer nations, to reduce the suffering of those \
who migrate. Whereas another person's sensitivity to the very same foundation \
may lead them to oppose those levels, because they believe it could lead to \
people they care about being harmed."""

# Single-axis GAL-TAN formulation (Hooghe, Marks & Wilson 2002; operationalised
# by the Chapel Hill Expert Survey, which is also the party-family source for
# directional error). Held byte-identical across all conditions.
AXIS_DEFINITIONS = """The GAL-TAN framework identifies political orientation on a single axis. \
Individuals and texts are placed according to the degree to which their views \
align with either pole, as follows:

  - Individuals at the GAL (Green/Alternative/Libertarian) pole support expanded
    personal freedoms and social liberalisation, for example: access to abortion,
    same-sex marriage, environmental protection, and multiculturalism.

  - Individuals at the TAN (Traditional/Authoritarian/Nationalist) pole prioritise
    order, tradition, and stability, and believe that the state should treat the
    maintenance of national identity as a priority, for example: restrictive
    immigration policy, emphasis on law and order, defence of traditional social
    institutions, and national sovereignty over supranational authority.

A text occupies a position on this axis to the extent that it argues FROM these \
commitments, not to the extent that it mentions the associated topics:

  - Environmental or immigration policy discussed in administrative or fiscal
    terms does not by itself indicate a position.
  - Nuclear expansion argued for energy security is not GAL.
  - Faster asylum processing argued on humanitarian grounds is not TAN."""

# PHASE 1 - Summarisation variants (prose output; no JSON schema)

MINIMAL = "Summarise the following text within 500 words.\n\n{manifesto}"

NEUTRAL = ("Summarise the following text within 500 words, while excluding "
           "ideological framing.\n\n{manifesto}")

FRAMING_PRESERVING = ("Summarise the following text within 500 words, while "
                      "maintaining the ideological framing.\n\n{manifesto}")

FRAMED = ("Summarise the following text, which is the {party} manifesto, "
          "within 500 words, while excluding ideological framing.\n\n{manifesto}")

# Transformation rule
DISTORTED = "Summarise following text within 500 words\n\n{manifesto}"

# Prompt identical to NEUTRAL apart from the additional call for party name removal.
ANONYMISED = ("Summarise the following text within 500 words, while "
              "excluding ideological framing. Do not name any political "
              "parties, or individuals.\n\n{manifesto}")

VARIANTS: dict[str, str] = {
    "minimal": MINIMAL,
    "neutral": NEUTRAL,
    "framing_preserving": FRAMING_PRESERVING,
    "framed": FRAMED,
    "distorted": DISTORTED,
    "anonymised": ANONYMISED,
}

ANONYMISED_INPUT: set[str] = {"anonymised"}
NEEDS_PARTY: set[str] = {"framed"}


# Variants carried into Phase 3. Distorted's hypothesis completes at the
# summary level and its single-draw limitation weakens downstream inference.
CARRIED_FORWARD: tuple[str, ...] = (
    "minimal", "neutral", "framing_preserving", "framed", "anonymised",
)


# PHASE 2 - Profile construction


_DATA_BLOCK = """The text below is drawn from {source_description}, a measure of the \
priorities of the British electorate.

{voter_priority_data}"""

PROFILE_EXPLICIT_MFT = f"""Moral Foundations Theory is a framework that separates human morality \
into the following categories:

{{mft_definitions}}

{_DATA_BLOCK}

Your task is to construct a Moral Foundations profile for this electorate on \
the basis of the data above.

{PROFILE_SCHEMA}"""

PROFILE_BIDIRECTIONAL_MFT = f"""Moral Foundations Theory is a framework that separates human morality \
into the following categories:

{{mft_definitions}}

Every category has a bidirectional nature. For example:

{{bidirectionality_examples}}

{_DATA_BLOCK}

Your task is to construct a Moral Foundations profile for this electorate on \
the basis of the data above, taking into account that each foundation may be \
invoked in either direction.

{PROFILE_SCHEMA}"""

PROFILE_AXIS = f"""{{axis_definitions}}

{_DATA_BLOCK}

Your task is to construct a positional profile for this electorate on the \
basis of the data above.

{AXIS_SCHEMA}"""

# Probe arm: no definitions supplied. Registered as a measurement of the
# model's internalised framework.
PROFILE_ZERO_DEFINITION = f"""{_DATA_BLOCK}

Your task is to construct a Moral Foundations profile for this electorate on \
the basis of the data above.

{PROFILE_SCHEMA}"""

PROFILE_PROMPTS: dict[str, str] = {
    "explicit_mft": PROFILE_EXPLICIT_MFT,
    "bidirectional_mft": PROFILE_BIDIRECTIONAL_MFT,
    "axis": PROFILE_AXIS,
    "zero_definition": PROFILE_ZERO_DEFINITION,
}

# Which schema each framework arm returns (drives parsing).
PROFILE_UNIT: dict[str, str] = {
    "explicit_mft": "foundations",
    "bidirectional_mft": "foundations",
    "axis": "poles",
    "zero_definition": "foundations",
}

SOURCE_CONDITIONS: tuple[str, ...] = (
    "ipsos", "bes", "both",
)

# ===========================================================================
# PHASE 3 - Pairwise comparison
# ===========================================================================

# Deliberately domain-neutral: naming the document type or the political
# context would supply framing the texts already carry, and would give the
# model a head start on identification. Same principle as the open probe.
_TEXTS_BLOCK = """Below are two texts.

{label_a}:
{text_a}

{label_b}:
{text_b}"""

COMPARE_BASELINE = f"""{_DATA_BLOCK}

{_TEXTS_BLOCK}

Judge whether {{ref_a}} or {{ref_b}} better reflects what these voters cared about.

{VERDICT_SCHEMA}"""

COMPARE_EXPLICIT_MFT = f"""Moral Foundations Theory is a framework that separates human morality \
into the following categories:

{{mft_definitions}}

The profile below describes the moral priorities of the British electorate.

{{profile}}

{_TEXTS_BLOCK}

Judge whether {{ref_a}} or {{ref_b}} aligns more closely with this profile.

{VERDICT_SCHEMA}"""

COMPARE_BIDIRECTIONAL_MFT = f"""Moral Foundations Theory is a framework that separates human morality \
into the following categories:

{{mft_definitions}}

Every category has a bidirectional nature. For example:

{{bidirectionality_examples}}

The profile below describes the moral priorities of the British electorate.

{{profile}}

{_TEXTS_BLOCK}

Judge whether {{ref_a}} or {{ref_b}} aligns more closely with this profile, \
taking into account that each foundation may be invoked in either direction.

{VERDICT_SCHEMA}"""

COMPARE_AXIS = f"""{{axis_definitions}}

The profile below describes the positional priorities of the British electorate.

{{profile}}

{_TEXTS_BLOCK}

Judge whether {{ref_a}} or {{ref_b}} aligns more closely with this profile.

{VERDICT_SCHEMA}"""

COMPARE_PROMPTS: dict[str, str] = {
    "baseline": COMPARE_BASELINE,
    "explicit_mft": COMPARE_EXPLICIT_MFT,
    "bidirectional_mft": COMPARE_BIDIRECTIONAL_MFT,
    "axis": COMPARE_AXIS,
}

# Prompt types that require a Phase 2 profile.
NEEDS_PROFILE: set[str] = {"explicit_mft", "bidirectional_mft", "axis"}

# MEASUREMENT INSTRUMENT - commitment extraction (Phase 1 stability)

EXTRACTION_PROMPT = """Extract every policy commitment stated in the text below.

Rules:
- One commitment per line.
- Write each as a verb phrase: an action followed by its object. For example: \
"raise the income tax threshold" or "build 300000 new homes".
- State each commitment at the level of a distinct policy action, not its \
implementation details. Do not subdivide a single commitment into steps.
- Include every commitment. Do not select, rank, prioritise, or judge importance.
- Use the text's own terms for policy objects. Do not substitute synonyms.
- Do not number the lines. No bullets, headings, preamble, or commentary.
- If the text states no policy commitments, output the single word NONE.

<text>
{summary}
</text>"""
