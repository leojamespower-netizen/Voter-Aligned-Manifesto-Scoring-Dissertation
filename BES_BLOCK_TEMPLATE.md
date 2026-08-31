# BES source block: fixed template

Status: **draft, not yet registered.** Freeze before the first confirmatory call.

## Why a fixed template

`profiles.py` passes the BES data to the model as a single string, in the
`bes` source arm and in both `both_*` arms. The BES instrument changed
substantially across the eight cycles, so an unconstrained summary would be
richer in later elections than earlier ones. Any improvement in later
elections could then be an artefact of the arm receiving more information,
not of the model reasoning better.

The template holds the *shape* of the block constant across all eight
elections. Slots that a cycle cannot fill are stated as absent rather than
omitted. The model therefore sees the same eight headings every time, and the
variation across cycles is declared in the text rather than hidden in it.

This is the same reasoning as the provenance tags already registered in
`build_source_block`: the caveat travels with the data.

## Blinding

Blinding is registered: no prompt in any phase names the election, the year
or the electoral context. A fieldwork date names the election as surely as
the year does, so SLOT 1 carries mode and RELATIVE timing only - "collected
approximately ten months before polling day", never "fielded in May 1996".

Section 4 of the methodology requires each data block to carry provenance
tags including the fieldwork window. That rule and the blinding rule cannot
both hold literally. Resolved in favour of blinding: mode, weighting, sample
basis and relative timing are rendered; absolute dates, study numbers and
instrument names go to `metadata()`, which reaches the cache key and the
audit record but never a prompt.

`render()` reads only the model-visible half of `ElectionData` and passes its output
through `assert_blinded()`, which rejects a four-digit year, a study number,
a month name or a party name. This duplicates the repository prompt lint
deliberately: the lint guards the prompt directory, this guards data
assembled at runtime, which the lint never sees.

Blinding cannot be complete. A salience list is self-dating to a reader who
knows British politics, which is why the design measures the leak with
identification probes rather than assuming it away. What this guarantees is
only that the pipeline does not hand the answer over for free.

## The eight slots

Always emitted, always in this order, one blank line between slots.

```
SLOT 1  Instrument and mode
SLOT 2  Sample
SLOT 3  Issue salience
SLOT 4  Left-right position
SLOT 5  Economic position (tax and spending)
SLOT 6  Social position (authority and liberty)
SLOT 7  European integration
SLOT 8  Instrument notes
```

## Absence notation

A slot that cannot be filled emits exactly this, with no elaboration:

```
SLOT n  <name>
NOT MEASURED in this survey cycle.
```

Absence is never expressed as a near-substitute, an estimate, or an item
from a different survey. A cycle that measured something related but not
equivalent emits `NOT MEASURED` and records the related item in SLOT 8.

Rationale: an approximate fill silently converts a missing measurement into
a present one, which is the exact failure the template exists to prevent.

## Slot definitions

**SLOT 1 — Instrument and mode.** How the survey was administered, and how
far ahead of polling day it was fielded, in relative terms. Mode is stated
explicitly because it changes across the series (face-to-face probability
sampling, through hybrid, to non-probability internet panel) and those mode
effects are documented to run in opposite directions across eras. Study
numbers, instrument names, wave labels and absolute dates are withheld -
see Blinding above.

**SLOT 2 — Sample.** Achieved n for the wave, weight described rather than
named (a variable name such as WTERA identifies the study),
sampling basis in one clause (probability sample / online panel /
continuation panel). Where the wave is a continuation of an earlier panel,
state the retention rate.

**SLOT 3 — Issue salience.** Top five issues by weighted share, with
percentages. State the number of categories in the coding frame and whether
the coding was manual or automated. Multi-mention frames state how many
mentions were counted.

**SLOT 4 — Left-right position.** Weighted mean and distribution on an
explicit 0-10 self-placement scale. Derived scales are acceptable only if
the derivation is stated.

**SLOT 5 — Economic position.** Weighted mean on the tax-versus-spending
trade-off, scale endpoints named in full.

**SLOT 6 — Social position.** Weighted mean on the authority-versus-liberty
dimension, scale endpoints named in full. The underlying item differs across
cycles (crime versus rights of the accused in 2005; derived battery in
2015-2024); the item is named in SLOT 8.

**SLOT 7 — European integration.** Weighted distribution. Format differs
across cycles (0-10 scale versus approve/disapprove categories) and the
format is stated inline, not harmonised.

**SLOT 8 — Instrument notes.** Free text, maximum four sentences. Records
the specific items behind slots 4-7, any format change from the preceding
cycle, and any related-but-not-equivalent item that a `NOT MEASURED` slot
excluded. This is the only slot whose length varies.

## Availability matrix

Verified against the data files and dictionaries held. Unverified cells are
marked and must be confirmed before freezing.

| Slot | 1997 | 2001 | 2005 | 2010 | 2015 | 2017 | 2019 | 2024 |
|---|---|---|---|---|---|---|---|---|
| 3 Salience | yes† | yes† | yes | yes | yes | yes | yes | yes* |
| 4 Left-right | yes | yes | no | no | yes | yes | yes | yes |
| 5 Economic | yes | yes | yes | yes | yes | yes | yes | yes |
| 6 Social | no | yes‡ | yes | no | yes | yes | yes | yes |
| 7 EU | no | yes | yes | yes | yes | yes | yes | yes |

† Measures the vote-deciding issue, not the most important issue facing the
country. A construct difference, not a coding difference.
‡ A single agree/disagree item, not a graded self-placement.
`*` 2024 salience coding is automated, not manual.
2015-2024 cells are from the v30 dictionary and remain unverified against
data until the aggregates exist.

Sources: 2005 and 2010 from SN 6607 (`pre_q*`, `aa*`); 2015-2024 from SN 8202
v30 (`leftRightW*`, `lr_scaleW*`, `al_scaleW*`, `redistSelfW*`,
`immigSelfW*`, `EUIntegrationSelfW*`, `mii_catW*` / `mii_cat_llmW*`).

## Consequence to register

The 1997-2010 cycles emit at least one `NOT MEASURED` slot; 2015-2024 emit
none. The `bes` arm therefore carries strictly more information in the later
cycles. This is a declared property of the design, not a nuisance: the
prediction that follows is that the `bes`-versus-`ipsos` contrast is
*larger* in 2015-2024 than in 1997-2010, and that any such difference is
attributable to instrument coverage rather than to model capability.

Registering that prediction in advance is what separates it from a post hoc
explanation of an awkward result.
