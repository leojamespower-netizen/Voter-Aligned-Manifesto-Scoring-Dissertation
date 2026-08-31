#BESIP aggregates to a the pre defined BES wide eight-slot block, for 2015-2024.

from pathlib import Path

from .bes_extract import ElectionData, Reading

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATES_CSV = REPO_ROOT / "data" / "besip_aggregates.csv"

CYCLES = (2015, 2017, 2019, 2024)

#Pre-campaign wave per election, and its fieldwork. Metadata only.
FIELDWORK = {
    2015: ("wave 4", "4-30 Mar 2015", "7 May 2015"),
    2017: ("wave 11", "24 Apr - 3 May 2017", "8 Jun 2017"),
    2019: ("wave 17", "1-12 Nov 2019", "12 Dec 2019"),
    2024: ("wave 26", "3-22 May 2024", "4 Jul 2024"),
}

#Which aggregate row fills which slot. lr_scale and al_scale are derived
#multi-item scales; leftRight and redistSelf are single self-placements.
SLOT_MEASURE = {
    "left_right": ("left_right", "0-10 self-placement, left to right"),
    "economic": ("economic", "0-10 self-placement on redistribution, "
                             "government should redistribute to ordinary "
                             "people"),
    "social": ("al_scale", "0-10 derived authority-liberty scale, higher is "
                           "more authoritarian"),
    "europe": ("europe", "0-10 self-placement on European integration, "
                         "higher is more favourable to integration"),
}


def _load(path=AGGREGATES_CSV):
    import pandas as pd
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/besip_aggregate.py against the "
            "SN 8202 file first.")
    return pd.read_csv(path)


def prepare_besip(cycle, path=AGGREGATES_CSV):
    if cycle not in CYCLES:
        raise ValueError(f"{cycle} is not a BESIP cycle; have {CYCLES}")

    rows = _load(path)
    rows = rows[rows["cycle"] == cycle]
    if rows.empty:
        raise ValueError(f"no aggregate rows for {cycle}")

    salience_rows = rows[rows["measure"] == "salience"]
    salience = [(str(r.label), round(float(r.value), 1))
                for r in salience_rows.sort_values("value", ascending=False)
                                      .itertuples()]

    def reading(slot):
        measure, description = SLOT_MEASURE[slot]
        match = rows[rows["measure"] == measure]
        if match.empty:
            return None
        row = match.iloc[0]
        return Reading(description, int(row["n"]),
                       mean=round(float(row["value"]), 2))

    wave, fieldwork, polling_day = FIELDWORK[cycle]
    llm_coded = bool((salience_rows["coding"] == "llm").any())

    notes = [
        "Salience categories are the 50-category frame applied to unprompted "
        "open-ended responses.",
    ]
    if llm_coded:
        notes.append(
            "These responses were categorised automatically rather than by "
            "hand, unlike earlier readings on the same instrument.")
    if reading("left_right") and rows[rows["measure"] == "immigration"].empty:
        notes.append(
            "No immigration self-placement was asked in this wave; the "
            "questions carried instead ask whether immigration benefits the "
            "culture and the economy, which are not equivalent.")
    notes.append(
        "The authority-liberty figure is a derived multi-item scale rather "
        "than a single self-placement, so it is not directly comparable to "
        "the single-item measures in the other slots.")

    return ElectionData(
        # metadata - recorded, never rendered
        cycle=cycle,
        study="SN 8202",
        instrument="BES Combined Internet Panel, 4th edition",
        fieldwork=f"{wave}, {fieldwork}, polling day {polling_day}",
        # model-visible
        mode="Internet panel",
        sample_basis="Non-probability online panel with refreshment samples",
        weight_note="the published single-wave weight",
        salience_question="Most important issue facing the country",
        salience_frame=f"{len(salience)} categories, "
                       + ("automated coding" if llm_coded
                          else "manually coded"),
        salience=salience,
        n_salience=int(salience_rows["n"].max()),
        notes=" ".join(notes),
        left_right=reading("left_right"),
        economic=reading("economic"),
        social=reading("social"),
        europe=reading("europe"),
    )
