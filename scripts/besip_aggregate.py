"""Aggregate BESIP waves to weighted summaries.

    python besip_aggregate.py bes_panel_ukds_v30_1.dta -o out.csv

Reads only the listed variables, so the 3 GB file is never fully loaded.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyreadstat

# Configuration

# pre-campaign wave per election, matching the 2005 and 2010 extractions
WAVES = {2015: 4, 2017: 11, 2019: 17, 2024: 26}

# lr_scale/al_scale are per wave-group, so the name carries the band
SCALE_BAND = {4: "W1_W5", 11: "W10_W12", 17: "W17_W19", 26: "W25W26"}

# weight prefix changes at wave 13, wt_full_W* before and wt_new_W* after
WEIGHT = {4: "wt_full_W4", 11: "wt_full_W11",
          17: "wt_new_W17", 26: "wt_new_W26"}

# salience frame is 1-50 with no code 20, non-responses are system missing
MIN_SALIENCE_CODE = 1
MAX_SALIENCE_CODE = 50

# scales are 0-10, leftRight* and redistSelf* code don't know as 9999
MIN_SCALE_VALUE = 0
MAX_SCALE_VALUE = 10

# 1. Command line

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dta", type=Path,
                        help="BESIP combined panel file (SN 8202)")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("besip_aggregates.csv"),
                        help="output CSV (default: besip_aggregates.csv)")
    return parser.parse_args()


def check_input(path):
    if not path.is_file():
        sys.exit(f"Not a file: {path}")
    if path.suffix.lower() != ".dta":
        sys.exit(f"Expected a .dta file, got: {path.name}")

# 2. Which variables to read

# manual coding to wave 25, LLM from wave 26
def salience_var(wave):
    return f"mii_cat_llmW{wave}" if wave >= 26 else f"mii_catW{wave}"


def scale_vars(wave):
    band = SCALE_BAND[wave]
    variables = {
        "left_right": f"leftRightW{wave}",
        "lr_scale": f"lr_scale{band}",
        "al_scale": f"al_scale{band}",
        "economic": f"redistSelfW{wave}",
        "europe": f"EUIntegrationSelfW{wave}",
    }
    # immigSelf* does not begin until wave 10, so wave 4 has no immigration
    # self-placement. Wave 4 carries immigCultural/immigEcon/immigrationLevel
    # instead, which ask different questions and are not equivalent.
    if wave >= 10:
        variables["immigration"] = f"immigSelfW{wave}"
    return variables


def wanted_variables(wave):
    return ([salience_var(wave), WEIGHT[wave]]
            + list(scale_vars(wave).values()))

# 3. First pass: variable names only

# first pass: names only, so a wrong name surfaces before the read
def check_variables(path, waves):
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)
    available = set(meta.column_names)

    wanted, missing = [], []
    for wave in waves:
        for var in wanted_variables(wave):
            (wanted if var in available else missing).append(var)

    if missing:
        print(f"WARNING: {len(missing)} expected variables not in file:",
              file=sys.stderr)
        for var in missing:
            print(f"    {var}", file=sys.stderr)
        print("Check these against the codebook before using the output.\n",
              file=sys.stderr)

    print(f"Reading {len(wanted)} of {len(available)} variables")
    return wanted

# 4. Second pass: the data

# usecols is what makes this feasible
def read_columns(path, columns):
    return pyreadstat.read_dta(str(path), usecols=columns)

# 5. Weighting

# weighted share per category. Categories as published; not rescaled
def weighted_percentages(df, var, weight, labels):
    valid = df[df[var].between(MIN_SALIENCE_CODE, MAX_SALIENCE_CODE)
               & (df[weight] > 0)]
    if valid.empty:
        return [], 0
    shares = valid.groupby(var)[weight].sum()
    shares = shares.div(shares.sum()).mul(100)
    out = [(labels.get(code, str(code)), round(float(pct), 2))
           for code, pct in shares.items()]
    out.sort(key=lambda row: -row[1])
    return out, len(valid)


def weighted_mean(df, var, weight):
    valid = df[df[var].between(MIN_SCALE_VALUE, MAX_SCALE_VALUE)
               & (df[weight] > 0)]
    if valid.empty:
        return None, 0
    mean = (valid[var] * valid[weight]).sum() / valid[weight].sum()
    return round(float(mean), 3), len(valid)

# 6. One wave at a time

def salience_rows(df, meta, wave, cycle, weight):
    var = salience_var(wave)
    if var not in df.columns:
        print(f"  wave {wave}: {var} absent, no salience recorded",
              file=sys.stderr)
        return []

    labels = meta.variable_value_labels.get(var, {})
    if not labels:
        print(f"  wave {wave}: {var} has no value labels, categories will "
              "be numeric", file=sys.stderr)

    percentages, n = weighted_percentages(df, var, weight, labels)
    return [{
        "cycle": cycle,
        "wave": wave,
        "measure": "salience",
        "variable": var,
        "label": label,
        "value": pct,
        "n": n,
        "coding": "llm" if wave >= 26 else "manual",
    } for label, pct in percentages]


def scale_rows(df, wave, cycle, weight):
    rows = []
    for measure, var in scale_vars(wave).items():
        if var not in df.columns:
            continue
        mean, n = weighted_mean(df, var, weight)
        if mean is None:
            continue
        rows.append({
            "cycle": cycle,
            "wave": wave,
            "measure": measure,
            "variable": var,
            "label": "",
            "value": mean,
            "n": n,
            "coding": "",
        })
    return rows


def wave_rows(df, meta, wave, cycle):
    weight = WEIGHT[wave]
    if weight not in df.columns:
        print(f"  wave {wave}: weight {weight} absent, wave skipped",
              file=sys.stderr)
        return []

    rows = (salience_rows(df, meta, wave, cycle, weight)
            + scale_rows(df, wave, cycle, weight))
    print(f"  wave {wave} ({cycle}) done")
    return rows

# 7. Write out

def write_csv(rows, out_path):
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(rows)} rows, "
          f"{out_path.stat().st_size / 1024:.0f} KB)")

# Assembly

def main():
    args = parse_args()
    check_input(args.dta)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    waves = sorted(WAVES.values())
    cycle_of = {wave: cycle for cycle, wave in WAVES.items()}

    columns = check_variables(args.dta, waves)
    df, meta = read_columns(args.dta, columns)

    rows = []
    for wave in waves:
        rows += wave_rows(df, meta, wave, cycle_of[wave])

    if not rows:
        sys.exit("No rows produced; check the warnings above.")

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
