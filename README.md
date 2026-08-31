# LLM Manifesto-Alignment Pipeline

Code and data for the dissertation: "Using Large Language Models to Predict
Electoral Outcomes Through Voter-Aligned Manifesto Scoring".

The pipeline constructs voter priority profiles from the Ipsos Issues Index
and the British Election Study, summarises UK party manifestos with an LLM,
runs pairwise voter-alignment comparisons, estimates Bradley-Terry scores,
and validates them against actual vote shares across eight UK general
elections (1997–2024).

## Reproducing the results (no API keys required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leojamespower-netizen/Voter-Aligned-Manifesto-Scoring-Dissertation/blob/main/execution_notebooks/reproduce_results.ipynb)

1. Click the badge above, or open `execution_notebooks/reproduce_results.ipynb` in Colab.
2. Runtime → Run all.
3. Compare the regenerated outputs to Tables/Figures [TODO: X–Y] in the dissertation.

This notebook rebuilds every result from the raw LLM responses committed in
`cache/`. It needs no API keys, costs nothing, and completes in a few minutes.

## Repository layout

- `phase_pipeline/` — all pipeline code.
- `scripts/` — steps run outside the pipeline, on a machine holding the
  licensed survey files.
- `tests/` — unit tests. Five require the survey files and skip without them.
- `execution_notebooks/execute_pipeline.ipynb` — the full pipeline **including
  API calls**. Requires `OPENAI_API_KEY`; documents how the cached responses
  were generated. Re-running it produces slightly different raw outputs due
  to LLM stochasticity (see dissertation §3.4).
- `execution_notebooks/reproduce_results.ipynb` — deterministic reproduction
  from cached responses (see above).
- `cache/` — every raw LLM response (JSON, one file per call) plus
  `call_log.csv`. This is the evidential record.
- `data/` — manifesto texts, Manifesto Project documents, survey aggregates,
  vote shares, and the sixteen election records read by Phase 2. Survey
  microdata is **not** committed: UK Data Service files are licensed for use
  but not redistribution (SN 3890, 4620, 6607, 8202).
- `outputs/` — reports written by each phase, plus tables and figures.

## Environment

Python 3.13 · install pinned dependencies with:

```
pip install -r requirements.txt
```

## API keys (only needed for the stages that call a model)

Set as an environment variable or Colab Secret, never committed:
`OPENAI_API_KEY`. No key is needed for `--dry-run`, the tests, or
`run_analysis`.

## Building the inputs

Survey microdata is not committed, so two steps run locally against files
downloaded from the UK Data Service.

The BES Internet Panel (SN 8202) is 3.26 GB and too large to process in the
pipeline, so it is aggregated once:

```
python scripts/besip_aggregate.py bes_panel_ukds_v30_1.dta -o data/besip_aggregates.csv
```

The sixteen election records — eight elections × two sources — are then
written to `data/voter_profiles/`:

```
python -m phase_pipeline.build_election_data --bes DIR --ipsos DIR
python -m phase_pipeline.build_election_data --bes DIR --ipsos DIR --check
```

Both outputs are committed, so the repository runs without the licensed
files. `--check` reports which records differ from what the readers now
produce; if a reader changes, rerun.

## Running the pipeline

Every stage takes `--dry-run`, which resolves its inputs, counts the calls it
would make and stops without touching the API.

```
python -m phase_pipeline.run_phase1 2024        # summaries, 300 calls
python -m phase_pipeline.run_probes 2024        # identification, 20 calls
python -m phase_pipeline.run_phase2 2024        # profiles, 60 calls
python -m phase_pipeline.run_phase3 2024        # comparisons, up to 1000 calls
python -m phase_pipeline.run_calibration        # noise floor, 100 calls
python -m phase_pipeline.run_analysis           # no calls
```

Run them in that order: each reads the report the previous one wrote, into
`outputs/`. Every API response is cached, so an interrupted run resumes
without re-billing.

Phase 3 takes `--variants`, `--arms` and `--sources` to restrict the grid,
and `--text-source cmp` to compare Manifesto Project coded text instead of
the Phase 1 summaries. Comparing the two runs is the content diagnostic; it
is registered for 2015, 2019 and 2024 only.

## Tests

```
python -m pytest tests/ -q
BES_DIR=/path/to/dta IPSOS_DIR=/path/to/ipsos python -m pytest tests/ -q
```

Twenty-seven tests. Five need the licensed survey files and skip when the
directories are absent, so the suite runs anywhere.

## Current state

Working: survey extraction for all eight elections and both sources; forty
manifestos; fifteen Manifesto Project documents; all four phases, the
calibration cell, the identification probes and the analysis stage;
twenty-seven tests.

Not yet run against a live API. Outstanding: polling averages for
`benchmark_against_polling`, an adversarial run for `adversarial_ablation`,
and anonymisation replacement lists if the identification probe indicates
they are needed.