# LLM Manifesto-Alignment Pipeline

Code, data and cached model responses for the project: *Using Large
Language Models to Predict Electoral Outcomes Through Voter-Aligned Manifesto
Scoring*.

The pipeline operates in three stages.
Stage 1: UK party manifestos are summarised via a JSON prompt sent to the OPENAI/ANTHROPIC API, via the command line. 
Stage 2: A separate command line instruction then prompts the same LLMs to construct Voter priority profiles using Ipsos Issues Index
and the British Election Study, filtered through the lens of Moral Foundations Theory, and the GAL-TAN Axis.
Stage 3: An additional command line instruction then send a separate set of prompts that draw from the cache of phase 1 and 2 outputs to facilitate a pairwise voter-alignment comparison using choix's Bradley-Terry package.
Analysis: The resultant log odds, Luce-Plackett probabilities and stability measures are checked against actual vote shares across eight UK general
elections (1997–2024).


Every model response is committed in `cache/`, so all
reported results can be regenerated without an API key.

## Reproducing the results (no API key required)

Two commands rebuild every table in the report from the raw responses in
`cache/`. Neither makes a network call:

```
python -m phase_pipeline.run_analysis --phase3 cache/comparisons --output outputs
python -m phase_pipeline.consolidate_metrics --output outputs
```

## Repository layout

- `phase_pipeline/` — the pipeline. A runner per phase, the modules that
  build and parse each phase's requests, `prompts.py` holding every prompt
  and registered setting, the survey readers, the analysis modules, and
  `llm_client.py`, which handles both APIs.
- `scripts/` — steps run outside the pipeline, on a machine holding the
  licensed survey files.
- `tests/` — fifty test functions. Five need the survey files and skip
  without them, so the suite runs anywhere.
- `cache/` — every raw model response, one JSON file per call, under
  `summaries/`, `summaries_rejected/`, `extractions/`, `profiles/`,
  `comparisons/` and `probes/`, plus `call_log.csv`. This is the evidential
  record behind every number in the dissertation.
- `data/` — the forty manifesto texts, the sixteen election records read by
  Phase 2, CHES party families, vote shares and pre-election polling
  averages. Survey data is **not** committed: UK Data Service files are
  licensed for use but not redistribution (SN 3890, 4620, 6607, 8202).
- `outputs/` — per-election analysis JSON and the consolidated tables.

## Environment

Python 3.13. Install the pinned dependencies:

```
pip install -r requirements.txt
```

## API keys (only for the stages that call a model)

`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, set as environment variables,
never committed. No key is needed for `--dry-run`, the tests,
or any analysis stage.

## Building each input

Two steps run locally against files downloaded from the UK Data Service,
because the data cannot be redistributed. Both of their outputs are
committed, so the repository runs without the licensed files.

The BES Internet Panel (SN 8202) is 3.26 GB and too large to process inside
the pipeline, so it is aggregated once:

```
python scripts/besip_aggregate.py bes_panel_ukds_v30_1.dta -o data/besip_aggregates.csv
```

The sixteen election records, eight elections by two sources, are then
written to `data/voter_profiles/`:

```
python -m phase_pipeline.build_election_data --bes DIR --ipsos DIR
python -m phase_pipeline.build_election_data --bes DIR --ipsos DIR --check
```

Every BES record is rendered into the same eight-slot structure, with a note
in any slot the study did not measure, so that a change in the instrument
cannot reach the model as an apparent change in the electorate. `--check`
reports which committed records differ from what the readers now produce; if
a reader changes, rerun.

## Running the pipeline

Every stage takes `--dry-run`, which resolves its inputs, counts the calls it
would make, and stops without touching the API.

```
python -m phase_pipeline.run_phase1 2024 --model gpt-5     # summaries
python -m phase_pipeline.run_probes 2024 --model gpt-5     # identification probes
python -m phase_pipeline.run_phase2 2024 --model gpt-5     # profiles
python -m phase_pipeline.run_phase3 2024 --model gpt-5     # comparisons
python -m phase_pipeline.run_analysis                      # no calls
```

Run them in that order: each reads the report the previous one wrote into
`outputs/`. Every response is cached under a key recording everything that
could change it, so an interrupted run resumes without re-billing, and a
changed prompt writes to a new key rather than overwriting the old one.

The registered grid is 21 cells per election per model: three summarising
instructions, crossed with a no-profile control plus three frameworks by two
surveys, with all ten pairs judged in both orders. That is 620 calls per
election per model across the four stages. `--variants`, `--arms` and
`--sources` restrict the grid; `--temperature` was used only for the 2024
Opus 4.5 comparison. The 2024 election was additionally run on an extended grid
of 50 cells, 1,380 calls per model condition, and the results of that run
fixed the narrower design used for the other seven.

One point of vocabulary: the code and the output tables use `arm` for what
the dissertation calls a framework.

## Tests

```
python -m pytest tests/ -q
$env:BES_DIR="C:\path\to\dta"; $env:IPSOS_DIR="C:\path\to\ipsos"; python -m pytest tests/ -q
```

## Status

Complete. All eight elections have been run on both models, with 2024 also
run on the extended grid and a second time for Opus 4.5 with sampling
randomness disabled. The cache holds every response, and the analysis stage
regenerates every reported table from it.




