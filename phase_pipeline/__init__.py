"""phase_pipeline: LLM manifesto-alignment pipeline.

Modules
-------
prompts         REGISTERED prompt wordings and schemas.
llm_client      All API calls, with disk caching, retries, and provenance.
summarise       Phase 1: summarisation variants. Six are registered, but the
                main series runs three.
commitments     PRIMARY Phase 1 stability: commitment extraction, survival,
                selection.
stability       Secondary text-similarity metric, and medoid selection for profiles.
profiles        Phase 2: electorate profiles. Four arms are
                registered, three are carried into Phase 3.
compare         Phase 3: pairwise comparison, both orderings, label
                randomisation, slot balance.
bradley_terry   Bradley-Terry estimation (regularised lsr_pairwise).
validate        Spearman validation, binary winner, and the four registered
                error types.
probe           Identification probe: measured anonymisation leakage.
ches            CHES party families, for directional error. Expert GAL-TAN
                positions are implemented but not used by the main series.
consolidate_metrics
                Per-run results tables, assembled from the phase reports.
"""