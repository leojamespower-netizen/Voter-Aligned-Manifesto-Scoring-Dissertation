"""phase_pipeline: LLM manifesto-alignment pipeline.

Modules
-------
prompts         REGISTERED prompt wordings and schemas (frozen artefact).
llm_client      All API calls, with disk caching, retries, and origin.
summarise       Phase 1: six summarisation variants.
commitments     PRIMARY Phase 1 stability: commitment extraction, survival,
                selection, threshold sensitivity.
stability       Secondary text-similarity metric; medoid selection for profiles.
profiles        Phase 2: model-derived electorate profiles, four arms.
compare         Phase 3: pairwise comparison, both orderings, label
                randomisation, slot balance.
bradley_terry   Bradley-Terry estimation (regularised lsr_pairwise).
validate        Spearman validation, binary winner, four error analyses.
probe           Identification probe: measured anonymisation leakage.
cmp             CMP corpus text and the content diagnostic layer.
ches            CHES party families (directional error) and expert GAL-TAN
                positions (external benchmark for the axis arm).
"""
