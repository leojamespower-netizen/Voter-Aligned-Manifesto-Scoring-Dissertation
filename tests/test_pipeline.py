"""Tests for the failures that do not raise.

    python -m pytest tests/ -v

Every bug found in this codebase so far produced plausible output rather
than an error: a substring match that deleted a thousand real responses, a
threshold that excluded nothing, two modules disagreeing about whether a key
was called slot_A or slot_a. None of those would fail a smoke test, and all
of them change the numbers. These tests target that class specifically.

The survey tests need the .dta files, which are licensed and not committed,
so they skip when the directory is absent. Everything else runs anywhere.
"""

import os
import pytest
from pathlib import Path

from phase_pipeline import compare, probe, validate
from phase_pipeline.bes_extract import (BLINDING_EXCLUSIONS, assert_blinded,
                                        build_election_data,
                                        weighted_percentages)
from phase_pipeline.ipsos_extract import IPSOS_NON_ANSWERS, prepare_ipsos
from phase_pipeline.profiles import parse_profile
from phase_pipeline.vote_shares import vote_shares

BES_DIR = os.environ.get("BES_DIR", "data/bes")
IPSOS_DIR = os.environ.get("IPSOS_DIR", "data/ipsos")

needs_bes = pytest.mark.skipif(not os.path.isdir(BES_DIR),
                               reason="BES .dta files not present")
needs_ipsos = pytest.mark.skipif(not os.path.isdir(IPSOS_DIR),
                                 reason="Ipsos files not present")


# known values
# If the weighting, the exclusions or the code frames change, these move.
# They are recorded here so a change has to be deliberate.

@needs_bes
def test_1997_health_share_is_stable():
    text, _ = build_election_data(1997, BES_DIR)
    assert "health                           33.6%" in text


@needs_bes
def test_1997_left_right_mean_is_stable():
    text, _ = build_election_data(1997, BES_DIR)
    assert "Weighted mean 2.39" in text


@needs_ipsos
def test_1997_ipsos_nhs_share_is_unnormalised():
    # 63 is the published figure. If the normalisation step at the foot of
    # the source file ever runs, this becomes about 21.
    text, _ = prepare_ipsos(1997, IPSOS_DIR)
    assert "63.0%" in text


# non-answer exclusion
# "na" is a substring of "National Health Service"; "not sure" is a
# substring of "Not sure of party". Both deleted real categories once.

def test_exclusion_is_by_code_not_by_label():
    import pandas as pd
    df = pd.DataFrame({
        "issue": [1, 1, 2, 998],
        "w": [1.0, 1.0, 1.0, 1.0],
    })
    labels = {1.0: "National Health Service", 2.0: "Educational standards",
              998.0: "don't know"}
    rows, n = weighted_percentages(df, "issue", "w", labels=labels,
                                   exclude_codes={998})
    names = [label for label, _ in rows]
    assert "National Health Service" in names
    assert "Educational standards" in names
    assert "don't know" not in names
    assert n == 3


def test_ipsos_exclusion_does_not_match_substrings():
    assert "Don't know" in IPSOS_NON_ANSWERS
    assert "NHS/Hospitals/Healthcare" not in IPSOS_NON_ANSWERS
    assert "Race relations" not in IPSOS_NON_ANSWERS


# blinding
# The guard has to reject identifying detail without rejecting real data.
# "Government Reform" and "labour market" are genuine survey categories.

@pytest.mark.parametrize("text", [
    "Fieldwork April 1997",
    "British Election Study SN 3890",
    "collected in June",
    "UKIP-neg 0.4%",
    "labour party 0.17%",
])
def test_assert_blinded_rejects_identifying_detail(text):
    with pytest.raises(ValueError):
        assert_blinded(text)


@pytest.mark.parametrize("text", [
    "Government Reform 2.1%",
    "labour market conditions",
    "Green issues 3.0%",
    "Weighted mean 2.39 on a 1-5 derived scale",
])
def test_assert_blinded_accepts_real_categories(text):
    assert_blinded(text)


@needs_bes
@needs_ipsos
def test_every_record_passes_blinding():
    for cycle in (1997, 2001, 2005, 2010, 2015, 2017, 2019, 2024):
        text, _ = build_election_data(cycle, BES_DIR)
        assert_blinded(text)
        text, _ = prepare_ipsos(cycle, IPSOS_DIR)
        assert_blinded(text)


@needs_bes
def test_blinding_exclusions_are_applied():
    text, _ = build_election_data(1997, BES_DIR)
    assert "tony blair" not in text.lower()
    assert "conservative party" not in text.lower()
    assert set(BLINDING_EXCLUSIONS[1997]) == {192, 193, 194}


# key agreement between modules
# compare.py writes the metadata; validate.py reads it. A disagreement
# about a key name returns an empty result rather than an error, which
# looks exactly like a null finding.

def _verdict(pair, slot_a, winner, **meta):
    base = {"election": "2024", "pair": list(pair), "scorer": "gpt-5",
            "condition": "c", "prompt_type": "baseline", "labelled": False,
            "run_index": 0, "slot_A": slot_a}
    base.update(meta)
    return {"winner_party": winner, "confidence": "high",
            "parse_error": False, "meta": base}


def test_flip_rate_reads_the_key_compare_writes():
    verdicts = [
        _verdict(("lab", "con"), "lab", "lab", variant="minimal"),
        _verdict(("lab", "con"), "lab", "con", variant="distorted"),
    ]
    result = validate._flip_rate(verdicts, "variant")
    assert result["n_comparisons"] == 1, "grouping found no comparisons"
    assert result["rate"] == 1.0


def test_positional_error_uses_the_same_metadata():
    verdicts = [
        _verdict(("lab", "con"), "lab", "lab"),
        _verdict(("lab", "con"), "con", "con"),
    ]
    assert validate.positional_error(verdicts) == 1.0


def test_framing_sensitivity_finds_the_variant_key():
    verdicts = []
    for variant in ("minimal", "neutral", "framing_preserving"):
        for slot in ("lab", "con"):
            verdicts.append(_verdict(("lab", "con"), slot, "lab",
                                     variant=variant))
    result = validate.framing_sensitivity(verdicts)
    assert result["framing_axis"]["n_comparisons"] > 0


def test_invariance_survives_missing_ordering_pairs():
    # one ordering only: the positional floor is unavailable, but the rest
    # of the analysis must still run
    verdicts = [_verdict(("lab", "con"), "lab", "lab", variant="minimal")]
    result = validate.invariance_violation(verdicts)
    assert result["positional_swap"] is None


# parsers
# The two places the code assumes something about what the model returns.
# Models wrap JSON in code fences and add commentary regardless of
# instructions.

def test_parse_verdict_handles_a_code_fence():
    response = {
        "text": '```json\n{"winner": "TEXT_E", "confidence": "high", '
                '"reasoning": "x", "evidence": "y"}\n```',
        "meta": {"slot_A": "lab", "slot_B": "con", "label_A": "TEXT_E",
                 "label_B": "TEXT_C", "pair": ("lab", "con")}
    }
    verdict = compare.parse_verdict(response)
    assert not verdict.get("parse_error")
    assert verdict["winner_party"] == "lab"


def test_parse_profile_flags_failure_rather_than_returning_none():
    # both orchestrators once checked for None, which is never returned
    result = parse_profile({"text": "not json at all", "meta": {}},
                           "explicit_mft")
    assert result is not None
    assert result["parse_error"] is True


def test_parse_failure_is_counted_not_dropped():
    verdicts = [
        _verdict(("lab", "con"), "lab", "lab"),
        {"winner_party": None, "parse_error": True, "meta": {}},
    ]
    assert validate.parse_failure_rate(verdicts) == 0.5


# thresholds and ranges
# A ceiling that excludes nothing is worse than no ceiling: it looks like
# a filter and behaves like a pass-through.

def test_scale_range_excludes_dont_know_codes():
    import pandas as pd
    df = pd.DataFrame({"scale": [0.0, 5.0, 10.0, 9999.0],
                       "w": [1.0, 1.0, 1.0, 1.0]})
    from phase_pipeline.bes_extract import weighted_mean
    mean, n = weighted_mean(df, "scale", "w", (0, 10))
    assert n == 3
    assert mean == 5.0


def test_probe_threshold_is_set_and_below_certainty():
    assert 0 < probe.FAILURE_THRESHOLD < 1


# structure
# Cheap assertions that catch a whole file going missing or a party set
# being edited by accident.

def test_every_election_has_five_parties():
    for cycle in (1997, 2001, 2005, 2010, 2015, 2017, 2019, 2024):
        assert len(vote_shares(cycle)) == 5


def test_all_pairs_is_complete_and_unordered():
    pairs = compare.all_pairs(["a", "b", "c", "d", "e"])
    assert len(pairs) == 10
    assert len({frozenset(p) for p in pairs}) == 10

def test_API_registration():
    from phase_pipeline.llm_client import PROVIDERS, GPT5, CLAUDE
    assert GPT5 in PROVIDERS and CLAUDE in PROVIDERS
    for spec in PROVIDERS.values():
        assert spec["sdk"] in ("openai", "anthropic")
        assert spec["key_var"] and spec["api_model"]


def test_missing_key():
    from phase_pipeline import llm_client
    import os
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    llm_client._CLIENTS.clear()
    try:
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            llm_client._api_request("hi", llm_client.CLAUDE, None, {})
    finally:
        if saved:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_unknown_model_identification():
    from phase_pipeline.llm_client import _api_request
    with pytest.raises(ValueError, match="Unknown model"):
        _api_request("hi", "not-a-model", None, {})


def test_cache_keys_separate_by_model():
    # the key format used throughout phase 1
    keys = {f"2024_lab_minimal_{m}_run1" for m in ("gpt-5", "claude")}
    assert len(keys) == 2, "a second model would overwrite the first's cache"


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    # stubs the two provider calls, recording what each was sent, and points
    # the cache and the call log at a temporary directory
    from types import SimpleNamespace
    from phase_pipeline import llm_client
    sent = {}
    failures = []  # exceptions to raise before succeeding

    def create(**kwargs):  # stands in for OpenAI's chat.completions.create
        if failures:
            raise failures.pop(0)
        sent.clear()
        sent.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    def post(body, key):  # stands in for the Messages API
        if failures:
            raise failures.pop(0)
        sent.clear()
        sent.update(body)
        return {"content": [{"type": "text", "text": "ok"}]}

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm_client, "_client", lambda model: client)
    monkeypatch.setattr(llm_client, "_anthropic_post", post)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(llm_client, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm_client, "CALL_LOG", tmp_path / "call_log.csv")
    monkeypatch.setattr(llm_client, "TEMPERATURE_OVERRIDE", None)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    (tmp_path / "probes").mkdir()
    return llm_client, sent, tmp_path / "probes", failures


def decoding_sent(sent):
    return {k: v for k, v in sent.items() if k not in ("model", "messages", "system")}  # drops the non-decoding fields


def test_every_subdir_used_has_decoding_settings():
    import ast, glob
    from phase_pipeline.llm_client import DECODING
    used = set()
    for f in glob.glob("phase_pipeline/*.py"):
        for node in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "subdir" and isinstance(kw.value, ast.Constant):
                        used.add(kw.value.value)
    assert used <= set(DECODING), f"no decoding registered for {used - set(DECODING)}"


def test_request_and_record_match(fake_api):
    llm_client, sent, _, _ = fake_api
    response = llm_client.call_llm("hi", "gpt-5", "k", subdir="probes")
    assert response["decoding"] == decoding_sent(sent)  # the cache file records what was sent


def test_temperature_is_uniform_across_stages():
    from phase_pipeline.llm_client import DECODING
    values = {s["temperature"] for s in DECODING.values()}
    assert len(values) == 1, f"stages disagree on temperature: {values}"


def test_temperature_override_separates_the_cache(fake_api):
    llm_client, sent, cache, _ = fake_api
    llm_client.call_llm("hi", "claude", "k", subdir="probes")
    llm_client.TEMPERATURE_OVERRIDE = 0.0
    llm_client.call_llm("hi", "claude", "k", subdir="probes")
    assert sent["temperature"] == 0.0  # the override is sent
    assert len(list(cache.iterdir())) == 2, "two temperatures shared a file"

def test_no_output_cap_is_registered():
    from phase_pipeline.llm_client import DECODING
    for subdir, settings in DECODING.items():
        assert "max_tokens" not in settings and "max_completion_tokens" not in settings, f"{subdir} is capped"

def test_gpt5_sends_no_decoding_parameters(fake_api):
    llm_client, sent, _, _ = fake_api
    llm_client.call_llm("hi", "gpt-5", "k", subdir="probes")
    assert decoding_sent(sent) == {}  # GPT-5 accepts only its defaults

def test_claude_sends_only_the_required_max_tokens(fake_api):
    llm_client, sent, _, _ = fake_api
    llm_client.call_llm("hi", "claude", "k", subdir="probes")
    assert decoding_sent(sent) == {"max_tokens": llm_client.ANTHROPIC_MAX_TOKENS}
    assert sent["model"] == "claude-opus-4-5-20251101"
    block = sent["messages"][0]["content"][0]
    assert block["text"] == "hi" and block["cache_control"] == {"type": "ephemeral"}  # prompt unchanged, marked for caching to reduce financial and time costs

def test_retry_recovers_from_transient_error(fake_api):
    llm_client, _, cache, failures = fake_api
    failures.extend([ConnectionError("dropped"), ConnectionError("dropped")])
    response = llm_client.call_llm("hi", "gpt-5", "k", subdir="probes")
    assert response["text"] == "ok"
    assert len(list(cache.iterdir())) == 1

def test_fake_api_runs_do_not_touch_the_real_call_log(fake_api):
    llm_client, _, _, _ = fake_api
    real = Path(llm_client.__file__).resolve().parent.parent / "cache" / "call_log.csv"
    before = real.read_bytes() if real.exists() else None
    llm_client.call_llm("hi", "gpt-5", "k", subdir="probes")
    assert (real.read_bytes() if real.exists() else None) == before

def test_consolidate_metrics_mean_rho(tmp_path, monkeypatch):
    # two cells in one arm, rho +1 and -1: the arm's mean rho must be 0
    import json
    from phase_pipeline import consolidate_metrics as cm
    right = ["lab", "con", "reform", "ld", "green"]
    def cell(ranking, rho):
        scores = {p: 5 - i for i, p in enumerate(ranking)}
        return {"spearman": {"rho": rho, "pvalue": 0.0, "n": 5}, "null_percentile": 0.5, "ranking": ranking,
                "binary_winner": False, "positional_error": 0.5, "parse_failure_rate": 0.0, "scores": scores,
                "alpha_sensitivity": {"0.1": scores}}
    report = {"election": 2024, "model": "test", "plan": {"calls_per_cell": 20},
              "vote_shares": {"lab": 33.7, "con": 23.7, "reform": 14.3, "ld": 12.2, "green": 6.4},
              "cells": {"minimal/explicit_mft/bes": cell(right, 1.0),
                        "neutral/explicit_mft/bes": cell(right[::-1], -1.0)}}
    (tmp_path / "p3.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["consolidate_metrics", str(tmp_path / "p3.json"), "--out", str(tmp_path), "--label", "t"])
    cm.main()
    t = json.loads((tmp_path / "tables_t.json").read_text(encoding="utf-8"))
    assert t["by_arm_source"] == [{"arm": "explicit_mft", "source": "bes", "n_cells": 2, "mean_rho": 0.0,
                                   "cells_rho_positive": 1, "winner_rate": 0, "mean_positional_error": 0.5, "rejected": 0}]
    assert t["by_arm_source"] == [{"arm": "explicit_mft", "source": "bes", "n_cells": 2, "mean_rho": 0.0,
    "cells_rho_positive": 1, "winner_rate": 0, "mean_positional_error": 0.5, "rejected": 0}]


def test_profile_design_tag_is_in_the_cache_keys():
    # pilot files are 2024_<arm>_<source>_<model>_runN; the main series must not read them
    import inspect
    from phase_pipeline import profiles, compare, prompts as P
    assert P.PROFILE_DESIGN == "forced"
    assert "P.PROFILE_DESIGN" in inspect.getsource(profiles.generate_profiles)
    assert "P.PROFILE_DESIGN" in inspect.getsource(compare.run_pair)

def test_polling_averages_2024_rank_the_parties_as_the_result_did():
    # the recorded 2024 averages must be one row per party and order them like the vote
    from phase_pipeline.vote_shares import polling_averages, vote_shares
    polling, shares = polling_averages(2024), vote_shares(2024)
    assert set(polling) == set(shares)
    assert sorted(polling, key=polling.get) == sorted(shares, key=shares.get)

def test_polling_averages_absent_election_returns_none(tmp_path):
    # an election with no rows, or no file at all, is None rather than an error
    from phase_pipeline.vote_shares import polling_averages
    assert polling_averages(1900) is None
    assert polling_averages(2024, path=tmp_path / "missing.csv") is None

def test_polling_benchmark_gap_is_polls_minus_pipeline():
    # perfect polls, backwards pipeline: gap is 2 and the pipeline does not match
    from phase_pipeline import validate
    shares = {"lab": 33.7, "con": 23.7, "reform": 14.3, "ld": 12.2, "green": 6.4}
    backwards = {p: -v for p, v in shares.items()}
    out = validate.benchmark_against_polling(backwards, shares, shares)
    assert out["rho_polling"] == pytest.approx(1.0) and out["rho_pipeline"] == pytest.approx(-1.0)
    assert out["gap"] == pytest.approx(2.0)
    assert not out["pipeline_matches_or_beats"] 

def test_verdict_names_the_text_not_the_position():
    # the question and the schema use the pseudonyms; "A"/"B" appear nowhere as references
    import inspect
    from phase_pipeline.vote_shares import ELECTIONS, vote_shares
    for election in ELECTIONS:
        parties = list(vote_shares(election))
        for run in range(10):
            labels = set(compare.run_mapping(parties, run).values())
            assert len(labels) == len(parties) and not labels & {"TEXT_A", "TEXT_B"}, f"{election} run {run}: {labels}"
    prompt = compare.build_compare_prompt("baseline", "one", "two", "TEXT_E", "TEXT_C")
    assert "whether TEXT_E or TEXT_C" in prompt and '"winner": "TEXT_E" | "TEXT_C"' in prompt
    assert " A or B" not in prompt and '"A" | "B"' not in prompt
    # a verdict naming the second text is placed in the second slot
    meta = {"slot_A": "lab", "slot_B": "con", "label_A": "TEXT_E", "label_B": "TEXT_C", "pair": ("lab", "con")}
    verdict = compare.parse_verdict({"text": '{"winner": "TEXT_C"}', "meta": meta})
    assert verdict["winner_party"] == "con"
    assert compare.parse_verdict({"text": '{"winner": "B"}', "meta": meta})["parse_error"]  # a bare letter is no longer an answer
    assert "LABEL_SCHEME" in inspect.getsource(compare.run_pair)

def test_score_spread_and_margin_come_from_the_raw_scores():
    # a decisive ranking and a near-tied one have the same rho; only the scores tell them apart
    from phase_pipeline import consolidate_metrics as cm
    shares = {"lab": 33.7, "con": 23.7, "reform": 14.3, "ld": 12.2, "green": 6.4}
    def cell(scores):
        return {"scores": scores, "ranking": sorted(scores, key=scores.get, reverse=True)}
    report = {"vote_shares": shares, "cells": {
        "minimal/axis/bes": cell({"lab": 2.0, "con": 1.0, "reform": 0.0, "ld": -1.0, "green": -2.0}),
        "neutral/axis/bes": cell({"lab": 0.02, "con": 0.01, "reform": 0.0, "ld": -0.01, "green": -0.02})}}
    rows = {r["variant"]: r for r in cm.score_rows(report)}
    assert rows["minimal"]["spread"] == 4.0 and rows["minimal"]["top_margin"] == 1.0
    assert rows["neutral"]["spread"] == pytest.approx(0.04) and rows["neutral"]["top_margin"] == pytest.approx(0.01)
    assert rows["minimal"]["pearson_log_share"] == pytest.approx(rows["neutral"]["pearson_log_share"])  # same order, same correlation


def test_main_grid_is_a_subset_of_the_narrowed_design():
    from phase_pipeline import prompts as P
    assert set(P.MAIN_VARIANTS) <= set(P.CARRIED_FORWARD)
    assert set(P.MAIN_ARMS) <= set(P.PROFILE_PROMPTS) and "baseline" not in P.MAIN_ARMS  # Phase 3 adds baseline
    assert set(P.MAIN_SOURCES) <= set(P.SOURCE_CONDITIONS)
    assert "minimal" in P.MAIN_VARIANTS and "anonymised" in P.MAIN_VARIANTS  # reference and blinding condition stay


def test_pipeline_defaults_to_the_main_grid(monkeypatch):
    from phase_pipeline import prompts as P, run_phase1, run_phase2, run_phase3, run_probes
    for module, argv in ((run_phase1, ["x", "2024"]), (run_phase2, ["x", "2024"]), (run_phase3, ["x", "2024"])):
        monkeypatch.setattr("sys.argv", argv)
        args = module.parse_args()
        if hasattr(args, "variants"):
            assert tuple(args.variants) == P.MAIN_VARIANTS
        if hasattr(args, "sources"):
            assert tuple(args.sources) == P.MAIN_SOURCES
    assert tuple(run_phase2.parse_args().arms) == P.MAIN_ARMS
    assert tuple(run_phase3.parse_args().arms) == ("baseline", *P.MAIN_ARMS)
    assert set(run_probes.DEFAULT_VARIANTS) <= set(P.MAIN_VARIANTS)  # probes read summaries Phase 1 made

def test_summarise_only_runs_the_requested_variants(fake_api):
    llm_client, _, _, _ = fake_api
    (llm_client.CACHE_DIR / "summaries").mkdir()
    from phase_pipeline import summarise
    responses = summarise.summarise_manifesto("2024", "lab", "text", "gpt-5",
                                              variants=["minimal", "anonymised"])
    keys = {r["cache_key"] for r in responses}
    assert len(responses) == 2 * summarise.N_RUNS
    assert all(("_minimal_" in k) or ("_anonymised_" in k) for k in keys)

def test_phase1_plan_counts_the_requested_variants():
    from phase_pipeline import run_phase1, summarise
    counts = run_phase1.plan({"lab": "x", "con": "y"}, "gpt-5", ["minimal", "anonymised"])
    assert counts["summary_calls"] == 2 * 2 * summarise.N_RUNS and counts["total_calls"] == 2 * counts["summary_calls"]


def test_summary_gate_rejects_fragments():
    from phase_pipeline.summarise import looks_like_summary
    manifesto = " ".join(f"word{i}" for i in range(3000))
    assert not looks_like_summary("Thanks to Claude for assistance with this document.", manifesto)  # fragment
    assert looks_like_summary(" ".join(f"summary{i}" for i in range(300)), manifesto)  # proper length
    assert looks_like_summary("short", "a short manifesto")  # no floor for tiny sources like the Referendum leaflet


def test_summary_gate_re_requests_and_store_reject(fake_api, monkeypatch):
    # first reply is a fragment, second a real summary: the fragment is kept, the summary is used
    llm_client, _, _, _ = fake_api
    (llm_client.CACHE_DIR / "summaries").mkdir()
    from phase_pipeline import summarise
    replies = iter(["Vote Green!"] + [" ".join(f"s{i}" for i in range(300))] * 20)
    monkeypatch.setattr(llm_client, "_anthropic_post", lambda body, key: {"content": [{"type": "text", "text": next(replies)}]})
    monkeypatch.setattr(summarise, "N_RUNS", 1)
    out = summarise.summarise_manifesto("2005", "green", " ".join(f"m{i}" for i in range(2000)), "claude", variants=["minimal"])
    assert out[0]["rejected_attempts"] == 1 and len(out[0]["text"].split()) == 300
    assert (llm_client.CACHE_DIR / "summaries_rejected" / "2005_green_minimal_claude_run1_rejected1.json").exists()


