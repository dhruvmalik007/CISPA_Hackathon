"""Tests for the model-equality method: statistics, decision rule, and budget safety.

The budget arithmetic is tested as carefully as the statistics, because overspending is
irreversible: 100 responses per endpoint is all a team gets for the whole competition.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from met import client, stats  # noqa: E402
from met.simulate import (  # noqa: E402
    decide,
    evaluate_against_truth,
    read_practice_truth,
    synthetic_dataset,
)

TRUTH_PATH = ROOT / "official" / "practice_answers.csv"


def _truth():
    return read_practice_truth(TRUTH_PATH)


# --------------------------------------------------------------------------------------
# MCC and the competition score
# --------------------------------------------------------------------------------------


def test_mcc_perfect_and_inverted():
    assert stats.mcc(tp=10, tn=20, fp=0, fn=0) == pytest.approx(1.0)
    assert stats.mcc(tp=0, tn=0, fp=10, fn=20) == pytest.approx(0.0)


def test_mcc_is_zero_when_prediction_is_constant():
    """A constant answer has an undefined MCC; the server floors it, and so must we."""
    assert stats.mcc(tp=0, tn=0, fp=0, fn=10) == 0.0
    assert stats.mcc(tp=0, tn=0, fp=10, fn=0) == 0.0


def test_mcc_never_negative():
    assert stats.mcc(tp=1, tn=1, fp=9, fn=9) == 0.0


def test_mcc_matches_the_textbook_formula():
    """Verify against the closed form directly, so the test does not depend on scikit-learn."""
    truth = [1, 1, 0, 0, 1, 0, 0, 1]
    pred = [1, 0, 0, 1, 1, 0, 1, 1]
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    expected = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert stats.mcc(tp, tn, fp, fn) == pytest.approx(max(0.0, expected))


def test_competition_score_weights_model_at_ninety_percent():
    """Family right but model wrong must leave only the 10% family term.

    Three endpoints are needed: with only two endpoints in one family there is a single pair and
    MCC is undefined whichever way it is answered, so the 90/10 split cannot be observed.
    """
    truth = {"a": ("F1", "M1"), "b": ("F1", "M2"), "c": ("F2", "M3")}
    # Same families, but every model distinct -> model question wrong on the a,b pair.
    guess = {"a": ("F1", "M1"), "b": ("F1", "M9"), "c": ("F2", "M3")}
    result = stats.competition_score(truth, guess)
    assert result["mcc_family"] == pytest.approx(1.0)
    assert result["mcc_model"] == pytest.approx(0.0)
    assert result["score"] == pytest.approx(0.1)


def test_model_mcc_ignores_cross_family_pairs():
    """Pairs from different families are never the same model, so they must not be counted."""
    truth = {"a": ("F1", "M1"), "b": ("F2", "M2"), "c": ("F1", "M1")}
    guess = {"a": ("F1", "M1"), "b": ("F2", "M2"), "c": ("F1", "M1")}
    result = stats.competition_score(truth, guess)
    # Only the (a,c) pair shares a family; the cross-family pairs are excluded entirely.
    assert result["model_pairs"] == {"tp": 1, "tn": 0, "fp": 0, "fn": 0}
    assert result["mcc_family"] == pytest.approx(1.0)


def test_two_endpoint_family_mcc_is_undefined_and_reported_as_zero():
    """Documents a real degeneracy: one pair, one true negative, no positives -> MCC undefined."""
    truth = {"a": ("F1", "M1"), "b": ("F2", "M2")}
    result = stats.competition_score(truth, truth)
    assert result["mcc_family"] == 0.0
    assert result["model_pairs"] == {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


# --------------------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------------------


def test_similarity_is_bounded_and_symmetric():
    a = "the quick brown fox jumps over the lazy dog"
    b = "the quick brown fox leaps over a lazy dog"
    assert 0.0 <= stats.similarity(a, b) <= 1.0
    assert stats.similarity(a, b) == pytest.approx(stats.similarity(b, a))


def test_identical_text_is_maximally_similar():
    text = "A straightforward and perfectly ordinary sentence."
    assert stats.similarity(text, text) == pytest.approx(1.0)
    # Whitespace and case are normalised, not treated as difference.
    assert stats.similarity(text, "  a STRAIGHTFORWARD and perfectly ordinary sentence. ") > 0.9


def test_watermark_like_insertions_are_tolerated():
    """A watermark is a small insertion; it must not by itself make a pair look different."""
    base = "The committee approved the proposal after a lengthy discussion of the budget."
    marked = "The committee approved the proposal after a lengthy discussion of the budget. \u200b\u2063"
    assert stats.similarity(base, marked) > 0.8


def test_different_continuations_are_not_similar():
    a = "The committee approved the proposal after a lengthy discussion of the budget."
    b = "Photosynthesis converts light energy into chemical energy stored in glucose molecules."
    assert stats.similarity(a, b) < 0.2


def test_empty_responses_are_handled():
    assert stats.similarity("", "") == 1.0
    assert stats.similarity("", "something") == 0.0


# --------------------------------------------------------------------------------------
# permutation test and multiple testing
# --------------------------------------------------------------------------------------


def test_permutation_test_detects_strong_agreement():
    rng = np.random.default_rng(0)
    values = np.full(25, 0.9) + rng.normal(scale=0.01, size=25)
    observed, p_value = stats.paired_permutation_test(values, rng)
    assert observed > 0.85
    assert p_value < 0.01


def test_permutation_test_does_not_reject_noise():
    rng = np.random.default_rng(1)
    values = rng.normal(scale=0.25, size=25)
    _, p_value = stats.paired_permutation_test(values, rng)
    assert p_value > 0.05


def test_permutation_test_detects_systematic_dissimilarity():
    """A pair that is reliably LESS similar than chance is evidence of different models."""
    rng = np.random.default_rng(2)
    values = np.full(25, 0.05) + rng.normal(scale=0.01, size=25)
    _, p_value = stats.paired_permutation_test(values, rng)
    assert p_value < 0.01


def test_permutation_test_honours_tiny_samples():
    rng = np.random.default_rng(3)
    _, p_value = stats.paired_permutation_test(np.array([0.9, 0.9]), rng)
    # n<3 cannot support a permutation argument; refusing to reject is the honest answer.
    assert p_value == 1.0


def test_benjamini_hochberg_is_monotone_and_bounded():
    adjusted = stats.benjamini_hochberg([0.001, 0.02, 0.04, 0.5, 0.9])
    assert np.all(adjusted >= 0) and np.all(adjusted <= 1)
    assert np.all(np.diff(adjusted[[0, 1, 2]]) >= -1e-12)


def test_benjamini_hochberg_controls_false_discoveries():
    """With no true signal, the adjusted p-values must not manufacture significance."""
    rng = np.random.default_rng(4)
    p_values = rng.uniform(size=528)  # 528 = the real number of endpoint pairs
    adjusted = stats.benjamini_hochberg(p_values, fdr=0.05)
    assert int((adjusted <= 0.05).sum()) == 0


def test_benjamini_hochberg_preserves_order():
    p_values = [0.5, 0.001, 0.2]
    adjusted = stats.benjamini_hochberg(p_values)
    assert adjusted[1] < adjusted[2] < adjusted[0]


# --------------------------------------------------------------------------------------
# budget safety: the irreversible part
# --------------------------------------------------------------------------------------


def test_plan_costs_nothing_and_reports_exact_query_count():
    with tempfile.TemporaryDirectory() as raw:
        store = client.AnswerStore(Path(raw) / "answers.jsonl")
        prompts = [("0", "p0"), ("1", "p1"), ("2", "p2")]
        result = client.plan(prompts, ["ep-01", "ep-02"], store)
        assert result["total_queries"] == 6
        assert result["per_endpoint"] == {"ep-01": 3, "ep-02": 3}


def test_plan_skips_answers_already_collected():
    with tempfile.TemporaryDirectory() as raw:
        store = client.AnswerStore(Path(raw) / "answers.jsonl")
        store.append({"endpoint": "ep-01", "id": "0", "repeat": 0, "prompt": "p0", "response": "x"})
        result = client.plan([("0", "p0")], ["ep-01", "ep-02"], store)
        # ep-01 already answered for prompt 0, ep-02 has not.
        assert result["total_queries"] == 1
        assert result["per_endpoint"] == {"ep-01": 0, "ep-02": 1}


def test_plan_respects_max_per_endpoint():
    with tempfile.TemporaryDirectory() as raw:
        store = client.AnswerStore(Path(raw) / "answers.jsonl")
        prompts = [(str(i), f"p{i}") for i in range(10)]
        result = client.plan(prompts, ["ep-01"], store, max_per_endpoint=4)
        assert result["total_queries"] == 4


def test_answer_store_survives_a_truncated_line():
    """A killed run leaves a partial final line; it must not destroy the usable answers."""
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "answers.jsonl"
        path.write_text(
            json.dumps({"endpoint": "ep-01", "id": "0", "repeat": 0, "prompt": "p", "response": "ok"})
            + "\n"
            + '{"endpoint": "ep-01", "id": "1", "rep'  # truncated
        )
        store = client.AnswerStore(path)
        assert len(store.rows) == 1
        assert store.rows[0]["response"] == "ok"


def test_answer_store_counts_by_endpoint():
    with tempfile.TemporaryDirectory() as raw:
        store = client.AnswerStore(Path(raw) / "answers.jsonl")
        for index in range(3):
            store.append({"endpoint": "ep-01", "id": str(index), "repeat": 0, "prompt": "p", "response": "x"})
        store.append({"endpoint": "ep-02", "id": "0", "repeat": 0, "prompt": "p", "response": "y"})
        assert store.count_by_endpoint() == {"ep-01": 3, "ep-02": 1}


def test_budget_remaining_never_negative():
    budget = client.EndpointBudget(endpoint="ep-01", limit=100, used=100)
    assert budget.remaining == 0
    budget.reserved = 5
    assert budget.remaining == 0


# --------------------------------------------------------------------------------------
# end-to-end on the published trap structure
# --------------------------------------------------------------------------------------


def test_practice_truth_encodes_the_real_trap_structure():
    """The published answers are the specification; assert the structure explicitly.

    Same-model disguises are system prompt, persona, watermark, temperature. LoRA at ANY strength
    and quantization are different models. Getting this backwards is the main way to fail.
    """
    truth = _truth()
    same = {"ref-03", "ref-05", "ref-06", "ref-08"}
    different = {"ref-02", "ref-04", "ref-07", "ref-09"}
    reference_model = truth["ref-01"][1]
    for endpoint in same:
        assert truth[endpoint][1] == reference_model, f"{endpoint} should be the same model"
    for endpoint in different:
        assert truth[endpoint][1] != reference_model, f"{endpoint} should be a different model"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pipeline_recovers_models_with_enough_prompts(seed):
    truth = _truth()
    data = synthetic_dataset(truth, n_prompts=30, seed=seed)
    labels, _tests = decide(data, sorted(truth), seed=seed)
    result = evaluate_against_truth(truth, labels)
    assert result["mcc_model"] == pytest.approx(1.0), "all same/different calls must be correct"


def test_pipeline_is_conservative_with_too_few_prompts():
    """With 5 prompts the method must not invent same-model pairs (false alarms are costly)."""
    truth = _truth()
    data = synthetic_dataset(truth, n_prompts=5, seed=0)
    labels, _tests = decide(data, sorted(truth), seed=0)
    result = evaluate_against_truth(truth, labels)
    assert result["model_pairs"]["fp"] == 0


def test_similarity_tensor_only_uses_prompts_both_endpoints_answered():
    by_prompt = {
        "q0": {"ep-01": "alpha beta gamma delta", "ep-02": "alpha beta gamma delta"},
        "q1": {"ep-01": "alpha beta gamma delta"},  # ep-02 missing
    }
    tensors = stats.build_similarity_tensor(by_prompt, ["ep-01", "ep-02"])
    assert tensors[("ep-01", "ep-02")].size == 1


def test_coverage_reports_uneven_collection():
    by_prompt = {
        "q0": {"ep-01": "a", "ep-02": "b"},
        "q1": {"ep-01": "a"},
    }
    cov = stats.coverage(by_prompt, ["ep-01", "ep-02"])
    assert cov["per_endpoint"] == {"ep-01": 2, "ep-02": 1}
    assert cov["complete_prompts"] == 1
