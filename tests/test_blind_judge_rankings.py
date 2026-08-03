import pandas as pd
import pytest

from scripts.blind_judge_rankings import (
    CallBudget,
    CandidateJudgment,
    JudgeResult,
    SECTION_MAXIMA,
    SectionEvidence,
    SectionScores,
    _agreement,
    _agreement_validation,
    _overall_score,
    _split_batch,
    _validate_judgments,
    _verdict,
)


def _judgment(candidate_key="J01", preferred_signals=None):
    scores = SectionScores(
        role_level_fit=12,
        uae_compliance_payroll_pro=18,
        core_hr_operations=12,
        hr_systems_office_tools=6,
        industrial_context=8,
        tenure_continuity=6,
        career_relevance_mix=5,
        preferred_signals=6,
        evidence_quality=4,
    )
    return CandidateJudgment(
        candidate_key=candidate_key,
        section_scores=scores,
        section_evidence_present=SectionEvidence(
            **{section: True for section in scores.model_dump()}
        ),
        credited_preferred_signals=preferred_signals or ["english"],
        strengths=[],
        gaps=[],
        rationale="Explicit evidence only.",
    )


def test_overall_score_and_verdict_are_derived_from_sections():
    judgment = _judgment()

    assert _overall_score(judgment) == 77
    assert _verdict(_overall_score(judgment)) == "strong"


def test_section_maxima_sum_to_100():
    assert sum(SECTION_MAXIMA.values()) == 100


def test_phantom_tagalog_credit_is_rejected():
    result = JudgeResult(
        evaluations=[_judgment(preferred_signals=["tagalog_or_filipino"])]
    )

    with pytest.raises(ValueError, match="phantom Tagalog/Filipino credit"):
        _validate_judgments(
            "fake-judge",
            {"J01"},
            result,
            {"J01": {"languages": []}},
        )


def test_tagalog_credit_accepts_explicit_filipino_alias():
    result = JudgeResult(
        evaluations=[_judgment(preferred_signals=["tagalog_or_filipino"])]
    )

    _validate_judgments(
        "fake-judge",
        {"J01"},
        result,
        {"J01": {"languages": [{"language": "Tagalog"}]}},
    )


def test_agreement_uses_fixed_structural_subset_and_enforces_gates():
    rows = []
    for index in range(6):
        for judge, offset in (("left", 0.0), ("right", 0.5)):
            row = {
                "candidate_id": f"candidate-{index}",
                "judge": judge,
                "overall_score": 50 + index + offset,
            }
            for section in SectionScores.model_fields:
                row[f"{section}_score"] = index + offset
                row[f"{section}_evidence_present"] = True
            rows.append(row)

    agreement = _agreement(
        pd.DataFrame(rows),
        ["left", "right"],
        {f"candidate-{index}" for index in range(4)},
    )
    validation = _agreement_validation(agreement)

    assert agreement["sections"]["tenure_continuity"]["fixed_subset"]["n"] == 4
    assert agreement["sections"]["preferred_signals"]["fixed_subset"]["n"] == 6
    assert validation["overall_pass"] is True
    assert all(result["pass"] for result in validation["section_gates"].values())


def test_section_disagreement_blocks_component_not_anchor():
    agreement = {
        "overall": {"n": 78, "spearman": 0.90, "mae": 5.0},
        "sections": {
            "tenure_continuity": {
                "fixed_subset": {"n": 76, "spearman": 0.20, "mae": 2.5}
            },
            "career_relevance_mix": {
                "fixed_subset": {"n": 76, "spearman": 0.70, "mae": 1.0}
            },
            "preferred_signals": {
                "fixed_subset": {"n": 78, "spearman": 0.70, "mae": 1.0}
            },
        },
    }

    validation = _agreement_validation(agreement)

    assert validation["overall_pass"] is True
    assert validation["all_pass"] is True
    assert validation["all_section_gates_pass"] is False
    assert validation["section_gates"]["tenure_continuity"]["pass"] is False


def test_call_budget_persists_across_resumes(tmp_path):
    state_path = tmp_path / "call_budget.json"
    first = CallBudget(limit=3, state_path=state_path, prior_used=2)

    first.claim("claude-opus-4.8", "batch 1")
    resumed = CallBudget(limit=3, state_path=state_path)

    assert resumed.used == 3
    with pytest.raises(RuntimeError, match="live call cap reached"):
        resumed.claim("claude-opus-4.8", "batch 2")


def test_split_batch_preserves_order_and_membership():
    batch = [{"candidate_key": f"J{index:02d}"} for index in range(1, 8)]

    parts = _split_batch(batch, enabled=True, split_size=4)

    assert [len(part) for part in parts] == [4, 3]
    assert [item for part in parts for item in part] == batch
    assert _split_batch(batch, enabled=False, split_size=4) == [batch]