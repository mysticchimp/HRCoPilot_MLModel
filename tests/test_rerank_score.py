import numpy as np
import pandas as pd
import pytest

from core import scoring
from core.embedding import build_rerank_candidate_text, build_rerank_jd_text
from core.reranking import sigmoid
from core.scoring import apply_rerank
from models.candidate import (
    CandidateEducation,
    CandidateLanguage,
    CandidateLocation,
    CandidateProfile,
)
from models.data_models import Company, JobRoleSchema, Skill
from models.enums import ImportanceLevel


def _jd():
    return JobRoleSchema(
        role="HR Assistant",
        company=Company(name="Prime Focus Group"),
        responsibilities=["Support payroll and HR administration"],
        skills=[Skill(skill="Payroll", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)],
    )


def _stage1_df():
    """5 candidates already Stage-1 scored. title/skill are equal across the pool, so
    within the reranked Head the order follows rerank_score alone. rerank_text encodes
    each candidate's intended rerank score, so the fake scorer is order-independent."""
    n = 5
    return pd.DataFrame({
        "candidate_id": ["C0", "C1", "C2", "C3", "C4"],
        "title_score": [0.5] * n,
        "skill_score": [0.5] * n,
        "similarity_score": [0.5] * n,
        "total_score": [0.9, 0.8, 0.7, 0.6, 0.5],   # Stage-1 order C0..C4
        "rerank_text": ["0.10", "0.95", "0.50", "0.30", "0.20"],
    })


def _fake_scores(spec, jd_text, cand_texts, cache_path=None):
    # decode the intended rerank score straight from rerank_text (order-independent)
    return np.array([float(t) for t in cand_texts])


def test_sigmoid_range_and_monotonic():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    vals = sigmoid(np.array([-10.0, 0.0, 10.0]))
    assert (vals >= 0).all() and (vals <= 1).all()
    assert vals[0] < vals[1] < vals[2]


def test_temperature_spreads_saturated_logits():
    hi = np.array([5.0, 8.0])                       # both ~1.0 under plain sigmoid
    gap_t1 = abs(float(np.diff(sigmoid(hi, 1.0))[0]))
    gap_t5 = abs(float(np.diff(sigmoid(hi, 5.0))[0]))
    assert gap_t5 > gap_t1                           # higher T de-saturates a congested head


def test_apply_rerank_none_is_stage1_order():
    out = apply_rerank(_stage1_df(), _jd(), None, top_k=3)
    assert out["candidate_id"].tolist() == ["C0", "C1", "C2", "C3", "C4"]
    assert "rerank_score" not in out.columns


def test_apply_rerank_reorders_head_only(monkeypatch):
    monkeypatch.setattr(scoring, "rerank_scores", _fake_scores)
    out = apply_rerank(_stage1_df(), _jd(), object(), top_k=3)
    # Head {C0,C1,C2} reordered by rerank (0.95,0.50,0.10); tail {C3,C4} untouched.
    assert out["candidate_id"].tolist() == ["C1", "C2", "C0", "C3", "C4"]


def test_apply_rerank_frozen_membership(monkeypatch):
    monkeypatch.setattr(scoring, "rerank_scores", _fake_scores)
    out = apply_rerank(_stage1_df(), _jd(), object(), top_k=3)
    order = out["candidate_id"].tolist()
    # C0 (CE score 0.10) sinks within the Head but never crosses into the tail.
    assert order.index("C0") < order.index("C3")
    # tail keeps its Stage-1 relative order
    assert order.index("C3") < order.index("C4")
    r = out.set_index("candidate_id")["rerank_score"]
    assert not np.isnan(r["C1"]) and np.isnan(r["C3"])


def test_apply_rerank_top_k_clamped_to_pool(monkeypatch):
    monkeypatch.setattr(scoring, "rerank_scores", _fake_scores)
    out = apply_rerank(_stage1_df(), _jd(), object(), top_k=99)   # whole pool reranked
    assert out["candidate_id"].tolist() == ["C1", "C2", "C3", "C4", "C0"]


def test_build_rerank_jd_text_includes_key_fields():
    txt = build_rerank_jd_text(_jd())
    assert "Role: HR Assistant" in txt
    assert "Payroll" in txt


def test_build_rerank_candidate_text_is_rich():
    prof = CandidateProfile(
        candidate_id="c1", job_title="HR Officer",
        skills=["Payroll", "WPS"], summary="Experienced HR admin",
        responsibilities="Handled visas and payroll",
        education=[CandidateEducation(degree="MBA", field="HR")],
        languages=[CandidateLanguage(language="Arabic")],
        certifications=["CIPD"], employers=["ACME"],
        location=CandidateLocation(city="Dubai", country="United Arab Emirates"),
        years_experience=4, seniority="mid",
    )
    txt = build_rerank_candidate_text(prof)
    for token in ["HR Officer", "Payroll", "MBA", "Arabic", "CIPD", "ACME", "Dubai"]:
        assert token in txt
