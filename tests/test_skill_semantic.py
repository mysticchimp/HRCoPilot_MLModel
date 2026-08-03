"""Tests for the semantic / hybrid skill matcher.

The fast tests inject hand-built unit vectors so they need no model. One
integration test loads all-mpnet to prove hybrid recovers a real synonym pair
that the char-level fuzzy matcher is blind to.
"""

import numpy as np
import pytest
from sentence_transformers import SentenceTransformer

from core.matching import weighted_fuzzy_skill_score
from core.skill_normalization import SkillSemanticIndex, build_skill_semantic_index
from models.data_models import Skill
from models.enums import ImportanceLevel


def _unit(vec):
    arr = np.array(vec, dtype=float)
    return arr / np.linalg.norm(arr)


# recruitment & talent acquisition are near (cos 0.8); welding is orthogonal.
_STUB_INDEX = SkillSemanticIndex({
    "recruitment": _unit([1.0, 0.0]),
    "talent acquisition": _unit([0.8, 0.6]),
    "welding": _unit([0.0, 1.0]),
})


def _essential(name: str) -> list[Skill]:
    return [Skill(skill=name, priority=ImportanceLevel.ESSENTIAL, proficiency_level=None)]


def test_semantic_index_cosine_lookup():
    assert _STUB_INDEX.similarity("talent acquisition", "recruitment") == pytest.approx(0.8)
    assert _STUB_INDEX.similarity("welding", "recruitment") == 0.0  # orthogonal
    assert _STUB_INDEX.similarity("unknown", "recruitment") == 0.0  # missing key
    # keys are looked up via normalize_skill, so raw casing/spacing still resolves
    assert _STUB_INDEX.similarity("Talent Acquisition", "Recruitment") == pytest.approx(0.8)


def test_build_index_empty_is_safe_without_model():
    idx = build_skill_semantic_index(["", None, 123], model=None)  # model untouched when empty
    assert idx.similarity("recruitment", "recruitment") == 0.0


def test_hybrid_recovers_synonym_fuzzy_misses():
    jd = _essential("Talent Acquisition")
    cand = ["Recruitment"]

    fuzzy_only = weighted_fuzzy_skill_score("c", jd, cand)
    assert fuzzy_only["score"] == 0.0  # char-level fuzzy can't bridge the synonym

    hybrid = weighted_fuzzy_skill_score(
        "c", jd, cand, semantic_index=_STUB_INDEX, semantic_threshold=0.5
    )
    assert hybrid["score"] == pytest.approx(0.8)  # graded by cosine
    assert "Recruitment" in hybrid["matched_skills"]


def test_semantic_threshold_gates_weak_matches():
    jd = _essential("Talent Acquisition")
    gated = weighted_fuzzy_skill_score(
        "c", jd, ["Recruitment"], semantic_index=_STUB_INDEX, semantic_threshold=0.9
    )
    assert gated["score"] == 0.0  # 0.8 cosine is below the 0.9 floor


def test_pure_semantic_mode_ignores_fuzzy():
    # A near-perfect fuzzy match still scores 0 when fuzzy is disabled and the pair
    # is semantically unknown to the index.
    jd = _essential("Payroll")
    result = weighted_fuzzy_skill_score(
        "c", jd, ["Payrol"], semantic_index=_STUB_INDEX,
        semantic_threshold=0.5, include_fuzzy=False,
    )
    assert result["score"] == 0.0


def test_semantic_index_none_is_backward_compatible():
    # Default path (no index) must behave exactly like the original fuzzy matcher.
    java = weighted_fuzzy_skill_score("c", _essential("Java"), ["JavaScript"])
    assert java["score"] == 0.0
    typo = weighted_fuzzy_skill_score("c", _essential("Python"), ["Pythn"])
    assert 0.85 < typo["score"] < 1.0


def test_hybrid_recovers_synonym_with_real_model():
    model = SentenceTransformer("all-mpnet-base-v2")
    idx = build_skill_semantic_index(["recruitment", "talent acquisition"], model)
    cos = idx.similarity("talent acquisition", "recruitment")
    # all-mpnet compresses skill-phrase similarity: genuine synonyms land ~0.45,
    # not the ~0.7+ one might expect. This is why the semantic floor must be low.
    assert 0.40 < cos < 0.55

    jd = _essential("Recruitment")
    fuzzy = weighted_fuzzy_skill_score("c", jd, ["Talent Acquisition"])["score"]
    hybrid = weighted_fuzzy_skill_score(
        "c", jd, ["Talent Acquisition"], semantic_index=idx, semantic_threshold=0.40
    )["score"]
    assert fuzzy == 0.0 < hybrid
