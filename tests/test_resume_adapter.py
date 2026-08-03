import pandas as pd

from core.adapters.resume_adapter import ResumeAdapter
from core.data import process_qualifications, process_skills
from core.embedding import build_candidate_embedding_input
from core.matching import weighted_fuzzy_qualification_score, weighted_fuzzy_skill_score
from models.data_models import Education, Skill
from models.enums import ImportanceLevel

CSV = "./data/resume_data.csv"


def _legacy_df(n: int) -> pd.DataFrame:
    df = pd.read_csv(CSV, encoding="utf-8-sig").head(n).copy()
    df.columns = df.columns.str.replace("\ufeff", "").str.strip()
    process_skills(df)
    process_qualifications(df)
    return df


def test_resume_adapter_skill_union_parity():
    # ResumeAdapter must reproduce the legacy combined skill set
    n = 50
    legacy = _legacy_df(n)
    profiles = ResumeAdapter().to_profiles(CSV)[:n]
    assert len(profiles) == n
    for p, (_, row) in zip(profiles, legacy.iterrows()):
        assert set(p.skills) == set(row["all_skills"])


def test_resume_adapter_scoring_parity():
    # The adapter representation must yield identical skill AND qualification
    # scores to the legacy dataframe representation, for a fixed synthetic JD.
    n = 60
    legacy = _legacy_df(n)
    profiles = ResumeAdapter().to_profiles(CSV)[:n]

    jd_skills = [
        Skill(skill="Python", priority=ImportanceLevel.ESSENTIAL, proficiency_level=None),
        Skill(skill="Java", priority=ImportanceLevel.IMPORTANT, proficiency_level=None),
        Skill(skill="Machine Learning", priority=ImportanceLevel.VALUABLE, proficiency_level=None),
    ]
    jd_edu = [Education(degree="Bachelor", field="Computer Science", priority=ImportanceLevel.ESSENTIAL)]

    for p, (_, row) in zip(profiles, legacy.iterrows()):
        legacy_skill = weighted_fuzzy_skill_score(p.candidate_id, jd_skills, row["all_skills"])
        adapter_skill = weighted_fuzzy_skill_score(p.candidate_id, jd_skills, p.skills)
        assert legacy_skill["score"] == adapter_skill["score"]

        legacy_qual = weighted_fuzzy_qualification_score(
            p.candidate_id, jd_edu,
            {"degrees": list(row["degree_names_norm"]), "fields": list(row["major_field_of_studies"])},
        )
        adapter_qual = weighted_fuzzy_qualification_score(
            p.candidate_id, jd_edu,
            {"degrees": p.degree_names, "fields": p.field_names},
        )
        assert legacy_qual["score"] == adapter_qual["score"]


def test_resume_adapter_ids_and_source():
    profiles = ResumeAdapter().to_profiles(CSV)[:5]
    assert [p.candidate_id for p in profiles] == ["C001", "C002", "C003", "C004", "C005"]
    assert all(p.source == "resume" for p in profiles)


def test_resume_adapter_embedding_text_no_nan_pollution():
    # ~50% of rows have a NaN career_objective; the adapter must not leak "nan"
    profiles = ResumeAdapter().to_profiles(CSV)[:300]
    for p in profiles:
        text = build_candidate_embedding_input(p)
        assert "nan" not in text.lower().split()
        if p.summary:
            assert p.summary in text
        if p.responsibilities:
            assert p.responsibilities in text
