import pytest
from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from evals.cases import EvalCase
from evals.runner import PipelineConfig, aggregate, evaluate_cases
from models.data_models import Company, JobRoleSchema, Skill
from models.enums import ImportanceLevel

LINKEDIN = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"


@pytest.fixture(scope="module")
def model():
    return SentenceTransformer("all-mpnet-base-v2")


@pytest.fixture(scope="module")
def profiles():
    return LinkedInAdapter().to_profiles(LINKEDIN)


def test_runner_ranks_and_finds_seed(model, profiles):
    # Build a JD directly from a well-populated seed (no LLM), then confirm the
    # runner ranks the seed near the top and computes the metric aggregate.
    seed = next(p for p in profiles if len(p.skills) >= 5 and p.summary)
    jd = JobRoleSchema(
        role=seed.job_title,
        company=Company(name="X"),
        responsibilities=[seed.responsibilities or "HR administration"][:1],
        skills=[Skill(skill=s, priority=ImportanceLevel.ESSENTIAL, proficiency_level=None) for s in seed.skills[:5]],
    )
    case = EvalCase(
        case_id="t", dataset="linkedin", jd_text="", parsed_jd=jd,
        relevance={seed.candidate_id: 1.0}, source="reverse_match", seed_id=seed.candidate_id,
    )

    per_case = evaluate_cases([case], profiles, model, PipelineConfig(title_mode="hybrid"))
    row = per_case[0]

    assert row["n_ranked"] > 0
    assert row["seed_rank"] is not None            # seed survived the gate and was scored
    assert row["seed_rank"] <= 20                  # a self-derived JD should rank the seed high

    summary = aggregate(per_case)
    assert summary["n_reverse_match"] == 1
    for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "seed_found_rate"):
        assert key in summary


def test_aggregate_mixes_reverse_and_gold():
    per_case = [
        {"case_id": "a", "source": "reverse_match", "n_ranked": 100, "hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "rr": 1.0, "seed_rank": 1},
        {"case_id": "b", "source": "reverse_match", "n_ranked": 100, "hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "rr": 0.5, "seed_rank": 2},
        {"case_id": "gold", "source": "llm_scored", "n_ranked": 145, "hit@1": 1.0, "hit@3": 1.0, "hit@5": 1.0, "hit@10": 1.0, "rr": 1.0, "ndcg@5": 0.8, "ndcg@10": 0.75},
    ]
    summary = aggregate(per_case)
    assert summary["n_reverse_match"] == 2 and summary["n_gold"] == 1
    assert summary["hit@1"] == 0.5 and summary["mrr"] == 0.75
    assert summary["ndcg@5"] == 0.8 and summary["ndcg@10"] == 0.75
