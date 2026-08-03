import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from core.jd_extraction import process_jd
from core.jd_generation import generate_jd_from_profile
from core.llm.base import LLMProvider
from models.candidate import CandidateProfile
from models.data_models import JobRoleSchema

FIXTURE_DIR = "evals/fixtures"
GOLD_SCORED_CSV = "data/Scored_FullPool_HR_Assistant_v2_2026-07-06_1843.csv"
BLIND_JUDGMENTS_CSV = "evals/judgments/blind_judgments_hr_assistant.csv"
GOLD_PARSED_JD = "jd/parsed/hr_assistant_prime_ac.json"
JDGEN_MODEL = "claude-opus-4.7"
PROMPT_VERSION = "v1"
GOLD_JUDGE_MODELS = ["claude-opus-4.8", "gpt-5.5"]
GOLD_RUBRIC_VERSION = "t3-tenure-relevance-language-v1"


@dataclass
class EvalCase:
    case_id: str
    dataset: str
    jd_text: str
    parsed_jd: JobRoleSchema
    relevance: dict          # candidate_id -> grade (grade > 0 means relevant)
    source: str              # "reverse_match" | "llm_scored"
    seed_id: str | None = None


def reduced_profile_payload(profile: CandidateProfile) -> dict:
    """Profile view fed to JD generation.

    Explicit skills, education, seniority, and years_experience are HELD OUT so the
    generated JD can't trivially echo the seed's own metadata. The pipeline (and its
    seniority_score / experience_score components) must recover the required level
    from title / responsibilities / summary instead (leakage mitigation).
    """
    return {
        "job_title": profile.job_title,
        "summary": profile.summary,
        "responsibilities": profile.responsibilities,
    }


def sample_seeds(profiles, n_per_group: int, group_key, seed: int = 13) -> list[CandidateProfile]:
    """Stratified sample: up to n_per_group profiles from each group (deterministic)."""
    rng = random.Random(seed)
    groups: dict = defaultdict(list)
    for profile in profiles:
        groups[group_key(profile)].append(profile)
    seeds = []
    for key in sorted(groups, key=lambda k: str(k)):
        members = groups[key][:]
        rng.shuffle(members)
        seeds.extend(members[:n_per_group])
    return seeds


def _fixture_path(dataset: str, key: str) -> str:
    return os.path.join(FIXTURE_DIR, dataset, f"{key}.json")


def load_fixture(dataset: str, key: str) -> EvalCase | None:
    path = _fixture_path(dataset, key)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        data = json.load(fh)
    return EvalCase(
        case_id=data["case_id"],
        dataset=data["dataset"],
        jd_text=data["jd_text"],
        parsed_jd=JobRoleSchema.model_validate(data["parsed_jd"]),
        relevance=data["relevance"],
        source=data["source"],
        seed_id=data.get("seed_id"),
    )


def save_fixture(case: EvalCase, key: str, meta: dict | None = None) -> None:
    path = _fixture_path(case.dataset, key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "jd_text": case.jd_text,
        "parsed_jd": case.parsed_jd.model_dump(mode="json"),
        "relevance": case.relevance,
        "source": case.source,
        "seed_id": case.seed_id,
        **(meta or {}),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def build_reverse_match_case(
    profile: CandidateProfile, provider: LLMProvider, dataset: str, reload: bool = False
) -> EvalCase:
    """Seed -> synthetic JD -> case with relevance {seed: 1}. Cached to disk."""
    if not reload:
        cached = load_fixture(dataset, profile.candidate_id)
        if cached is not None:
            return cached
    jd_text = generate_jd_from_profile(reduced_profile_payload(profile), provider=provider)
    parsed_jd = process_jd(jd_text, provider=provider)
    # De-leak location: the seed's location is held OUT of JD generation
    # (reduced_profile_payload omits it), but the LLM still invents a generic location
    # (e.g. "Dubai") that isn't tied to the seed. Scoring against that invented value
    # unfairly penalizes the seed on location_score, so reverse-match holds location out
    # entirely — it is structurally unable to evaluate location fit. The gold case (a real
    # posting with a real location) is the anchor that calibrates the location weight.
    parsed_jd.location = None
    # De-leak language for the same reason: candidate languages are held out of
    # reduced_profile_payload, yet generated reverse JDs frequently invent language
    # requirements. They are unrelated to the seed and cannot fairly evaluate language fit.
    parsed_jd.language_proficiency = None
    case = EvalCase(
        case_id=f"{dataset}:{profile.candidate_id}",
        dataset=dataset,
        jd_text=jd_text,
        parsed_jd=parsed_jd,
        relevance={profile.candidate_id: 1.0},
        source="reverse_match",
        seed_id=profile.candidate_id,
    )
    save_fixture(case, profile.candidate_id, {"model": JDGEN_MODEL, "prompt_version": PROMPT_VERSION})
    return case


def build_linkedin_gold_case(
    profiles, jd_path: str, provider: LLMProvider, reload: bool = False, key: str = "_gold_hr_assistant"
) -> EvalCase:
    """Silver LinkedIn case: JD text + blind 2-judge consensus Judge grades.

    Labels were repointed from the single-LLM fit_0_10 (Scored_FullPool) to the blind
    two-judge grades (evals/judgments/blind_judgments_hr_assistant.csv) — a stronger silver anchor the
    pipeline agrees with far more (NDCG@10 ~0.95 vs ~0.69). Graded set = the blind top-50 union;
    ungraded candidates are absent (treated as 0 relevance), so this is a HEAD anchor (NDCG@10/20).
    """
    if not reload:
        cached = load_fixture("linkedin", key)
        if cached is not None:
            return cached
    judged = pd.read_csv(BLIND_JUDGMENTS_CSV)
    id_to_score = dict(zip(judged["candidate_id"], judged["judge_mean_score"]))
    relevance = {
        profile.candidate_id: float(id_to_score[profile.candidate_id])
        for profile in profiles if profile.candidate_id in id_to_score
    }
    with open(jd_path) as fh:
        jd_text = fh.read()
    if os.path.exists(GOLD_PARSED_JD):
        with open(GOLD_PARSED_JD) as fh:
            parsed_jd = JobRoleSchema.model_validate(json.load(fh))
    else:
        parsed_jd = process_jd(jd_text, provider=provider)
    case = EvalCase(
        case_id="linkedin:gold_hr_assistant",
        dataset="linkedin",
        jd_text=jd_text,
        parsed_jd=parsed_jd,
        relevance=relevance,
        source="llm_scored",
        seed_id=None,
    )
    save_fixture(
        case,
        key,
        {
            "label_source": BLIND_JUDGMENTS_CSV,
            "judge_models": GOLD_JUDGE_MODELS,
            "rubric_version": GOLD_RUBRIC_VERSION,
            "anchor_kind": "silver_judge_grade",
            "jd_source": jd_path,
        },
    )
    return case
