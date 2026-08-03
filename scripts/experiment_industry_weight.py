"""Experiment: HVAC industry weighted 'essential' (current jd/parsed) vs the champion's
'valuable'. Does boosting HVAC make the pipeline rank more like the original single-LLM —
and at what cost to agreement with the blind judges?

Outputs go to evals/experiments/ (kept SEPARATE from the champion deliverables in
evals/results/). Reads the JD from jd/parsed/ as-is, so edit the industry priority there.

    COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/experiment_industry_weight.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.data import profiles_to_dataframe
from core.embedding import build_similarity_spec, embed_profiles
from core.filtering import filter_by_job_title
from core.scoring import (
    calculate_experience_score,
    calculate_industry_score,
    calculate_language_score,
    calculate_location_score,
    calculate_qualification_score,
    calculate_seniority_score,
    calculate_similarity_score,
    calculate_skill_score,
    calculate_total_score,
)
from models.data_models import JobRoleSchema
from models.mappings import similarity_model_config

LINKEDIN_CSV = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"
GOLD_CSV = "data/Scored_FullPool_HR_Assistant_v2_2026-07-06_1843.csv"
JD_STORE = "jd/parsed/hr_assistant_prime_ac.json"
JUDGMENTS = "evals/judgments/blind_judgments_hr_assistant.csv"
CHAMPION_VS_LLM = "evals/results/pipeline_vs_llm_hr_assistant.csv"  # champion = HVAC valuable
OUT_DIR = "evals/experiments"
TAG = "hvac_essential"


def _sp(a, b):
    return pd.Series(list(a)).corr(pd.Series(list(b)), method="spearman")


def score_pool(profiles, jd, model, sim_spec):
    df = profiles_to_dataframe(profiles)
    df = filter_by_job_title(df, jd.role, threshold=0.4, model=model, mode="hybrid", hard=False)
    df = calculate_skill_score(df, jd, model=model, skill_mode="hybrid")
    df = calculate_qualification_score(df, jd)
    df = calculate_seniority_score(df, jd)
    df = calculate_experience_score(df, jd)
    df = calculate_industry_score(df, jd)
    df = calculate_language_score(df, jd)
    df = calculate_location_score(df, jd)
    df = calculate_similarity_score(df, jd, sim_spec.model, query_instruction=sim_spec.query_instruction)
    df = calculate_total_score(df, jd)
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    df["pipeline_rank"] = df.index + 1
    return df


def main():
    jd = JobRoleSchema.model_validate(json.load(open(JD_STORE)))
    ind = {i.industry: i.priority.value for i in (jd.experience.industry_experience or [])}
    print(f"JD industry_experience priorities: {ind}\n")

    profiles = LinkedInAdapter().to_profiles(LINKEDIN_CSV)
    model = SentenceTransformer("all-mpnet-base-v2")
    sim_spec = build_similarity_spec(similarity_model_config, base_model=model)
    slug = similarity_model_config["model_name"].replace("/", "_")
    embed_profiles(profiles, sim_spec.model, cache_path=f".ai-recruiter/emb_linkedin_v2_{slug}.pkl",
                   model_key=sim_spec.model_key, doc_instruction=sim_spec.doc_instruction,
                   batch_size=sim_spec.batch_size)
    df = score_pool(profiles, jd, model, sim_spec)

    judges = pd.read_csv(JUDGMENTS)[["candidate_id", "consensus_rank"]]
    gold = pd.read_csv(GOLD_CSV, skiprows=2)[["linkedinUrl", "rank"]].rename(columns={"rank": "llm_rank"})
    m = df.merge(gold, left_on="linkedin_url", right_on="linkedinUrl", how="inner")
    mj = df.merge(judges, on="candidate_id", how="inner")

    exp_llm = _sp(m["pipeline_rank"], m["llm_rank"])
    exp_judge = _sp(mj["pipeline_rank"], mj["consensus_rank"])

    # champion (HVAC=valuable) baseline from the committed deliverables
    ch = pd.read_csv(CHAMPION_VS_LLM)
    chj = pd.read_csv(JUDGMENTS)
    ch_llm = _sp(ch["pipeline_rank"], ch["llm_rank"])
    ch_judge = _sp(chj["pipeline_rank"], chj["consensus_rank"])

    os.makedirs(OUT_DIR, exist_ok=True)
    run_cols = ["pipeline_rank", "candidate_id", "job_title", "total_score",
                "industry_score", "skill_score", "similarity_score", "linkedin_url"]
    df[run_cols].to_csv(f"{OUT_DIR}/pipeline_run_hr_assistant_{TAG}.csv", index=False)
    (m.sort_values("pipeline_rank")[["pipeline_rank", "candidate_id", "llm_rank", "industry_score", "total_score"]]
       .to_csv(f"{OUT_DIR}/pipeline_vs_llm_hr_assistant_{TAG}.csv", index=False))

    print("=== Spearman:      pipeline vs LLM   |   pipeline vs blind judges ===")
    print(f"  champion  (HVAC valuable) :  LLM {ch_llm:+.3f}      judges {ch_judge:+.3f}")
    print(f"  experiment(HVAC essential):  LLM {exp_llm:+.3f}      judges {exp_judge:+.3f}")
    print(f"  delta                     :  LLM {exp_llm - ch_llm:+.3f}      judges {exp_judge - ch_judge:+.3f}")
    print(f"\nwrote {OUT_DIR}/pipeline_run_hr_assistant_{TAG}.csv")
    print(f"wrote {OUT_DIR}/pipeline_vs_llm_hr_assistant_{TAG}.csv")


if __name__ == "__main__":
    main()
