"""Real pipeline run: score all 145 LinkedIn candidates against the HR-Assistant JD,
dump the ranking, and compare it to the LLM-scored Scored_FullPool ranking.

Uses the JD already extracted from the real posting
(jd/HR Assistant — Prime Focus Group (Prime AC).md) and cached in the gold fixture,
so this run is deterministic and needs no LLM call.

    COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/run_hr_assistant.py
"""

import argparse
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
from core.reranking import build_rerank_model
from core.scoring import (
    apply_rerank,
    calculate_attrition_score,
    calculate_education_relevance_score,
    calculate_experience_relevance_score,
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
from evals.cases import load_fixture
from evals.metrics import ndcg_at_k
from models.data_models import JobRoleSchema
from models.mappings import RERANK_GTE, candidate_score_weights, rerank_top_k, similarity_model_config

LINKEDIN_CSV = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"
GOLD_CSV = "data/Scored_FullPool_HR_Assistant_v2_2026-07-06_1843.csv"
JD_STORE = "jd/parsed/hr_assistant_prime_ac.json"
CACHE = ".ai-recruiter/emb_linkedin_v2.pkl"
RUN_OUT = "evals/results/pipeline_run_hr_assistant.csv"
CMP_OUT = "evals/results/pipeline_vs_llm_hr_assistant.csv"


def load_jd(gold):
    """Prefer the editable JD store (jd/parsed/...); fall back to the gold fixture."""
    if os.path.exists(JD_STORE):
        with open(JD_STORE) as fh:
            return JobRoleSchema.model_validate(json.load(fh))
    return gold.parsed_jd


def score_pool(profiles, jd, model, sim_spec=None, rerank_spec=None, rerank_top_k=50, rerank_cache_path=None):
    df = profiles_to_dataframe(profiles)
    df = filter_by_job_title(df, jd.role, threshold=0.4, model=model, mode="hybrid", hard=False)
    df = calculate_skill_score(df, jd, model=model, skill_mode="hybrid")
    df = calculate_qualification_score(df, jd)
    df = calculate_seniority_score(df, jd)
    df = calculate_experience_score(df, jd)
    df = calculate_industry_score(df, jd)
    df = calculate_language_score(df, jd)
    df = calculate_location_score(df, jd)
    df = calculate_attrition_score(df, jd)
    df = calculate_experience_relevance_score(df, jd)
    df = calculate_education_relevance_score(df, jd)
    df = calculate_similarity_score(
        df, jd,
        sim_spec.model if sim_spec else model,
        query_instruction=sim_spec.query_instruction if sim_spec else None,
    )
    df = calculate_total_score(df, jd)
    df = apply_rerank(df, jd, rerank_spec, top_k=rerank_top_k, cache_path=rerank_cache_path)
    df = df.reset_index(drop=True)
    df["pipeline_rank"] = df.index + 1
    return df


def spearman(a, b):
    return pd.Series(a).corr(pd.Series(b), method="spearman")


def kendall(a, b):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(a, b).correlation)
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true",
                    help="enable Stage-2 cross-encoder rerank (RERANK_GTE); writes *_rerank.csv")
    ap.add_argument("--rerank-device", default=None, help="force reranker device (e.g. cpu if MPS NaNs)")
    ap.add_argument("--rerank-dtype", default=None, choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--rerank-temp", type=float, default=None, help="override sigmoid temperature")
    args = ap.parse_args()

    profiles = LinkedInAdapter().to_profiles(LINKEDIN_CSV)
    gold = load_fixture("linkedin", "_gold_hr_assistant")
    jd = load_jd(gold)

    model = SentenceTransformer("all-mpnet-base-v2")
    sim_spec = build_similarity_spec(similarity_model_config, base_model=model)
    emb_model = sim_spec.model if sim_spec else model
    cache = CACHE
    if sim_spec:
        slug = similarity_model_config["model_name"].replace("/", "_")
        cache = f".ai-recruiter/emb_linkedin_v2_{slug}.pkl"
    embed_profiles(
        profiles, emb_model, cache_path=cache,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else 32,
    )

    rerank_spec = None
    rerank_cache = None
    if args.rerank:
        cfg = dict(RERANK_GTE)
        if args.rerank_device:
            cfg["device"] = args.rerank_device
        if args.rerank_dtype:
            cfg["dtype"] = args.rerank_dtype
        if args.rerank_temp is not None:
            cfg["temperature"] = args.rerank_temp
        print(f"[rerank] Stage-2 ON: {cfg['model_name']} (top_k={rerank_top_k}, T={cfg['temperature']})")
        rerank_spec = build_rerank_model(cfg)
        rerank_cache = ".ai-recruiter/rerank_hr_assistant.pkl"

    df = score_pool(profiles, jd, model, sim_spec=sim_spec, rerank_spec=rerank_spec,
                    rerank_top_k=rerank_top_k, rerank_cache_path=rerank_cache)

    run_out = RUN_OUT.replace(".csv", "_rerank.csv") if args.rerank else RUN_OUT
    cmp_out = CMP_OUT.replace(".csv", "_rerank.csv") if args.rerank else CMP_OUT

    run_cols = [
        "pipeline_rank", "candidate_id", "job_title", "total_score", "title_score",
        "skill_score", "qualification_score", "seniority_score", "experience_score",
        "industry_score", "language_score", "location_score", "attrition_score", "experience_relevance_score", "education_relevance_score", "similarity_score", "data_completeness_level", "linkedin_url",
    ]
    if "rerank_score" in df.columns:
        run_cols.insert(run_cols.index("similarity_score") + 1, "rerank_score")
    os.makedirs("evals", exist_ok=True)
    df[run_cols].to_csv(run_out, index=False)

    # --- compare vs LLM-scored Scored_FullPool ---
    gold_df = pd.read_csv(GOLD_CSV, skiprows=2)
    gcols = ["linkedinUrl", "rank", "fit_0_10", "current_title", "reasoning", "matched_signals"]
    merged = df.merge(gold_df[gcols], left_on="linkedin_url", right_on="linkedinUrl", how="inner")
    merged = merged.rename(columns={"rank": "llm_rank", "fit_0_10": "llm_fit_0_10"})
    merged["rank_delta"] = merged["pipeline_rank"] - merged["llm_rank"]

    ranked_ids = df["candidate_id"].tolist()
    ndcg = {k: ndcg_at_k(ranked_ids, gold.relevance, k) for k in (5, 10, 20)}
    sp = spearman(merged["pipeline_rank"], merged["llm_rank"])
    kt = kendall(merged["pipeline_rank"].tolist(), merged["llm_rank"].tolist())

    llm_top = {k: set(merged.sort_values("llm_rank").head(k)["candidate_id"]) for k in (10, 20)}
    pipe_top = {k: set(df.head(k)["candidate_id"]) for k in (10, 20)}
    overlap = {k: len(llm_top[k] & pipe_top[k]) for k in (10, 20)}

    cmp_cols = [
        "candidate_id", "current_title", "pipeline_rank", "llm_rank", "rank_delta",
        "total_score", "llm_fit_0_10", "similarity_score", "skill_score",
        "seniority_score", "experience_score", "industry_score", "language_score",
        "location_score", "reasoning", "matched_signals",
    ]
    merged.sort_values("pipeline_rank")[cmp_cols].to_csv(cmp_out, index=False)

    # --- report ---
    active = ["title", "skill", "similarity"]
    if jd.qualifications and jd.qualifications.education:
        active.append("qualification")
    if jd.experience and jd.experience.level:
        active.append("seniority")
    if jd.experience and (jd.experience.years_total or jd.experience.years_relevant):
        active.append("experience")
    if jd.industry or (jd.experience and jd.experience.industry_experience):
        active.append("industry")
    if jd.language_proficiency and candidate_score_weights["language_score"] > 0:
        active.append("language")
    if jd.location and candidate_score_weights["location_score"] > 0:
        active.append("location")
    if candidate_score_weights["attrition_score"] > 0:
        active.append("attrition")
    if candidate_score_weights["experience_relevance_score"] > 0:
        active.append("experience_relevance")
    if candidate_score_weights["education_relevance_score"] > 0:
        active.append("education_relevance")
    print(f"JD: {jd.role} | pool: {len(df)} | active components: {', '.join(active)}")
    print(f"matched vs LLM pool: {len(merged)}/{len(df)}\n")

    print("=== pipeline top 15 (with LLM rank/fit) ===")
    top = merged.sort_values("pipeline_rank").head(15)
    print(f"{'#':>3} {'cand_id':<24} {'total':>6} {'sim':>5} {'skill':>5} "
          f"{'sen':>4} {'exp':>4} {'ind':>4} {'LLM#':>5} {'fit':>4}")
    for _, r in top.iterrows():
        print(f"{r['pipeline_rank']:>3} {r['candidate_id'][:24]:<24} {r['total_score']:6.3f} "
              f"{r['similarity_score']:5.2f} {r['skill_score']:5.2f} {r['seniority_score']:4.2f} "
              f"{r['experience_score']:4.2f} {r['industry_score']:4.2f} {int(r['llm_rank']):>5} {int(r['llm_fit_0_10']):>4}")

    print("\n=== agreement with LLM-scored Scored_FullPool ===")
    print(f"Spearman rho (rank corr): {sp:.3f}")
    print(f"Kendall  tau (rank corr): {kt:.3f}")
    print(f"top-10 overlap: {overlap[10]}/10   top-20 overlap: {overlap[20]}/20")

    print("\n=== silver Judge-grade anchor ===")
    print(f"NDCG@5/10/20: {ndcg[5]:.3f} / {ndcg[10]:.3f} / {ndcg[20]:.3f}")

    print("\n=== biggest disagreements (pipeline ranks FAR from LLM) ===")
    worst = merged.reindex(merged["rank_delta"].abs().sort_values(ascending=False).index).head(6)
    for _, r in worst.iterrows():
        direction = "pipeline HIGH / LLM low" if r["rank_delta"] < 0 else "pipeline LOW / LLM high"
        print(f"  {r['candidate_id'][:28]:<28} pipe#{int(r['pipeline_rank']):>3} vs LLM#{int(r['llm_rank']):>3} "
              f"(fit {int(r['llm_fit_0_10'])})  [{direction}]")

    print(f"\nwrote {run_out}\nwrote {cmp_out}")


if __name__ == "__main__":
    main()
