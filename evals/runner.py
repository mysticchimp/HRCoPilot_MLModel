from dataclasses import dataclass

from core.data import profiles_to_dataframe
from core.embedding import SimilaritySpec, embed_profiles
from core.filtering import filter_by_job_title
from core.reranking import RerankSpec
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
from evals.metrics import hit_at_k, mean, ndcg_at_k, rank_of, reciprocal_rank


@dataclass
class PipelineConfig:
    """All tunable knobs the harness sweeps. Defaults = current pipeline behavior."""
    title_mode: str = "hybrid"          # fuzzy | semantic | hybrid
    title_threshold: float = 0.4
    title_hard: bool = False
    filter_by_skills: bool = False
    skill_threshold: float = 0.25        # optional candidate filter threshold
    skill_match_floor: float = 70        # fuzzy similarity floor; above it credit is graded
    skill_mode: str = "hybrid"          # fuzzy | semantic | hybrid
    skill_semantic_threshold: float | None = None  # None -> models.mappings.skill_semantic_threshold
    filter_by_qualifications: bool = False
    qual_threshold: float = 0.2
    weights: dict | None = None         # None -> models.mappings.candidate_score_weights
    normalize_components: bool = False   # min-max normalize each component before weighting
    rerank_top_k: int = 50               # Stage-2 Head size (only used when a rerank_spec is passed)


def rank_candidates(df_full, parsed_jd, model, config: PipelineConfig,
                    sim_spec: "SimilaritySpec | None" = None,
                    rerank_spec: "RerankSpec | None" = None,
                    rerank_cache_path: str | None = None) -> list[str]:
    """Run the scoring pipeline for one JD and return candidate_ids best-first."""
    df = df_full.copy()
    title_model = model if config.title_mode != "fuzzy" else None
    df = filter_by_job_title(
        df, parsed_jd.role, config.title_threshold,
        model=title_model, mode=config.title_mode, hard=config.title_hard,
    )
    if len(df) == 0:
        return []
    df = calculate_skill_score(
        df,
        parsed_jd,
        config.filter_by_skills,
        config.skill_threshold,
        match_threshold=config.skill_match_floor,
        model=model,
        skill_mode=config.skill_mode,
        semantic_threshold=config.skill_semantic_threshold,
    )
    df = calculate_qualification_score(df, parsed_jd, config.filter_by_qualifications, config.qual_threshold)
    df = calculate_seniority_score(df, parsed_jd)
    df = calculate_experience_score(df, parsed_jd)
    df = calculate_industry_score(df, parsed_jd)
    df = calculate_language_score(df, parsed_jd)
    df = calculate_location_score(df, parsed_jd)
    df = calculate_attrition_score(df, parsed_jd)
    df = calculate_experience_relevance_score(df, parsed_jd)
    df = calculate_education_relevance_score(df, parsed_jd)
    sim_model = sim_spec.model if sim_spec else model
    sim_query_instruction = sim_spec.query_instruction if sim_spec else None
    df = calculate_similarity_score(df, parsed_jd, sim_model, query_instruction=sim_query_instruction)
    df = calculate_total_score(df, parsed_jd, weights=config.weights, normalize=config.normalize_components)
    df = apply_rerank(df, parsed_jd, rerank_spec, top_k=config.rerank_top_k,
                      weights=config.weights, normalize=config.normalize_components,
                      cache_path=rerank_cache_path)
    return df["candidate_id"].tolist()


def evaluate_cases(cases, profiles, model, config: PipelineConfig | None = None,
                   embedding_cache_path: str | None = None, ks=(1, 3, 5, 10),
                   sim_spec: "SimilaritySpec | None" = None,
                   rerank_spec: "RerankSpec | None" = None,
                   rerank_cache_path: str | None = None) -> list[dict]:
    """Embed the pool once, then rank + score each case.

    `sim_spec` isolates the `similarity_score` embedding model (Option B): profile
    embeddings use `sim_spec.model` while title/skill keep the base `model`. When None,
    the base `model` embeds everything (current behavior).
    """
    config = config or PipelineConfig()
    emb_model = sim_spec.model if sim_spec else model
    embed_profiles(
        profiles, emb_model, cache_path=embedding_cache_path,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else 32,
    )
    df_full = profiles_to_dataframe(profiles)

    per_case = []
    for case in cases:
        ranked = rank_candidates(df_full, case.parsed_jd, model, config, sim_spec=sim_spec,
                                 rerank_spec=rerank_spec, rerank_cache_path=rerank_cache_path)
        relevant = {cid for cid, grade in case.relevance.items() if grade > 0}
        row = {"case_id": case.case_id, "source": case.source, "n_ranked": len(ranked)}
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(ranked, relevant, k)
        row["rr"] = reciprocal_rank(ranked, relevant)
        if case.seed_id:
            row["seed_rank"] = rank_of(ranked, case.seed_id)
        if case.source == "llm_scored":
            for k in (5, 10):
                row[f"ndcg@{k}"] = ndcg_at_k(ranked, case.relevance, k)
        per_case.append(row)
    return per_case


def aggregate(per_case, ks=(1, 3, 5, 10)) -> dict:
    """Aggregate per-case rows: hit@k + MRR over reverse-match, NDCG over gold."""
    reverse = [r for r in per_case if r["source"] == "reverse_match"]
    gold = [r for r in per_case if r["source"] == "llm_scored"]

    summary: dict = {"n_reverse_match": len(reverse), "n_gold": len(gold)}
    if reverse:
        for k in ks:
            summary[f"hit@{k}"] = round(mean([r[f"hit@{k}"] for r in reverse]), 4)
        summary["mrr"] = round(mean([r["rr"] for r in reverse]), 4)
        summary["seed_found_rate"] = round(mean([1.0 if r.get("seed_rank") else 0.0 for r in reverse]), 4)
    if gold:
        for k in (5, 10):
            summary[f"ndcg@{k}"] = round(mean([r.get(f"ndcg@{k}", 0.0) for r in gold]), 4)
    return summary
