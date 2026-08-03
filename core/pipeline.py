import logging

import pandas as pd
from sentence_transformers import SentenceTransformer

from core.adapters.base import CandidateAdapter
from core.adapters.resume_adapter import ResumeAdapter
from core.data import profiles_to_dataframe
from core.embedding import build_similarity_spec, embed_profiles
from core.filtering import filter_by_job_title
from core.jd_extraction import load_sample_jd, process_jd
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
from models.candidate import CandidateProfile
from models.data_models import JobRoleSchema
from models.mappings import rerank_model_config as CHAMPION_RERANK_CONFIG
from models.mappings import rerank_top_k as CHAMPION_RERANK_TOP_K
from models.mappings import similarity_model_config as CHAMPION_SIM_CONFIG


def run_pipeline(
    jd_text: str | None = None,
    resume_csv_path: str = './data/resume_data.csv',
    jd_file_path: str = './jd/sample_jd_01.txt',
    embedding_model_name: str = 'all-mpnet-base-v2',
    adapter: CandidateAdapter | None = None,
    source=None,
    profiles: list[CandidateProfile] | None = None,
    title_score_threshold: float = 0.4,
    title_mode: str = 'hybrid',
    title_hard: bool = False,
    filter_by_skills: bool = False,
    skill_score_threshold: float = 0.25,
    skill_match_threshold: float = 70,
    skill_mode: str = 'hybrid',
    skill_semantic_threshold: float | None = None,
    filter_by_qualifications: bool = False,
    qualification_score_threshold: float = 0.2,
    top_n: int = 10,
    embedding_cache_path: str | None = None,
    processed_jd: JobRoleSchema | None = None,
    weights: dict | None = None,
    normalize_components: bool = False,
    similarity_model_config: dict | None = CHAMPION_SIM_CONFIG,
    rerank_model_config: dict | None = CHAMPION_RERANK_CONFIG,
    rerank_top_k: int = CHAMPION_RERANK_TOP_K,
) -> pd.DataFrame:
    """Run the full candidate matching pipeline over any adapter's candidates.

    By default uses the ResumeAdapter on `resume_csv_path`; pass a different
    `adapter` + `source` (or a pre-built `profiles` list) to score another dataset
    through the same pipeline.
    """
    logging.info("Starting candidate matching pipeline...")

    logging.info(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    sim_spec = build_similarity_spec(similarity_model_config, base_model=model)
    rerank_spec = build_rerank_model(rerank_model_config)
    emb_model = sim_spec.model if sim_spec else model

    logging.info("Loading candidates via adapter...")
    if profiles is None:
        if adapter is None:
            adapter = ResumeAdapter()
            source = source if source is not None else resume_csv_path
        profiles = adapter.to_profiles(source)

    logging.info(f"Embedding {len(profiles)} candidate profiles...")
    embed_profiles(
        profiles, emb_model, cache_path=embedding_cache_path,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else 32,
    )
    df_candidates = profiles_to_dataframe(profiles)

    logging.info("Loading and processing job description...")
    if processed_jd is None:
        if jd_text is None:
            jd_text = load_sample_jd(jd_file_path)
        processed_jd = process_jd(jd_text)

    logging.info(f"Scoring job title (mode={title_mode}, hard={title_hard})...")
    title_model = model if title_mode != 'fuzzy' else None
    df_filtered = filter_by_job_title(
        df_candidates, processed_jd.role, title_score_threshold,
        model=title_model, mode=title_mode, hard=title_hard,
    )

    logging.info("Scoring skills...")
    df_filtered = calculate_skill_score(
        df_filtered,
        processed_jd,
        filter_by_skills,
        skill_score_threshold,
        match_threshold=skill_match_threshold,
        model=model,
        skill_mode=skill_mode,
        semantic_threshold=skill_semantic_threshold,
    )

    logging.info("Scoring qualifications...")
    df_filtered = calculate_qualification_score(df_filtered, processed_jd, filter_by_qualifications, qualification_score_threshold)

    logging.info("Scoring seniority...")
    df_filtered = calculate_seniority_score(df_filtered, processed_jd)

    logging.info("Scoring experience...")
    df_filtered = calculate_experience_score(df_filtered, processed_jd)

    logging.info("Scoring industry...")
    df_filtered = calculate_industry_score(df_filtered, processed_jd)

    logging.info("Scoring languages...")
    df_filtered = calculate_language_score(df_filtered, processed_jd)

    logging.info("Scoring location...")
    df_filtered = calculate_location_score(df_filtered, processed_jd)

    logging.info("Scoring tenure/attrition...")
    df_filtered = calculate_attrition_score(df_filtered, processed_jd)

    logging.info("Scoring experience relevance...")
    df_filtered = calculate_experience_relevance_score(df_filtered, processed_jd)

    logging.info("Scoring education relevance...")
    df_filtered = calculate_education_relevance_score(df_filtered, processed_jd)

    logging.info("Scoring by similarity...")
    df_filtered = calculate_similarity_score(
        df_filtered, processed_jd,
        sim_spec.model if sim_spec else model,
        query_instruction=sim_spec.query_instruction if sim_spec else None,
    )

    logging.info("Calculating final score...")
    df_scored = calculate_total_score(df_filtered, processed_jd, weights=weights, normalize=normalize_components)

    logging.info(f"Reranking (Stage 2) and returning top {top_n} candidates.")
    df_ranked = apply_rerank(
        df_scored, processed_jd, rerank_spec, top_k=rerank_top_k,
        weights=weights, normalize=normalize_components,
    )
    cols_to_display = [
        'candidate_id', 'job_title', 'total_score', 'title_score', 'skill_score', 'matched_skills',
        'qualification_score', 'matched_qualifications', 'seniority_score', 'experience_score', 'industry_score',
        'language_score', 'location_score', 'attrition_score', 'experience_relevance_score', 'education_relevance_score', 'similarity_score',
    ]
    if 'rerank_score' in df_ranked.columns:
        cols_to_display.append('rerank_score')
    top_candidates = df_ranked.head(top_n)

    logging.info("Pipeline finished.")
    return top_candidates[cols_to_display].reset_index(drop=True)
