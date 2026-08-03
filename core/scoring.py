import pandas as pd
import torch
from statistics import median
from sentence_transformers import SentenceTransformer, util
from core.matching import weighted_fuzzy_skill_score, weighted_fuzzy_qualification_score
from core.embedding import build_jd_embedding_input, build_rerank_jd_text, log_truncation
from core.industry_normalization import industry_present, jd_industry_requirements
from core.reranking import RerankSpec, rerank_scores
from core.language_normalization import jd_language_requirements, normalize_candidate_languages
from core.location_normalization import (
    jd_location_requirements,
    normalize_city,
    normalize_country,
    normalize_country_code,
)
from core.skill_normalization import build_skill_semantic_index
from models.data_models import Skill, JobRoleSchema
from models.mappings import (
    candidate_score_weights,
    seniority_rank_map,
    seniority_under_penalty,
    seniority_over_penalty,
    experience_under_penalty_per_year,
    experience_over_penalty_per_year,
    location_city_match,
    location_country_match,
    location_mismatch,
    attribute_weight_by_importance,
    skill_semantic_threshold,
    attrition_tenure_bands,
    attrition_short_stint_months,
    attrition_chronic_min_short_roles,
    attrition_chronic_hop_score,
    attrition_short_stint_score,
    attrition_neutral,
    attrition_early_career_years,
    attrition_early_career_max_roles,
    attrition_early_career_floor,
    experience_relevance_adjacent_credit,
    experience_relevance_neutral,
    education_relevant_fields,
    education_business_fields,
    education_hr_certs,
    education_relevant_score,
    education_business_score,
    education_neutral,
)

def calculate_skill_score(
    df: pd.DataFrame,
    jd: JobRoleSchema,
    filter: bool = False,
    threshold: float = 0.25,
    match_threshold: float = 70,
    model: SentenceTransformer | None = None,
    skill_mode: str = 'fuzzy',
    semantic_threshold: float | None = None,
) -> pd.DataFrame:
    """Score candidate skills against the JD's required skills + technologies.

    skill_mode: 'fuzzy' (char-level rapidfuzz + aliases; the original behavior),
    'semantic' (embedding cosine only), or 'hybrid' (max of the two, mirroring the
    title matcher). Semantic modes need `model`; they embed each unique skill
    string in the pool once, so per-candidate matching stays a cheap vector lookup.
    """
    jd_skills = jd.skills.copy()
    if jd.technologies:
        for t in jd.technologies:
            jd_skills.append(Skill(skill=t.technology, priority=t.priority, proficiency_level=None))

    semantic_index = None
    if skill_mode in ('semantic', 'hybrid') and model is not None:
        pool_skills = [s.skill for s in jd_skills]
        for cand_skills in df['skills']:
            if isinstance(cand_skills, (list, tuple)):
                pool_skills.extend(s for s in cand_skills if isinstance(s, str))
        semantic_index = build_skill_semantic_index(pool_skills, model)
    sem_threshold = semantic_threshold if semantic_threshold is not None else skill_semantic_threshold

    skill_results = df.apply(
        lambda x: weighted_fuzzy_skill_score(
            x['candidate_id'], jd_skills, x['skills'],
            score_threshold=match_threshold,
            semantic_index=semantic_index,
            semantic_threshold=sem_threshold,
            include_fuzzy=(skill_mode != 'semantic'),
        ),
        axis=1
    )

    skill_results_df = pd.DataFrame(skill_results.tolist())
    df['skill_score'] = skill_results_df['score']
    df['matched_skills'] = skill_results_df['matched_skills']

    if filter:
        df = df[df['skill_score'] >= threshold]
    return df

def calculate_qualification_score(df: pd.DataFrame, jd: JobRoleSchema, filter: bool = False, threshold: float = 0.2) -> pd.DataFrame:
    if jd.qualifications and jd.qualifications.education:
        qualification_results = df.apply(
            lambda x: weighted_fuzzy_qualification_score(
                x['candidate_id'], 
                jd.qualifications.education,# type: ignore
                {"degrees": x['degrees'], "fields": x['fields']} 
            ), axis=1)
        qualification_results_df = pd.DataFrame(qualification_results.tolist())
        df['qualification_score'] = qualification_results_df['score']
        df['matched_qualifications'] = qualification_results_df['matched_qualifications']
        if filter:
            df = df[df['qualification_score'] >= threshold]
    else:
        df['qualification_score'] = 0.0
        df['matched_qualifications'] = None
    return df

def calculate_seniority_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score candidate seniority against the JD's required level.

    Candidate and JD share the ordinal scale entry<mid<senior<executive<c_level.
    A candidate on the required level scores 1.0; the score decays with the level
    gap, penalizing under-qualification more than over-qualification. Candidates
    with an unknown/missing level get a neutral 0.5. When the JD does not specify a
    level every candidate gets 0.5 and calculate_total_score drops the component
    from the weighted blend (so it never penalizes JDs that omit seniority).
    """
    required = jd.experience.level if jd.experience else None
    jd_rank = seniority_rank_map.get(required) if required else None
    if jd_rank is None:
        df['seniority_score'] = 0.5
        return df

    def _score(level):
        cand_rank = seniority_rank_map.get(level)
        if cand_rank is None:
            return 0.5  # candidate missing/unknown seniority -> neutral
        diff = cand_rank - jd_rank
        if diff == 0:
            return 1.0
        penalty = seniority_over_penalty if diff > 0 else seniority_under_penalty
        return max(0.0, 1.0 - penalty * abs(diff))

    df['seniority_score'] = df['seniority'].apply(_score)
    return df

def calculate_experience_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score candidate years of experience against the JD's required range.

    Uses experience.years_total, falling back to years_relevant. A candidate whose
    total years land inside [min, max] scores 1.0; below the minimum the score
    decays by experience_under_penalty_per_year, above the maximum by the gentler
    experience_over_penalty_per_year (over-qualification is a milder signal than a
    shortfall). Candidates with unknown years get a neutral 0.5. When the JD gives
    no usable range every candidate gets 0.5 and calculate_total_score drops the
    component from the weighted blend.
    """
    req = (jd.experience.years_total or jd.experience.years_relevant) if jd.experience else None
    lo = req.min if req else None
    hi = req.max if req else None
    if lo is None and hi is None:
        df['experience_score'] = 0.5
        return df

    def _score(years):
        if years is None or pd.isna(years):
            return 0.5  # candidate missing years -> neutral
        if lo is not None and years < lo:
            return max(0.0, 1.0 - experience_under_penalty_per_year * (lo - years))
        if hi is not None and years > hi:
            return max(0.0, 1.0 - experience_over_penalty_per_year * (years - hi))
        return 1.0

    df['experience_score'] = df['years_experience'].apply(_score)
    return df

def calculate_industry_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score candidate sector fit against the JD's target industries.

    Alias-based: each JD industry requirement (priority-weighted) is checked for
    presence in the candidate's derived `sector_text` (employers + title + role
    descriptions). Score = matched priority-weight / total priority-weight.
    Candidates with no sector text get a neutral 0.5; when the JD names no
    industries the component is inert (0.5) and calculate_total_score drops it.
    """
    requirements = jd_industry_requirements(jd)
    if not requirements:
        df['industry_score'] = 0.5
        return df
    total_weight = sum(attribute_weight_by_importance[p] for _, p in requirements) or 1.0

    def _score(text):
        if not isinstance(text, str) or not text.strip():
            return 0.5  # no sector evidence -> neutral
        matched = sum(attribute_weight_by_importance[p] for industry, p in requirements
                      if industry_present(industry, text))
        return matched / total_weight

    df['industry_score'] = df['sector_text'].apply(_score)
    return df

def calculate_language_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score candidate language fit against the JD's required languages.

    Presence-based and priority-weighted (mirrors calculate_industry_score): each
    JD language (from language_proficiency) is checked for presence in the
    candidate's normalized language set; score = matched priority-weight / total
    priority-weight. The required proficiency level is NOT gated (candidate
    proficiency is noisy free-text — see core/language_normalization). Candidates
    with no listed languages get a neutral 0.5; when the JD names no languages the
    component is inert (0.5) and calculate_total_score drops it from the blend.
    """
    requirements = jd_language_requirements(jd)
    if not requirements:
        df['language_score'] = 0.5
        return df
    total_weight = sum(attribute_weight_by_importance[p] for _, p in requirements) or 1.0

    def _score(names):
        cand = normalize_candidate_languages(names)
        if not cand:
            return 0.5  # no language evidence -> neutral
        matched = sum(attribute_weight_by_importance[p] for language, p in requirements
                      if language in cand)
        return matched / total_weight

    df['language_score'] = df['languages'].apply(_score)
    return df

def calculate_location_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score candidate location fit against the JD's target cities + countries.

    Country is the primary gate; the city refines within it (a natural
    country > city hierarchy). A matching city is the strongest, most specific
    signal (full credit). Inside a matching country an OMITTED city is not
    penalized — the primary requirement is met and the city is simply unknown, so
    it gets the benefit of the doubt (full credit, equal to a city match) — while a
    confirmed DIFFERENT city earns partial credit (location_country_match). A wrong
    country is a miss. When the JD names only a country, any in-country candidate is
    full credit. Candidates with no location data at all get a neutral 0.5; when the
    JD names no location the component is inert (0.5) and calculate_total_score
    drops it from the blend.
    """
    jd_cities, jd_countries = jd_location_requirements(jd)
    if not jd_cities and not jd_countries:
        df['location_score'] = 0.5
        return df

    def _score(row):
        city = normalize_city(row['location_city'])
        cand_country = (normalize_country(row['location_country'])
                        or normalize_country_code(row['location_country_code']))
        if not city and not cand_country:
            return 0.5  # no location evidence at all -> neutral (unknown)

        # a matching city is the strongest, most specific signal
        if jd_cities and city and city in jd_cities:
            return location_city_match

        # country is the primary gate; the city refines within it
        if jd_countries and cand_country and cand_country in jd_countries:
            # right country: an omitted city is NOT penalized (benefit of the doubt,
            # the primary requirement is met); a confirmed DIFFERENT city is.
            if not jd_cities or not city:
                return location_city_match       # city unspecified / not required
            return location_country_match        # confirmed different city in-country
        if jd_countries and cand_country:
            return location_mismatch             # confirmed wrong country
        # country can't be confirmed (candidate country missing, or city-only JD):
        # a confirmed non-matching city is a weak miss; otherwise nothing to assess
        return location_mismatch if city else 0.5

    df['location_score'] = df.apply(_score, axis=1)
    return df

def calculate_attrition_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score job stability (flight-risk) from per-role tenure (C5 P2).

    Product-motivated and JD-independent (mirrors location_score): job stability
    always matters, so this is active whenever positions were parsed. The base is the
    MEDIAN COMPLETED-permanent tenure mapped through attrition_tenure_bands
    (contractors excluded upstream; the current role is excluded from the median
    because it is right-censored — we only know it is >= its tenure-so-far). A median
    under the lowest band is a chronic hopper (>= attrition_chronic_min_short_roles
    short permanent roles) or a one-off short stint. The current role then acts as a
    LIFT-ONLY signal: a long current tenure is a lower bound on loyalty and can only
    RAISE the score, while a short current stint is inconclusive (not yet a departure)
    and never lowers it. The entry-level pool is guarded by an early-career floor
    (short histories are normal for juniors), and too little history to assess returns
    a neutral 0.5. All thresholds live in models/mappings.py — the ablation tunes the
    WEIGHT, not this shape.
    """
    def _score(row):
        n_dated = row.get('n_dated_roles') or 0
        if n_dated < 2:
            return attrition_neutral  # can't assess
        perm = row.get('completed_perm_tenures') or []
        if not perm:
            return attrition_neutral  # nothing completed-permanent to judge
        med = median(perm)
        for months, score in attrition_tenure_bands:
            if med >= months:
                base = score
                break
        else:  # median below the lowest band -> chronic hop vs one-off short stint
            n_short = sum(1 for t in perm if t < attrition_short_stint_months)
            base = (attrition_chronic_hop_score if n_short >= attrition_chronic_min_short_roles
                    else attrition_short_stint_score)
        # the current role is right-censored: a LONG current tenure is a lower bound on
        # loyalty and can only RAISE the score (map it through the same bands via max);
        # a short current stint is inconclusive (not yet a departure) and never lowers it.
        current = row.get('current_tenure_months')
        if current is not None and not pd.isna(current):
            for months, score in attrition_tenure_bands:
                if current >= months:
                    base = max(base, score)
                    break
        years = row.get('years_experience')
        if (years is not None and not pd.isna(years)
                and years < attrition_early_career_years and n_dated <= attrition_early_career_max_roles):
            base = max(base, attrition_early_career_floor)  # don't punish juniors
        return base

    df['attrition_score'] = df.apply(_score, axis=1)
    return df

def calculate_experience_relevance_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Score how much of a candidate's tenure is HR-relevant vs adjacent (C5 P3).

    Distinguishes *relevant* HR years from *total* years: each role's title is
    classified relevant / adjacent / unrelated (core.positions.classify_role) and
    the score is the tenure-weighted ratio
    (relevant + adjacent_credit*adjacent) / total_dated_months, clamped to [0, 1].
    Candidates with no dated/titled roles get a neutral 0.5. Product-motivated and
    JD-independent; the WEIGHT is tuned by the ablation (and checked for redundancy
    against similarity_score and experience_score).
    """
    def _score(row):
        total = row.get('total_dated_months')
        if total is None or pd.isna(total) or total <= 0:
            return experience_relevance_neutral
        rel = row.get('relevant_months')
        adj = row.get('adjacent_months')
        rel = 0 if rel is None or pd.isna(rel) else rel
        adj = 0 if adj is None or pd.isna(adj) else adj
        ratio = (rel + experience_relevance_adjacent_credit * adj) / total
        return max(0.0, min(1.0, ratio))

    df['experience_relevance_score'] = df.apply(_score, axis=1)
    return df

def calculate_education_relevance_score(df: pd.DataFrame, jd: JobRoleSchema) -> pd.DataFrame:
    """Soft HR/business-degree + HR-cert tie-breaker in [0.5, 1.0] (C5 P4).

    A directly-HR degree (human resources, psychology, law) or an HR cert
    (CIPD/CHRP/SHRM/aPHR) scores 1.0; a business-adjacent degree 0.75; anything
    else — including no education data — the 0.5 neutral floor. The score is the MAX
    credential tier, so it can only REWARD, never drop a candidate below neutral:
    it is a soft bonus, never a gate (the JD requires no degree). Distinct from the
    gate-style qualification_score. The WEIGHT is tuned by the ablation.
    """
    def _tier(text: str) -> float:
        t = text.lower()
        if any(k in t for k in education_relevant_fields):
            return education_relevant_score
        if any(k in t for k in education_business_fields):
            return education_business_score
        return education_neutral

    def _score(row):
        best = None
        for field in (row.get('fields') or []):
            if isinstance(field, str) and field and field != 'N/A':
                tier = _tier(field)
                best = tier if best is None else max(best, tier)
        for cert in (row.get('certifications') or []):
            if isinstance(cert, str) and any(k in cert.lower() for k in education_hr_certs):
                best = education_relevant_score if best is None else max(best, education_relevant_score)
        return best if best is not None else education_neutral

    df['education_relevance_score'] = df.apply(_score, axis=1)
    return df

def calculate_similarity_score(
    df: pd.DataFrame,
    jd: JobRoleSchema,
    model: SentenceTransformer,
    query_instruction: str | None = None,
) -> pd.DataFrame:
    jd_text = build_jd_embedding_input(jd)
    if query_instruction:
        jd_text = f"{query_instruction}{jd_text}"
    log_truncation(model, [jd_text], "JD")
    # .float(): fp16/bf16 similarity models emit half-precision JD vectors, but the
    # cached candidate embeddings are stored as float lists (float32) — cast so the
    # cos_sim matmul sees matching dtypes.
    jd_embedding = model.encode(jd_text, convert_to_tensor=True).cpu().float()
    candidate_embeddings = torch.stack([torch.tensor(vec, dtype=torch.float32) for vec in df['profile_embedding']])
    similarities = util.cos_sim(candidate_embeddings, jd_embedding).squeeze().cpu().numpy()
    df['similarity_score'] = similarities
    return df

def calculate_total_score(
    df: pd.DataFrame, jd: JobRoleSchema, weights: dict | None = None, normalize: bool = False,
    component_cols: dict | None = None,
) -> pd.DataFrame:
    # include only the components active for this JD, then renormalize their
    # weights so they always sum to 1 (keeps total_score comparable across JDs).
    weights = weights or candidate_score_weights
    active_components = ['title_score', 'skill_score', 'similarity_score']
    if jd.qualifications and jd.qualifications.education:
        active_components.append('qualification_score')
    exp = jd.experience
    if exp and exp.level and 'seniority_score' in df.columns:
        active_components.append('seniority_score')
    if exp and 'experience_score' in df.columns:
        years_req = exp.years_total or exp.years_relevant
        if years_req and (years_req.min is not None or years_req.max is not None):
            active_components.append('experience_score')
    if 'industry_score' in df.columns and (jd.industry or (exp and exp.industry_experience)):
        active_components.append('industry_score')
    if 'language_score' in df.columns and jd.language_proficiency:
        active_components.append('language_score')
    if 'location_score' in df.columns and jd.location and (jd.location.cities or jd.location.countries):
        active_components.append('location_score')
    # product-motivated, JD-independent (like a candidate-side signal): active
    # whenever positions were parsed. Per-candidate insufficient history -> 0.5.
    if 'attrition_score' in df.columns:
        active_components.append('attrition_score')
    if 'experience_relevance_score' in df.columns:
        active_components.append('experience_relevance_score')
    if 'education_relevance_score' in df.columns:
        active_components.append('education_relevance_score')

    weight_sum = sum(weights.get(c, 0.0) for c in active_components) or 1.0

    def component(col: str):
        # Optional min-max normalization across the candidate pool so components
        # on different natural scales (raw cosine ~0.3-0.6 vs 0-1 fuzzy) contribute
        # variation proportional to their weight, not their natural range.
        # component_cols remaps a component to a different column (used by apply_rerank
        # to feed rerank_score through similarity_score's weight slot in the Head).
        series = df[(component_cols or {}).get(col, col)]
        if normalize:
            low, high = series.min(), series.max()
            return (series - low) / (high - low) if high > low else series * 0.0
        return series

    df['total_score'] = sum(
        (weights.get(c, 0.0) / weight_sum) * component(c) for c in active_components
    )
    return df


def apply_rerank(
    df: pd.DataFrame,
    jd: JobRoleSchema,
    rerank_spec: "RerankSpec | None" = None,
    top_k: int = 50,
    weights: dict | None = None,
    normalize: bool = False,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Produce the FINAL best-first ranking, applying the Stage-2 cross-encoder rerank.

    df must already carry a Stage-1 `total_score` (from calculate_total_score). This
    function ALWAYS returns df sorted best-first, so every entry point can route its
    ranking through it: when rerank_spec is None it is just the Stage-1 sort (no-op).

    With a rerank_spec, the top-`top_k` Head is re-scored by the cross-encoder and the
    resulting `rerank_score` (in [0,1]) REPLACES similarity_score in the fusion slot for
    the Head only; the Head is re-sorted by the recomputed total_score and the tail
    (rank > top_k) keeps its Stage-1 order beneath it (FROZEN MEMBERSHIP). The returned
    frame is already in final order — callers must NOT re-sort by total_score (the Head's
    v2 score and the tail's v1 score are on different scales). Needs a `rerank_text` column.
    """
    if df.empty:
        return df
    df = df.sort_values('total_score', ascending=False).reset_index(drop=True)
    if rerank_spec is None:
        return df

    k = min(top_k, len(df))
    head = df.iloc[:k].copy()
    tail = df.iloc[k:].copy()

    jd_text = build_rerank_jd_text(jd)
    cand_texts = head['rerank_text'].fillna('').tolist() if 'rerank_text' in head.columns else [''] * len(head)
    head['rerank_score'] = rerank_scores(rerank_spec, jd_text, cand_texts, cache_path=cache_path)
    # B1 full-replace: the CE score takes similarity_score's 0.45 slot within the Head;
    # every other component is reused unchanged from Stage 1.
    head = calculate_total_score(
        head, jd, weights=weights, normalize=normalize,
        component_cols={'similarity_score': 'rerank_score'},
    )
    head = head.sort_values('total_score', ascending=False)
    if 'rerank_score' not in tail.columns:
        tail['rerank_score'] = float('nan')
    return pd.concat([head, tail], ignore_index=True)
