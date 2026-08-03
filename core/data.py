import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils.parsing import parse_list_str
from core.normalization import normalize_degree
from core.embedding import build_rerank_candidate_text
from core.positions import relevance_features, tenure_features
from core.completeness import data_completeness
from models.candidate import CandidateProfile

def process_candidate_data(
    csv_path: str,
    embedding_model: SentenceTransformer,
    encoding: str = "utf-8-sig",
    cache_path="./data/df_candidates.pkl",
    reload: bool = False
) -> pd.DataFrame:
    if os.path.exists(cache_path) and not reload:
        return pd.read_pickle(cache_path)
    
    df = pd.read_csv(csv_path, encoding=encoding)

    # this is done since `job_position_name` has a prefix in its column name
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    df["candidate_id"] = ["C" + str(i).zfill(3) for i in range(1, len(df) + 1)]
    df.set_index("candidate_id", drop=False)
    
    process_skills(df)
    process_qualifications(df)
    build_profile_embeddings(df, embedding_model)
    
    df.to_pickle(cache_path)
    return df

def process_skills(df: pd.DataFrame):
    df['skills'] = df['skills'].apply(parse_list_str)
    df['related_skils_in_job'] = df['related_skils_in_job'].apply(parse_list_str)
    df['certification_skills'] = df['certification_skills'].apply(parse_list_str)

    def combine_skills(row):
        skills = row['skills'] if isinstance(row['skills'], list) else []
        related_skills = [x for sublist in row['related_skils_in_job'] if isinstance(sublist, list) for x in sublist]
        certification_skills = [x for sublist in row['certification_skills'] if isinstance(sublist, list) for x in sublist]
        all_skills = skills + related_skills + certification_skills
        return list(set(all_skills))

    df['all_skills'] = df.apply(combine_skills, axis=1)

def normalize_degree_list(degrees: list[str], threshold=80):
    degree_texts = [degree_text.lower().strip() for degree_text in degrees if isinstance(degree_text, str)]
    return [normalize_degree(deg, threshold) for deg in degree_texts]

def process_qualifications(df: pd.DataFrame):
    df['degree_names'] = df['degree_names'].apply(parse_list_str)
    df['degree_names_norm'] = df['degree_names'].apply(normalize_degree_list)
    df['major_field_of_studies'] = df['major_field_of_studies'].apply(parse_list_str)
    df["qualifications"] = df.apply(
        lambda row: [f"{a} in {b}" if b != 'N/A' else a for a, b in zip(row["degree_names_norm"], row["major_field_of_studies"])],
        axis=1
    )

def build_candidate_profile_text(row):
    parts = []
    if row.get('career_objective'):
        parts.append(f"Career Objectives: {row['career_objective']}")
    if row.get('responsibilities'):
        parts.append(f"Responsibilities: {row['responsibilities']}")
    return ' '.join(parts)

def build_profile_embeddings(df: pd.DataFrame, model: SentenceTransformer):
    df['profile_text'] = df.apply(build_candidate_profile_text, axis=1)
    
    tqdm.pandas(desc="Building Profile Embeddings")
    candidate_profile_embeddings = model.encode(
        df['profile_text'].tolist(), 
        convert_to_tensor=True, 
        show_progress_bar=True
    )
    df['profile_embedding'] = [emb.cpu().numpy().tolist() for emb in candidate_profile_embeddings]


def _sector_text(p: CandidateProfile) -> str:
    """Free-text blob used for alias-based industry/sector matching."""
    parts = [p.job_title or "", p.summary or "", p.responsibilities or ""]
    if p.employers:
        parts.append(" ".join(p.employers))
    return " ".join(part for part in parts if part).strip()


def _language_names(p: CandidateProfile) -> list[str]:
    """Raw candidate language names (normalization happens in the language scorer)."""
    if not p.languages:
        return []
    return [lang.language for lang in p.languages if lang.language]


def profiles_to_dataframe(profiles: list[CandidateProfile]) -> pd.DataFrame:
    """Bridge canonical CandidateProfile objects into the DataFrame the scorers consume."""
    rows = []
    for p in profiles:
        loc = p.location
        tf = tenure_features(p.positions)
        rf = relevance_features(p.positions)
        dc = data_completeness(p.summary, p.responsibilities, len(p.skills or []), tf["n_dated_roles"])
        rows.append({
            "candidate_id": p.candidate_id,
            "job_title": p.job_title,
            "skills": p.skills,
            "degrees": p.degree_names,
            "fields": p.field_names,
            "certifications": p.certifications or [],
            "summary": p.summary,
            "responsibilities": p.responsibilities,
            "seniority": p.seniority,
            "years_experience": p.years_experience,
            "positions": [pos.model_dump() for pos in p.positions],
            "current_tenure_months": tf["current_tenure_months"],
            "completed_perm_tenures": tf["completed_perm_tenures"],
            "n_dated_roles": tf["n_dated_roles"],
            "relevant_months": rf["relevant_months"],
            "adjacent_months": rf["adjacent_months"],
            "total_dated_months": rf["total_dated_months"],
            "data_completeness_level": dc["level"],
            "data_completeness_missing": dc["missing"],
            "sector_text": _sector_text(p),
            "languages": _language_names(p),
            "location_city": loc.city if loc else None,
            "location_country": loc.country if loc else None,
            "location_country_code": loc.country_code if loc else None,
            "profile_embedding": p.profile_embedding,
            "rerank_text": build_rerank_candidate_text(p),
            "source": p.source,
            "linkedin_url": p.raw.get("linkedinUrl") if isinstance(p.raw, dict) else None,
        })
    return pd.DataFrame(rows)
