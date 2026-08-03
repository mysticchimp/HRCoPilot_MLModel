import numpy as np
import pandas as pd
from sentence_transformers import util

from core.matching import fuzzy_match


def score_job_title(df: pd.DataFrame, job_title: str, model=None, mode: str = "fuzzy") -> pd.DataFrame:
    """Add a `title_score` column: fuzzy, semantic, or hybrid (max of both)."""
    titles = df["job_title"].fillna("").astype(str).tolist()
    fuzzy_scores = fuzzy_match(titles, [job_title])[:, 0] / 100.0

    if mode in ("semantic", "hybrid") and model is not None:
        title_embeddings = model.encode(titles, convert_to_tensor=True)
        jd_embedding = model.encode([job_title], convert_to_tensor=True)
        semantic_scores = util.cos_sim(title_embeddings, jd_embedding).cpu().numpy().reshape(-1)
        scores = np.maximum(fuzzy_scores, semantic_scores) if mode == "hybrid" else semantic_scores
    else:
        scores = fuzzy_scores

    df = df.copy()
    df["title_score"] = scores
    return df


def filter_by_job_title(
    df: pd.DataFrame,
    job_title: str,
    threshold: float = 0.4,
    model=None,
    mode: str = "fuzzy",
    hard: bool = True,
) -> pd.DataFrame:
    """Score candidates by job-title match, optionally applying a hard cutoff.

    mode: 'fuzzy' (token_set_ratio), 'semantic' (embedding cos-sim), or 'hybrid'
          (max of the two — a lenient recall gate).
    hard: when True, drop candidates below `threshold`; when False, keep everyone
          and expose `title_score` as a soft signal only.
    """
    df = score_job_title(df, job_title, model=model, mode=mode)
    if hard:
        return df[df["title_score"] >= threshold]
    return df
