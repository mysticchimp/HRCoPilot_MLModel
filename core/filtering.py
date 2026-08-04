import numpy as np
import pandas as pd

from core.matching import fuzzy_match
from models.mappings import encode_batch_size


def _cosine_rows_to_vec(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity without retaining a full-batch torch tensor."""
    mat = np.asarray(matrix, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    mat_n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    v_n = v / (np.linalg.norm(v) + 1e-12)
    return (mat_n @ v_n).astype(np.float32)


def score_job_title(df: pd.DataFrame, job_title: str, model=None, mode: str = "fuzzy") -> pd.DataFrame:
    """Add a `title_score` column: fuzzy, semantic, or hybrid (max of both)."""
    titles = df["job_title"].fillna("").astype(str).tolist()
    fuzzy_scores = fuzzy_match(titles, [job_title])[:, 0] / 100.0

    if mode in ("semantic", "hybrid") and model is not None:
        # convert_to_numpy + small batch_size: avoid one padded torch encode of the whole pool.
        title_embeddings = model.encode(
            titles,
            convert_to_numpy=True,
            batch_size=encode_batch_size,
            show_progress_bar=False,
        )
        jd_embedding = model.encode(
            [job_title],
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False,
        )
        semantic_scores = _cosine_rows_to_vec(title_embeddings, jd_embedding[0])
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
