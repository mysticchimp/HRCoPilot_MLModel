"""Thin FastAPI wrapper around the scoring pipeline.

POST /score
  body: { jd_text, candidates: [{ candidate_id, raw_profile }] }
  → ApifyJsonAdapter → run_pipeline → swipe-card JSON (tagged with caller candidate_id)

Embedding models (all-mpnet + optional Qwen similarity) are loaded once at startup
and reused. Uvicorn 0.52+ runs lifespan startup *before* binding the listen port, so
``warm_scoring_models()`` completes before any request can be accepted. We still gate
``/score`` on an explicit ready flag as belt-and-suspenders.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.adapters.apify_json_adapter import ApifyJsonAdapter
from core.data import profiles_to_dataframe
from core.jd_extraction import process_jd
from core.model_cache import warm_scoring_models
from core.pipeline import run_pipeline
from core.swipe import build_card
from models.candidate import CandidateProfile

logger = logging.getLogger(__name__)

# Optional offline JD cache (e.g. jd/parsed/hr_assistant_prime_ac.json) for local
# sanity without a live Anthropic call. Leave unset in production.
_JD_CACHE_PATH = os.environ.get("JD_CACHE_PATH") or None

# Populated by lifespan; reused on every /score.
_embedding_model = None
_sim_spec = None
_rerank_spec = None
# Explicit ready gate: False until warm_scoring_models() returns and globals are set.
# Uvicorn binds the port only after lifespan yields, but /score also checks this.
_models_ready = False
_startup_rss_mb: float | None = None


def _require_ready() -> None:
    if not _models_ready or _embedding_model is None:
        raise HTTPException(
            status_code=503,
            detail="Scoring models are still warming up — retry shortly.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embedding_model, _sim_spec, _rerank_spec, _models_ready, _startup_rss_mb
    from core.model_cache import resolve_base_embedding_dtype, resolve_similarity_model_config

    _models_ready = False
    raw_sim_env = os.environ.get("SIMILARITY_MODEL")
    raw_dtype_env = os.environ.get("BASE_EMBEDDING_DTYPE")
    resolved_dtype = resolve_base_embedding_dtype()
    resolved_sim_cfg = resolve_similarity_model_config()
    sim_mode = "mpnet-only" if resolved_sim_cfg is None else "qwen"

    # print() so Render captures this even if app logger level is unset.
    print(
        f"Config: similarity_model={sim_mode}, embedding_dtype={resolved_dtype}, "
        f"SIMILARITY_MODEL_env={raw_sim_env!r}, BASE_EMBEDDING_DTYPE_env={raw_dtype_env!r}",
        flush=True,
    )
    logger.info(
        "Config: similarity_model=%s, embedding_dtype=%s "
        "(env SIMILARITY_MODEL=%r BASE_EMBEDDING_DTYPE=%r)",
        sim_mode,
        resolved_dtype,
        raw_sim_env,
        raw_dtype_env,
    )
    print(
        "Warming scoring models (blocking; uvicorn has not bound the listen port yet)...",
        flush=True,
    )
    # Sync call on purpose: must finish before yield. Uvicorn awaits lifespan.startup()
    # and only then create_server() — so port-open cannot precede this return.
    _embedding_model, _sim_spec, _rerank_spec = warm_scoring_models()
    try:
        from core.mem_trace import rss_mb

        _startup_rss_mb = rss_mb()
        sim_key = None if _sim_spec is None else _sim_spec.model_key
        print(
            f"Model warm-up complete; process_rss_mb={_startup_rss_mb:.1f} "
            f"(post-load + encode-prime baseline; sim={sim_key})",
            flush=True,
        )
        logger.info(
            "Model warm-up complete; process_rss_mb=%.1f "
            "(post-load + encode-prime baseline; sim=%s)",
            _startup_rss_mb,
            sim_key,
        )
    except Exception:  # noqa: BLE001
        _startup_rss_mb = None
        logger.info("Model warm-up complete")
        print("Model warm-up complete (rss unavailable)", flush=True)

    _models_ready = True
    print(
        "READY: models loaded; lifespan yielding — uvicorn will bind port next "
        f"(models_ready=True process_rss_mb={_startup_rss_mb})",
        flush=True,
    )
    yield
    _models_ready = False


app = FastAPI(title="Contra6 Scoring API", version="0.1.0", lifespan=lifespan)


class ScoreCandidateIn(BaseModel):
    candidate_id: str
    raw_profile: dict[str, Any] = Field(default_factory=dict)


class ScoreRequest(BaseModel):
    jd_text: str
    candidates: list[ScoreCandidateIn]


class ScoreResponse(BaseModel):
    count: int
    cards: list[dict[str, Any]]


def _adapt_candidates(candidates: list[ScoreCandidateIn]) -> list[CandidateProfile]:
    adapter = ApifyJsonAdapter()
    profiles: list[CandidateProfile] = []
    for i, item in enumerate(candidates):
        record = dict(item.raw_profile or {})
        # Stamp the caller's id so results join back to their DB rows.
        record["_candidate_id"] = item.candidate_id
        profile = adapter.to_profile(record, i)
        profile.candidate_id = item.candidate_id
        profiles.append(profile)
    return profiles


def score_candidates(jd_text: str, candidates: list[ScoreCandidateIn]) -> list[dict[str, Any]]:
    _require_ready()
    if not jd_text or not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text is required")
    if not candidates:
        raise HTTPException(status_code=400, detail="candidates must be a non-empty list")

    profiles = _adapt_candidates(candidates)
    by_id = {p.candidate_id: p for p in profiles}

    jd = process_jd(jd_text, cache_path=_JD_CACHE_PATH)
    df = run_pipeline(
        jd_text=jd_text,
        profiles=profiles,
        processed_jd=jd,
        top_n=len(profiles),
        embedding_model=_embedding_model,
        sim_spec=_sim_spec,
        rerank_spec=_rerank_spec,
    )

    # run_pipeline returns a display subset; merge card metadata back in.
    meta_cols = ["candidate_id", "sector_text", "data_completeness_level"]
    meta = profiles_to_dataframe(profiles)[meta_cols]
    df = df.merge(meta, on="candidate_id", how="left")
    df = df.reset_index(drop=True)
    df["pipeline_rank"] = df.index + 1

    cards: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        profile = by_id.get(cid)
        if profile is None:
            continue
        card = build_card(profile, row.to_dict(), jd)
        card["candidate_id"] = cid  # preserve caller id
        cards.append(card)
    return cards


@app.get("/health")
def health():
    from core.model_cache import resolve_base_embedding_dtype, resolve_similarity_model_config

    sim_cfg = resolve_similarity_model_config()
    return {
        "status": "ok" if _models_ready else "warming",
        "models_ready": _models_ready and _embedding_model is not None,
        "similarity_model": "mpnet-only" if sim_cfg is None else "qwen",
        "embedding_dtype": resolve_base_embedding_dtype(),
        "sim_spec_loaded": _sim_spec is not None,
        "startup_process_rss_mb": _startup_rss_mb,
        "SIMILARITY_MODEL_env": os.environ.get("SIMILARITY_MODEL"),
        "BASE_EMBEDDING_DTYPE_env": os.environ.get("BASE_EMBEDDING_DTYPE"),
    }


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest):
    print(
        f"/score request: n_candidates={len(body.candidates)} "
        f"jd_chars={len(body.jd_text or '')} models_ready={_models_ready} "
        f"startup_process_rss_mb={_startup_rss_mb}",
        flush=True,
    )
    logger.info(
        "/score request: n_candidates=%s jd_chars=%s models_ready=%s",
        len(body.candidates),
        len(body.jd_text or ""),
        _models_ready,
    )
    try:
        cards = score_candidates(body.jd_text, body.candidates)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("score failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScoreResponse(count=len(cards), cards=cards)
