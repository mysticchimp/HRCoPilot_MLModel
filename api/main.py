"""Thin FastAPI wrapper around the scoring pipeline.

POST /score
  body: { jd_text?, parsed_jd?, candidates: [{ candidate_id, raw_profile }] }
  → ApifyJsonAdapter → run_pipeline → swipe-card JSON (tagged with caller candidate_id)
  At least one of jd_text / parsed_jd is required. When parsed_jd is set, Claude is
  skipped and scoring_mode="parsed"; otherwise process_jd runs and scoring_mode="llm".

Embedding models (all-mpnet + optional Qwen similarity) are loaded once at startup
and reused. Uvicorn 0.52+ runs lifespan startup *before* binding the listen port, so
``warm_scoring_models()`` completes before any request can be accepted. We still gate
``/score`` on an explicit ready flag as belt-and-suspenders.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from core.adapters.apify_json_adapter import ApifyJsonAdapter
from core.data import profiles_to_dataframe
from core.embedding import embed_profiles
from core.filtering import filter_by_job_title
from core.jd_extraction import process_jd, resolve_jd_cache_dir
from core.mem_trace import RequestMemTrace, rss_mb
from core.model_cache import warm_scoring_models
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
from core.swipe import build_card
from models.candidate import CandidateProfile
from models.data_models import JobRoleSchema
from models.mappings import encode_batch_size, rerank_top_k

logger = logging.getLogger(__name__)

# Optional single-file JD cache (eval / local offline). Hash-keyed dir is separate
# (JD_CACHE_DIR, default .ai-recruiter/jd_cache) — see core.jd_extraction.process_jd.
_JD_CACHE_PATH = os.environ.get("JD_CACHE_PATH") or None
# Temporary production investigation: per-stage RSS in Render logs.
_SCORE_MEM_TRACE = os.environ.get("SCORE_MEM_TRACE", "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)
# Soft capacity gate (measured 2026-08-04): local n=10/25/40 climbs ~43–48 MB
# (sub-linear); prod idle ~964 + peak@10 ~1017. Conservative linearisation of the
# prod climb (53/10 ≈ 5.3 MB/cand) crosses 2GB−150MB reserve near n≈176. Soft max
# 100 leaves headroom for diverse skill pools and allocator noise. Override via env.
_SCORE_MAX_CANDIDATES = int(os.environ.get("SCORE_MAX_CANDIDATES", "100"))

# Populated by lifespan; reused on every /score.
_embedding_model = None
_sim_spec = None
_rerank_spec = None
# Explicit ready gate: False until warm_scoring_models() returns and globals are set.
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

    print(
        f"Config: similarity_model={sim_mode}, embedding_dtype={resolved_dtype}, "
        f"SIMILARITY_MODEL_env={raw_sim_env!r}, BASE_EMBEDDING_DTYPE_env={raw_dtype_env!r} "
        f"SCORE_MEM_TRACE={_SCORE_MEM_TRACE} SCORE_MAX_CANDIDATES={_SCORE_MAX_CANDIDATES} "
        f"JD_CACHE_PATH={_JD_CACHE_PATH!r} JD_CACHE_DIR={resolve_jd_cache_dir()}",
        flush=True,
    )
    logger.info(
        "Config: similarity_model=%s, embedding_dtype=%s "
        "(env SIMILARITY_MODEL=%r BASE_EMBEDDING_DTYPE=%r SCORE_MEM_TRACE=%s "
        "SCORE_MAX_CANDIDATES=%s JD_CACHE_PATH=%r JD_CACHE_DIR=%s)",
        sim_mode,
        resolved_dtype,
        raw_sim_env,
        raw_dtype_env,
        _SCORE_MEM_TRACE,
        _SCORE_MAX_CANDIDATES,
        _JD_CACHE_PATH,
        resolve_jd_cache_dir(),
    )
    print(
        "Warming scoring models (blocking; uvicorn has not bound the listen port yet)...",
        flush=True,
    )
    _embedding_model, _sim_spec, _rerank_spec = warm_scoring_models()
    try:
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
    jd_text: str | None = None
    candidates: list[ScoreCandidateIn]
    # Optional pre-parsed JD (e.g. Sourcing_Apify scoring brief). When set,
    # skips Claude for this request; scoring_mode="parsed".
    parsed_jd: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _require_jd_text_or_parsed(self) -> ScoreRequest:
        has_parsed = self.parsed_jd is not None
        has_text = bool((self.jd_text or "").strip())
        if not has_parsed and not has_text:
            raise ValueError("jd_text or parsed_jd is required")
        return self


class ScoreResponse(BaseModel):
    count: int
    cards: list[dict[str, Any]]
    scoring_mode: Literal["parsed", "llm"]


def _require_batch_fits(n: int) -> None:
    if n > _SCORE_MAX_CANDIDATES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"batch too large for current capacity: got {n} candidates, "
                f"max {_SCORE_MAX_CANDIDATES} per /score call "
                f"(SCORE_MAX_CANDIDATES; 2GB Render Standard headroom). "
                f"Split the pool and call /score in chunks."
            ),
        )


def _adapt_candidates(candidates: list[ScoreCandidateIn]) -> list[CandidateProfile]:
    adapter = ApifyJsonAdapter()
    profiles: list[CandidateProfile] = []
    for i, item in enumerate(candidates):
        record = dict(item.raw_profile or {})
        record["_candidate_id"] = item.candidate_id
        profile = adapter.to_profile(record, i)
        profile.candidate_id = item.candidate_id
        profiles.append(profile)
    return profiles


def _payload_stats(candidates: list[ScoreCandidateIn], jd_text: str) -> str:
    raw_chars = 0
    max_raw = 0
    for c in candidates:
        n = len(json.dumps(c.raw_profile or {}, ensure_ascii=False))
        raw_chars += n
        max_raw = max(max_raw, n)
    return (
        f"n={len(candidates)} jd_chars={len(jd_text or '')} "
        f"raw_profile_chars_total={raw_chars} raw_profile_chars_max={max_raw}"
    )


def score_candidates(
    jd_text: str | None,
    candidates: list[ScoreCandidateIn],
    mem: RequestMemTrace | None = None,
    parsed_jd: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], Literal["parsed", "llm"]]:
    _require_ready()
    has_parsed = parsed_jd is not None
    has_text = bool((jd_text or "").strip())
    if not has_parsed and not has_text:
        raise HTTPException(
            status_code=400,
            detail="jd_text or parsed_jd is required",
        )
    if not candidates:
        raise HTTPException(status_code=400, detail="candidates must be a non-empty list")
    _require_batch_fits(len(candidates))

    def _m(label: str, extra: str = "") -> None:
        if mem is not None:
            mem.mark(label, logger=logger, extra=extra)

    _m("01_after_body_parsed", _payload_stats(candidates, jd_text or ""))

    profiles = _adapt_candidates(candidates)
    by_id = {p.candidate_id: p for p in profiles}
    _m("02_after_adapter_profiles")

    if has_parsed:
        jd = JobRoleSchema.model_validate(parsed_jd)
        scoring_mode: Literal["parsed", "llm"] = "parsed"
        _m("03_after_process_jd", "source=request_parsed_jd")
    else:
        cache_dir = resolve_jd_cache_dir()
        _m(
            "03_before_process_jd",
            f"jd_cache_path={_JD_CACHE_PATH!r} jd_cache_dir={cache_dir}",
        )
        jd = process_jd(jd_text or "", cache_path=_JD_CACHE_PATH)
        scoring_mode = "llm"
        _m("03_after_process_jd", "source=process_jd")

    model = _embedding_model
    sim_spec = _sim_spec
    rerank_spec = _rerank_spec
    emb_model = sim_spec.model if sim_spec else model

    embed_profiles(
        profiles,
        emb_model,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else encode_batch_size,
    )
    _m("04_after_embed_profiles")

    df = profiles_to_dataframe(profiles)
    _m("05_after_profiles_to_dataframe")

    df = filter_by_job_title(df, jd.role, 0.4, model=model, mode="hybrid", hard=False)
    _m("06_after_title_score")

    df = calculate_skill_score(df, jd, model=model, skill_mode="hybrid")
    _m("07_after_skill_score")

    df = calculate_qualification_score(df, jd)
    _m("08_after_qualification")

    df = calculate_seniority_score(df, jd)
    df = calculate_experience_score(df, jd)
    df = calculate_industry_score(df, jd)
    df = calculate_language_score(df, jd)
    df = calculate_location_score(df, jd)
    df = calculate_attrition_score(df, jd)
    df = calculate_experience_relevance_score(df, jd)
    df = calculate_education_relevance_score(df, jd)
    _m("09_after_structured_scorers")

    df = calculate_similarity_score(
        df,
        jd,
        emb_model,
        query_instruction=sim_spec.query_instruction if sim_spec else None,
    )
    _m("10_after_similarity_score")

    df = calculate_total_score(df, jd)
    _m("11_after_total_score")

    df = apply_rerank(df, jd, rerank_spec, top_k=rerank_top_k)
    df = df.head(len(profiles)).reset_index(drop=True)
    _m("12_after_rerank_head")

    meta_cols = ["candidate_id", "sector_text", "data_completeness_level"]
    meta = profiles_to_dataframe(profiles)[meta_cols]
    df = df.merge(meta, on="candidate_id", how="left")
    df = df.reset_index(drop=True)
    df["pipeline_rank"] = df.index + 1
    _m("14_after_merge_meta")

    cards: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        profile = by_id.get(cid)
        if profile is None:
            continue
        card = build_card(profile, row.to_dict(), jd)
        card["candidate_id"] = cid
        cards.append(card)
    _m("15_after_build_cards")

    response = {"count": len(cards), "cards": cards}
    payload = json.dumps(response)
    _m("16_after_json_serialize", f"response_json_bytes={len(payload)}")

    if mem is not None:
        mem.summary(logger=logger)

    return cards, scoring_mode


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
        "score_mem_trace": _SCORE_MEM_TRACE,
        "score_max_candidates": _SCORE_MAX_CANDIDATES,
        "jd_cache_path": _JD_CACHE_PATH,
        "jd_cache_dir": str(resolve_jd_cache_dir()) if resolve_jd_cache_dir() else None,
        "SIMILARITY_MODEL_env": os.environ.get("SIMILARITY_MODEL"),
        "BASE_EMBEDDING_DTYPE_env": os.environ.get("BASE_EMBEDDING_DTYPE"),
    }


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest):
    req_id = uuid.uuid4().hex[:8]
    mem = None
    if _SCORE_MEM_TRACE:
        mem = RequestMemTrace(baseline_mb=_startup_rss_mb, request_id=req_id)

    mode_hint = "parsed" if body.parsed_jd is not None else "llm"
    print(
        f"/score request id={req_id} n_candidates={len(body.candidates)} "
        f"jd_chars={len(body.jd_text or '')} parsed_jd={body.parsed_jd is not None} "
        f"models_ready={_models_ready} "
        f"startup_process_rss_mb={_startup_rss_mb} mem_trace={_SCORE_MEM_TRACE} "
        f"max_candidates={_SCORE_MAX_CANDIDATES}",
        flush=True,
    )
    logger.info(
        "/score request id=%s n_candidates=%s jd_chars=%s has_parsed_jd=%s mode_hint=%s",
        req_id,
        len(body.candidates),
        len(body.jd_text or ""),
        body.parsed_jd is not None,
        mode_hint,
    )
    try:
        cards, scoring_mode = score_candidates(
            body.jd_text,
            body.candidates,
            mem=mem,
            parsed_jd=body.parsed_jd,
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        if mem is not None:
            mem.mark("ERROR_before_raise", logger=logger, extra=str(exc)[:200])
            mem.summary(logger=logger)
        logger.exception("score failed id=%s", req_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScoreResponse(count=len(cards), cards=cards, scoring_mode=scoring_mode)
