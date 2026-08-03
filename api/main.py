"""Thin FastAPI wrapper around the scoring pipeline.

POST /score
  body: { jd_text, candidates: [{ candidate_id, raw_profile }] }
  → ApifyJsonAdapter → run_pipeline → swipe-card JSON (tagged with caller candidate_id)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.adapters.apify_json_adapter import ApifyJsonAdapter
from core.data import profiles_to_dataframe
from core.jd_extraction import process_jd
from core.pipeline import run_pipeline
from core.swipe import build_card
from models.candidate import CandidateProfile

logger = logging.getLogger(__name__)

app = FastAPI(title="Contra6 Scoring API", version="0.1.0")

# Optional offline JD cache (e.g. jd/parsed/hr_assistant_prime_ac.json) for local
# sanity without a live Anthropic call. Leave unset in production.
_JD_CACHE_PATH = os.environ.get("JD_CACHE_PATH") or None


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
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(body: ScoreRequest):
    try:
        cards = score_candidates(body.jd_text, body.candidates)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("score failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ScoreResponse(count=len(cards), cards=cards)
