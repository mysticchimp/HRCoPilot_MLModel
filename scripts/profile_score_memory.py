"""Full /score lifecycle memory profile (investigation only — no product fix).

Replays the same path as ``api.main.score_candidates`` against the real 10-candidate
fixture, with RSS + tracemalloc snapshots at every stage and a background peak sampler.

    COPILOT_SKIP_CLI_DOWNLOAD=1 JD_CACHE_PATH=jd/parsed/hr_assistant_prime_ac.json \\
      uv run python scripts/profile_score_memory.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("COPILOT_SKIP_CLI_DOWNLOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from core.data import profiles_to_dataframe  # noqa: E402
from core.embedding import embed_profiles  # noqa: E402
from core.filtering import filter_by_job_title  # noqa: E402
from core.jd_extraction import process_jd  # noqa: E402
from core.mem_trace import MemTrace  # noqa: E402
from core.model_cache import warm_scoring_models  # noqa: E402
from core.scoring import (  # noqa: E402
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
from core.swipe import build_card  # noqa: E402
from core.adapters.apify_json_adapter import ApifyJsonAdapter  # noqa: E402
from models.mappings import rerank_top_k  # noqa: E402

FIXTURE = Path(os.environ.get("SCORE_FIXTURE", str(ROOT / ".ai-recruiter" / "real_score_batch_10.json")))
JD_CACHE = os.environ.get("JD_CACHE_PATH", "jd/parsed/hr_assistant_prime_ac.json")


def main():
    try:
        import psutil  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "psutil"])

    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    jd_text = data["jd_text"]
    raw_candidates = data["candidates"]
    print(f"fixture={FIXTURE} n={len(raw_candidates)} jd_chars={len(jd_text)}")

    print("warming models...")
    embedding_model, sim_spec, rerank_spec = warm_scoring_models()
    print(f"models ready (sim_batch={sim_spec.batch_size if sim_spec else None})")

    trace = MemTrace()
    trace.start(use_tracemalloc=True)

    # --- 1. request body as Python objects (already loaded JSON = parsed body) ---
    body_candidates = [
        {"candidate_id": c["candidate_id"], "raw_profile": c["raw_profile"]}
        for c in raw_candidates
    ]
    # Force materialization / retain refs like FastAPI would
    _ = json.dumps(body_candidates)
    trace.mark("01_after_body_parsed")

    # --- 2. ApifyJsonAdapter ---
    adapter = ApifyJsonAdapter()
    profiles = []
    for i, item in enumerate(body_candidates):
        record = dict(item["raw_profile"] or {})
        record["_candidate_id"] = item["candidate_id"]
        profile = adapter.to_profile(record, i)
        profile.candidate_id = item["candidate_id"]
        profiles.append(profile)
    by_id = {p.candidate_id: p for p in profiles}
    trace.mark("02_after_adapter_profiles")

    # --- 3. JD parse (cached) ---
    jd = process_jd(jd_text, cache_path=JD_CACHE)
    trace.mark("03_after_process_jd")

    # --- 4. run_pipeline stages (same order as core.pipeline.run_pipeline) ---
    emb_model = sim_spec.model if sim_spec else embedding_model
    embed_profiles(
        profiles,
        emb_model,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else 2,
    )
    trace.mark("04_after_embed_profiles")

    df = profiles_to_dataframe(profiles)
    trace.mark("05_after_profiles_to_dataframe")

    df = filter_by_job_title(df, jd.role, 0.4, model=embedding_model, mode="hybrid", hard=False)
    trace.mark("06_after_title_score")

    df = calculate_skill_score(df, jd, model=embedding_model, skill_mode="hybrid")
    trace.mark("07_after_skill_score")

    df = calculate_qualification_score(df, jd)
    trace.mark("08_after_qualification")

    df = calculate_seniority_score(df, jd)
    df = calculate_experience_score(df, jd)
    df = calculate_industry_score(df, jd)
    df = calculate_language_score(df, jd)
    df = calculate_location_score(df, jd)
    df = calculate_attrition_score(df, jd)
    df = calculate_experience_relevance_score(df, jd)
    df = calculate_education_relevance_score(df, jd)
    trace.mark("09_after_structured_scorers")

    df = calculate_similarity_score(
        df, jd, emb_model,
        query_instruction=sim_spec.query_instruction if sim_spec else None,
    )
    trace.mark("10_after_similarity_score")

    df = calculate_total_score(df, jd)
    trace.mark("11_after_total_score")

    df = apply_rerank(df, jd, rerank_spec, top_k=rerank_top_k)
    df = df.head(len(profiles)).reset_index(drop=True)
    trace.mark("12_after_rerank_head")

    # --- 5. response assembly (same as api.main.score_candidates) ---
    meta = profiles_to_dataframe(profiles)[["candidate_id", "sector_text", "data_completeness_level"]]
    trace.mark("13_after_second_profiles_to_df")

    df = df.merge(meta, on="candidate_id", how="left")
    df = df.reset_index(drop=True)
    df["pipeline_rank"] = df.index + 1
    trace.mark("14_after_merge_meta")

    cards = []
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        profile = by_id.get(cid)
        if profile is None:
            continue
        card = build_card(profile, row.to_dict(), jd)
        card["candidate_id"] = cid
        cards.append(card)
    trace.mark("15_after_build_cards")

    response = {"count": len(cards), "cards": cards}
    payload = json.dumps(response)
    trace.mark("16_after_json_serialize")
    print(f"response_json_bytes={len(payload)}")

    # Hold refs so nothing is freed early (mirrors request still in flight)
    _keep = (body_candidates, profiles, by_id, jd, df, cards, response, payload)
    trace.mark("17_all_refs_still_live")

    trace.stop()
    print()
    print(trace.report())
    print()
    print(f"KEEP_ALIVE_OBJECTS={len(_keep)} cards={len(cards)}")


if __name__ == "__main__":
    main()
