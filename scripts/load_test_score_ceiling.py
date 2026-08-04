"""Local candidate-count ceiling probe (mpnet-only, mirrors Render combo D).

Builds a 25-candidate batch by cloning the real 10-profile fixture with unique
ids (preserves realistic about/experience length), then runs the same staged
pipeline as scripts/profile_score_memory.py for N=10 and N=25.

    COPILOT_SKIP_CLI_DOWNLOAD=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
      BASE_EMBEDDING_DTYPE=fp16 SIMILARITY_MODEL=mpnet-only \\
      JD_CACHE_PATH=jd/parsed/hr_assistant_prime_ac.json \\
      uv run python scripts/load_test_score_ceiling.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("COPILOT_SKIP_CLI_DOWNLOAD", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("BASE_EMBEDDING_DTYPE", "fp16")
os.environ.setdefault("SIMILARITY_MODEL", "mpnet-only")

from core.adapters.apify_json_adapter import ApifyJsonAdapter  # noqa: E402
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
from models.mappings import encode_batch_size, rerank_top_k  # noqa: E402

FIXTURE = Path(os.environ.get("SCORE_FIXTURE", str(ROOT / ".ai-recruiter" / "real_score_batch_10.json")))
JD_CACHE = os.environ.get("JD_CACHE_PATH", "jd/parsed/hr_assistant_prime_ac.json")
SIZES = [int(x) for x in os.environ.get("CEILING_SIZES", "10,25").split(",") if x.strip()]


def expand_candidates(base: list[dict], n: int) -> list[dict]:
    """Clone real profiles with unique ids until len == n (cycle through base)."""
    out: list[dict] = []
    i = 0
    while len(out) < n:
        src = base[i % len(base)]
        clone = {
            "candidate_id": f"{src['candidate_id']}__clone{len(out)}",
            "raw_profile": copy.deepcopy(src["raw_profile"]),
        }
        out.append(clone)
        i += 1
    return out


def run_once(jd_text: str, raw_candidates: list[dict], embedding_model, sim_spec, rerank_spec) -> dict:
    trace = MemTrace()
    trace.start(use_tracemalloc=False)

    body_candidates = [
        {"candidate_id": c["candidate_id"], "raw_profile": c["raw_profile"]}
        for c in raw_candidates
    ]
    _ = json.dumps(body_candidates)
    trace.mark("01_after_body_parsed")

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

    jd = process_jd(jd_text, cache_path=JD_CACHE)
    trace.mark("03_after_process_jd")

    emb_model = sim_spec.model if sim_spec else embedding_model
    embed_profiles(
        profiles,
        emb_model,
        model_key=sim_spec.model_key if sim_spec else None,
        doc_instruction=sim_spec.doc_instruction if sim_spec else None,
        batch_size=sim_spec.batch_size if sim_spec else encode_batch_size,
    )
    trace.mark("04_after_embed_profiles")

    df = profiles_to_dataframe(profiles)
    trace.mark("05_after_profiles_to_dataframe")

    df = filter_by_job_title(df, jd.role, 0.4, model=embedding_model, mode="hybrid", hard=False)
    trace.mark("06_after_title_score")

    df = calculate_skill_score(df, jd, model=embedding_model, skill_mode="hybrid")
    trace.mark("07_after_skill_score")

    df = calculate_qualification_score(df, jd)
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
    df = apply_rerank(df, jd, rerank_spec, top_k=rerank_top_k)
    df = df.head(len(profiles)).reset_index(drop=True)
    trace.mark("12_after_rerank_head")

    meta = profiles_to_dataframe(profiles)[["candidate_id", "sector_text", "data_completeness_level"]]
    df = df.merge(meta, on="candidate_id", how="left")
    df = df.reset_index(drop=True)
    df["pipeline_rank"] = df.index + 1

    cards = []
    for _, row in df.iterrows():
        cid = row["candidate_id"]
        profile = by_id.get(cid)
        if profile is None:
            continue
        card = build_card(profile, row.to_dict(), jd)
        card["candidate_id"] = cid
        cards.append(card)
    _ = json.dumps({"count": len(cards), "cards": cards})
    trace.mark("16_after_json_serialize")
    trace.stop()

    by_label = {s.label: s for s in trace.samples}
    embed_delta = by_label["04_after_embed_profiles"].delta_from_prev_mb
    skill_delta = by_label["07_after_skill_score"].delta_from_prev_mb
    climb = trace.peak_rss_mb - trace.baseline_rss_mb
    return {
        "n": len(raw_candidates),
        "baseline_mb": round(trace.baseline_rss_mb, 1),
        "peak_mb": round(trace.peak_rss_mb, 1),
        "climb_mb": round(climb, 1),
        "embed_delta_mb": round(embed_delta, 1),
        "skill_delta_mb": round(skill_delta, 1),
        "report": trace.report(),
    }


def main() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    jd_text = data["jd_text"]
    base = data["candidates"]
    sizes = [len(json.dumps(c.get("raw_profile") or {})) for c in base]
    print(
        f"fixture={FIXTURE} base_n={len(base)} jd_chars={len(jd_text)} "
        f"raw_json_chars avg={sum(sizes)//len(sizes)} min={min(sizes)} max={max(sizes)}"
    )
    print("warming models (fp16 mpnet-only)...")
    embedding_model, sim_spec, rerank_spec = warm_scoring_models()
    print(f"sim_spec={None if sim_spec is None else sim_spec.model_key}")

    results = []
    for n in SIZES:
        cands = expand_candidates(base, n)
        print(f"\n===== N={n} =====", flush=True)
        row = run_once(jd_text, cands, embedding_model, sim_spec, rerank_spec)
        results.append(row)
        print(row["report"])
        print(
            f"SUMMARY n={row['n']} baseline={row['baseline_mb']} "
            f"peak={row['peak_mb']} climb={row['climb_mb']} "
            f"embed_d={row['embed_delta_mb']} skill_d={row['skill_delta_mb']}"
        )

    print("\n=== CEILING MATH ===")
    by_n = {r["n"]: r for r in results}
    if 10 in by_n and 25 in by_n:
        a, b = by_n[10], by_n[25]
        climb_per = (b["climb_mb"] - a["climb_mb"]) / (25 - 10)
        embed_per = (b["embed_delta_mb"] - a["embed_delta_mb"]) / (25 - 10)
        print(f"local climb@10={a['climb_mb']} MB  climb@25={b['climb_mb']} MB")
        print(f"marginal climb per extra candidate (10→25)={climb_per:.2f} MB")
        print(f"marginal embed delta per extra candidate={embed_per:.2f} MB")
        # Production: idle ~964, peak@10 ~1017 → climb ~53
        prod_idle = float(os.environ.get("PROD_IDLE_MB", "964"))
        prod_peak_10 = float(os.environ.get("PROD_PEAK_10_MB", "1017"))
        prod_climb_10 = prod_peak_10 - prod_idle
        # Scale production climb using local linearity ratio if local climb@10 > 0
        if a["climb_mb"] > 0:
            scale = prod_climb_10 / a["climb_mb"]
        else:
            scale = 1.0
        # Prefer production per-candidate from measured 10-cand climb if linear:
        # but we only have one production point; use local marginal * (prod_climb/local_climb)
        prod_marginal = climb_per * scale
        plan_mb = float(os.environ.get("PLAN_MB", "2048"))
        # Leave 150MB OS/allocator headroom
        headroom = plan_mb - prod_idle - 150
        if prod_marginal > 0:
            max_n = int(headroom / prod_marginal)
        else:
            max_n = 10_000
        # Also compute from absolute: idle + climb_10 + (n-10)*marginal < plan-150
        safe_n = 10
        for n in range(10, 500):
            est = prod_idle + prod_climb_10 + (n - 10) * prod_marginal
            if est >= plan_mb - 150:
                break
            safe_n = n
        print(f"prod_idle={prod_idle} prod_peak@10={prod_peak_10} prod_climb@10={prod_climb_10:.1f}")
        print(f"local→prod scale factor={scale:.2f}  prod_marginal≈{prod_marginal:.2f} MB/cand")
        print(f"safe_max_candidates (plan={plan_mb}MB, 150MB reserve) ≈ {safe_n}")
        print(json.dumps({"results": results, "safe_max_candidates": safe_n, "prod_marginal_mb": round(prod_marginal, 2)}, indent=2))


if __name__ == "__main__":
    main()
