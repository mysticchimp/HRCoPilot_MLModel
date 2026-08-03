"""Benchmark Copilot models on the JD-generation task: latency + blind quality judging.

Runs the candidate models in parallel on the SAME candidate profile (the exact
reduced-profile input the eval harness uses), times each call individually, then
asks a judge model to score/rank the outputs blind (models anonymized as A/B/C).

    uv run python scripts/bench_models.py
    uv run python scripts/bench_models.py --sequential          # cleaner per-model latency
    uv run python scripts/bench_models.py --models gpt-5.5 claude-sonnet-4.5

You compare latency from the table; the judge compares quality.
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.jd_generation import generate_jd_from_profile
from core.llm import get_provider
from evals.cases import reduced_profile_payload

LINKEDIN_CSV = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"
DEFAULT_MODELS = ["claude-opus-4.7", "claude-sonnet-4.5", "gpt-5.5"]
JUDGE_MODEL = "claude-opus-4.8"


def pick_profile(candidate_id=None):
    profiles = LinkedInAdapter().to_profiles(LINKEDIN_CSV)
    if candidate_id:
        return next(p for p in profiles if p.candidate_id == candidate_id)
    for profile in profiles:
        if profile.summary and profile.responsibilities and len(profile.skills) >= 5:
            return profile
    return profiles[0]


def generate_one(model, payload):
    """Generate a JD with one model; time the call and capture errors."""
    provider = get_provider("copilot", model=model)
    start = time.perf_counter()
    try:
        jd = generate_jd_from_profile(payload, provider=provider)
        return {"model": model, "latency": time.perf_counter() - start, "jd": jd, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"model": model, "latency": time.perf_counter() - start, "jd": None, "error": f"{type(exc).__name__}: {exc}"}


def build_judge_prompt(labeled):
    blocks = [f"=== Candidate {label} ===\n{jd}" for label, _model, jd in labeled]
    return (
        "You are judging job descriptions that were each generated from the SAME candidate profile "
        "by different models. Rate each on: relevance to the profile, specificity (concrete, not generic), "
        "realism, structure, and absence of fluff/hallucination.\n\n"
        + "\n\n".join(blocks)
        + "\n\nRespond with:\n"
        "1) A table: Candidate | Score /10 | one-line reason\n"
        "2) A ranking best-to-worst\n"
        "3) A 2-sentence overall verdict."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--sequential", action="store_true", help="run models one at a time (cleaner latency)")
    args = parser.parse_args()

    profile = pick_profile(args.candidate_id)
    payload = reduced_profile_payload(profile)
    print(f"Profile: {profile.candidate_id} | {profile.job_title} | seniority={profile.seniority}")
    print(f"Task: generate JD from reduced profile (skills/education held out).")
    print(f"Models: {args.models} | judge: {args.judge_model} | mode: {'sequential' if args.sequential else 'parallel'}\n")

    wall_start = time.perf_counter()
    if args.sequential:
        results = [generate_one(m, payload) for m in args.models]
    else:
        with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
            results = list(pool.map(lambda m: generate_one(m, payload), args.models))
    wall = time.perf_counter() - wall_start

    print("=== LATENCY ===")
    for r in sorted(results, key=lambda r: r["latency"]):
        print(f"  {r['model']:20} {r['latency']:7.1f}s   {'OK' if not r['error'] else r['error']}")
    print(f"  {'wall-clock (batch)':20} {wall:7.1f}s\n")

    ok = [r for r in results if r["jd"]]
    if len(ok) < 2:
        for r in ok:
            print(f"--- {r['model']} ---\n{r['jd']}\n")
        print("Not enough successful generations to judge.")
        return

    # anonymize + shuffle so the judge is blind to model identity
    labels = ["A", "B", "C", "D", "E"]
    random.shuffle(ok)
    labeled = [(labels[i], r["model"], r["jd"]) for i, r in enumerate(ok)]

    print(f"=== QUALITY JUDGE ({args.judge_model}, blind) ===")
    try:
        verdict = get_provider("copilot", model=args.judge_model).generate_text(build_judge_prompt(labeled))
        print(verdict.strip())
    except Exception as exc:  # noqa: BLE001
        print(f"[judge failed] {type(exc).__name__}: {exc}")

    print("\n=== label -> model ===")
    for label, model, _ in labeled:
        print(f"  {label} = {model}")

    print("\n=== generated JDs ===")
    for label, model, jd in labeled:
        print(f"\n----- {label} ({model}) -----\n{jd.strip()}")


if __name__ == "__main__":
    main()
