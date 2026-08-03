"""Run the reverse-matching eval harness.

Examples (from project root):
    # generate a tiny linkedin pilot (uses Copilot claude-opus-4.7)
    uv run python scripts/run_eval.py --dataset linkedin --n-per-group 1 --max-seeds 2 --generate

    # re-run offline against cached fixtures (deterministic, no LLM)
    uv run python scripts/run_eval.py --dataset linkedin
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.adapters.resume_adapter import ResumeAdapter
from core.llm import get_provider
from evals.cases import EvalCase, build_linkedin_gold_case, build_reverse_match_case, load_fixture, sample_seeds
from evals.runner import PipelineConfig, aggregate, evaluate_cases
from evals.skew import dataset_skew
from models.mappings import similarity_model_config

RESUME_CSV = "data/resume_data.csv"
LINKEDIN_CSV = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"
# The JD the Scored_FullPool (fit_0_10) grades were actually produced against.
HR_JD = "jd/HR Assistant — Prime Focus Group (Prime AC).md"


def load_profiles(dataset):
    if dataset == "resume":
        return ResumeAdapter().to_profiles(RESUME_CSV)
    return LinkedInAdapter().to_profiles(LINKEDIN_CSV)


def _build_or_load(seed, dataset, provider, generate, reload=False):
    try:
        if generate:
            return build_reverse_match_case(seed, provider, dataset, reload=reload)
        return load_fixture(dataset, seed.candidate_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip seed {seed.candidate_id}] {type(exc).__name__}: {exc}", flush=True)
        return None


def build_cases(dataset, profiles, args, provider):
    group_key = (lambda p: p.job_title) if dataset == "resume" else (lambda p: p.seniority)
    seeds = sample_seeds(profiles, args.n_per_group, group_key=group_key)
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]

    results: list[EvalCase | None] = [None] * len(seeds)
    if args.concurrency > 1 and args.generate:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_build_or_load, s, dataset, provider, args.generate, args.reload): i for i, s in enumerate(seeds)}
            for done, fut in enumerate(as_completed(futures), start=1):
                results[futures[fut]] = fut.result()
                print(f"  [{done}/{len(seeds)}] fixtures ready", flush=True)
    else:
        for i, seed in enumerate(seeds, start=1):
            results[i - 1] = _build_or_load(seed, dataset, provider, args.generate, args.reload)
            if args.generate:
                print(f"  [{i}/{len(seeds)}] fixtures ready", flush=True)
    cases: list[EvalCase] = [case for case in results if case is not None]

    if dataset == "linkedin":
        if args.generate:
            try:
                cases.append(build_linkedin_gold_case(profiles, HR_JD, provider, reload=args.reload))
            except Exception as exc:  # noqa: BLE001
                print(f"[skip gold] {type(exc).__name__}: {exc}", flush=True)
        else:
            gold = load_fixture("linkedin", "_gold_hr_assistant")
            if gold:
                cases.append(gold)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["resume", "linkedin"], default="linkedin")
    parser.add_argument("--n-per-group", type=int, default=2)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1, help="parallel fixture-generation workers")
    parser.add_argument("--generate", action="store_true", help="call the LLM for missing fixtures")
    parser.add_argument("--reload", action="store_true", help="force-regenerate reverse-match fixtures even if cached")
    parser.add_argument("--provider", default="copilot")
    parser.add_argument("--model", default="claude-opus-4.7")
    parser.add_argument("--title-mode", default="hybrid")
    parser.add_argument("--skill-mode", default=None, choices=["fuzzy", "semantic", "hybrid"],
                        help="skill matcher (default: PipelineConfig default = hybrid)")
    champion_similarity = similarity_model_config or {}
    parser.add_argument("--embedding-model", default=champion_similarity.get("model_name", "all-mpnet-base-v2"),
                        help="isolated embedding model for similarity_score; title/skill keep all-mpnet")
    parser.add_argument("--sim-query-instruction", default=champion_similarity.get("query_instruction") or "",
                        help="instruction prefix prepended to the JD (query side) for instruction-tuned encoders")
    parser.add_argument("--sim-doc-instruction", default=champion_similarity.get("doc_instruction") or "",
                        help="instruction prefix prepended to candidate profiles (document side)")
    parser.add_argument("--sim-device", default=champion_similarity.get("device") or "",
                        help="force device for the similarity model (e.g. 'cpu' for Jasper, which NaNs on Apple MPS)")
    parser.add_argument("--sim-dtype", default=champion_similarity.get("dtype", "auto"), choices=["auto", "fp32", "fp16", "bf16"],
                        help="similarity-model dtype; fp16 halves MPS memory (auto: fp32 on cpu, model default on mps)")
    parser.add_argument("--sim-max-seq", type=int, default=champion_similarity.get("max_seq_length", 1024),
                        help="cap max_seq_length of the similarity model (memory; still >all-mpnet's 384)")
    parser.add_argument("--sim-batch-size", type=int, default=champion_similarity.get("batch_size", 16),
                        help="encode batch size for the similarity model (lower = less peak memory)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    provider = get_provider(args.provider, model=args.model) if args.generate else None
    profiles = load_profiles(args.dataset)

    print(f"=== SKEW ({args.dataset}, n={len(profiles)}) ===")
    print(json.dumps(dataset_skew(profiles), indent=2, default=str))

    gen_start = time.perf_counter()
    cases = build_cases(args.dataset, profiles, args, provider)
    print(f"\n{len(cases)} cases loaded/generated in {time.perf_counter() - gen_start:.1f}s")
    if not cases:
        print("No cases — run with --generate to create fixtures.")
        return

    os.makedirs(".ai-recruiter", exist_ok=True)
    base_model = SentenceTransformer("all-mpnet-base-v2")  # drives title + skill semantic legs
    cfg_kwargs = {"title_mode": args.title_mode}
    if args.skill_mode:
        cfg_kwargs["skill_mode"] = args.skill_mode
    config = PipelineConfig(**cfg_kwargs)

    # Isolated similarity model (Option B). CLI defaults come from the production
    # champion mapping so a plain offline run regenerates the actual champion baseline.
    sim_spec = None
    cache_path = f".ai-recruiter/emb_{args.dataset}.pkl"
    is_default = (
        args.embedding_model == "all-mpnet-base-v2"
        and not args.sim_query_instruction
        and not args.sim_doc_instruction
    )
    if not is_default:
        # Prefer the shared builder so fp16 loads via torch_dtype (no .half() spike).
        from core.embedding import build_similarity_spec

        sim_cfg = {
            "model_name": args.embedding_model,
            "query_instruction": args.sim_query_instruction or None,
            "doc_instruction": args.sim_doc_instruction or None,
            "dtype": args.sim_dtype,
            "device": args.sim_device,
            "max_seq_length": args.sim_max_seq,
            "batch_size": args.sim_batch_size,
        }
        sim_spec = build_similarity_spec(sim_cfg, base_model=base_model)
        slug = args.embedding_model.replace("/", "_")
        if args.sim_query_instruction or args.sim_doc_instruction:
            slug += "_instr"
        cache_path = f".ai-recruiter/emb_{args.dataset}_{slug}.pkl"
        print(f"[similarity] isolated model={args.embedding_model} "
              f"q_instr={bool(args.sim_query_instruction)} d_instr={bool(args.sim_doc_instruction)}")

    per_case = evaluate_cases(
        cases, profiles, base_model, config,
        embedding_cache_path=cache_path, sim_spec=sim_spec,
    )
    summary = aggregate(per_case)

    print("\n=== METRICS ===")
    print(json.dumps(summary, indent=2))
    if args.out:
        emb_meta = {
            "embedding_model": args.embedding_model,
            "sim_query_instruction": args.sim_query_instruction or None,
            "sim_doc_instruction": args.sim_doc_instruction or None,
        }
        with open(args.out, "w") as fh:
            json.dump(
                {"config": vars(config), "embedding": emb_meta, "summary": summary, "per_case": per_case},
                fh, indent=2, default=str,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
