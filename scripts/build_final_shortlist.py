"""Build the final recruiter shortlist for the HR-Assistant JD.

Reciprocal Rank Fusion (RRF) of three rankings over the blind-judged pool (the top-50
pipeline ∪ top-50 LLM union = 78 candidates, so every plausible shortlist member is covered):
  - the ML pipeline ranking      (the scalable system output),
  - the original single-LLM ranking (Scored_FullPool `fit_0_10`),
  - the blind two-judge consensus (highest-fidelity relevance).

RRF (score = Σ 1/(k+rankᵢ)) is robust to any single ranker's misses. The fused ranking is
the recommended ONE-OFF deliverable; the ML pipeline alone remains the scalable product output.
DEFAULT fusion = pipeline + blind judges (the two trusted signals); pass --with-llm to also fold in
the original single-LLM ranking (the weakest signal, kept optional).

    COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/build_final_shortlist.py [--top-n 40] [--with-llm]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

JUDGMENTS = "evals/judgments/blind_judgments_hr_assistant.csv"
OUT = "evals/results/final_top30_combined.csv"
RRF_K = 60  # standard RRF constant; damps the influence of any single top rank


def _rrf_term(rank, k=RRF_K):
    return 1.0 / (k + rank) if pd.notna(rank) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=40, help="shortlist size to write (cut at 30 or 40)")
    ap.add_argument("--with-llm", action="store_true",
                    help="also fold in the original single-LLM ranking (weakest signal; default: pipeline + judges only)")
    args = ap.parse_args()

    j = pd.read_csv(JUDGMENTS)
    rat_col = next((c for c in j.columns if c.endswith("_rationale")), None)

    j["rrf"] = j["pipeline_rank"].map(_rrf_term) + j["consensus_rank"].map(_rrf_term)
    if args.with_llm:
        j["rrf"] += j["llm_rank"].map(_rrf_term)
    j = j.sort_values("rrf", ascending=False).reset_index(drop=True)
    j["final_rank"] = j.index + 1

    cols = ["final_rank", "candidate_id", "current_title", "rrf",
            "pipeline_rank", "llm_rank", "judge_mean_score", "consensus_rank"]
    rename = {"consensus_rank": "judge_rank"}
    if rat_col:
        cols.append(rat_col)
        rename[rat_col] = "why"
    out = j[cols].rename(columns=rename).head(args.top_n)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    fusion = "pipeline + LLM + judges" if args.with_llm else "pipeline + judges"
    print(f"wrote top-{args.top_n} -> {OUT}  (RRF fusion: {fusion})\n")
    show = ["final_rank", "candidate_id", "current_title", "pipeline_rank", "llm_rank", "judge_mean_score", "judge_rank"]
    print(out.head(15)[show].to_string(index=False))


if __name__ == "__main__":
    main()
