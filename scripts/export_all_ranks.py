"""Export all 145 candidates with their three rankings — ML pipeline, original single-LLM,
and blind two-judge consensus — into one CSV for ad-hoc sorting/analysis in Excel.

Candidates outside the blind-judged pool (the top-50 union) get BLANK judge columns.

    COPILOT_SKIP_CLI_DOWNLOAD=1 uv run python scripts/export_all_ranks.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

PIPELINE = "evals/results/pipeline_run_hr_assistant.csv"
GOLD = "data/Scored_FullPool_HR_Assistant_v2_2026-07-06_1843.csv"
JUDGMENTS = "evals/judgments/blind_judgments_hr_assistant.csv"
OUT = "evals/results/all_candidate_ranks.csv"


def main():
    p = pd.read_csv(PIPELINE)
    g = (pd.read_csv(GOLD, skiprows=2)[["linkedinUrl", "rank", "fit_0_10", "current_title"]]
         .rename(columns={"rank": "llm_rank", "fit_0_10": "llm_fit_0_10"}))
    j = pd.read_csv(JUDGMENTS)
    rat_col = next((c for c in j.columns if c.endswith("_rationale")), None)
    jcols = ["candidate_id", "consensus_rank", "judge_mean_score"] + ([rat_col] if rat_col else [])
    jr = {"consensus_rank": "judge_rank"}
    if rat_col:
        jr[rat_col] = "why"
    j = j[jcols].rename(columns=jr)

    df = p.merge(g, left_on="linkedin_url", right_on="linkedinUrl", how="left")
    df = df.merge(j, on="candidate_id", how="left")
    df["current_title"] = df["current_title"].fillna(df["job_title"])

    cols = ["candidate_id", "current_title", "pipeline_rank", "llm_rank",
            "judge_rank", "judge_mean_score", "llm_fit_0_10", "total_score"]
    if rat_col:
        cols.append("why")
    out = df[cols].copy()
    # nullable ints -> blanks stay blank, and judged ranks print as "1" not "1.0"
    for c in ("llm_rank", "judge_rank", "llm_fit_0_10"):
        out[c] = out[c].astype("Int64")
    out["total_score"] = out["total_score"].round(4)
    out = out.sort_values("pipeline_rank")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    n = len(out)
    n_judged = int(out["judge_rank"].notna().sum())
    print(f"wrote {OUT}  ({n} candidates; {n_judged} with a blind rank, {n - n_judged} blank)")
    print(out.head(10)[["candidate_id", "pipeline_rank", "llm_rank", "judge_rank", "judge_mean_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
