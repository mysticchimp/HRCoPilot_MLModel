# evals/ — evaluation harness & artifacts

The harness **code** stays at the top level (imported as `evals.*`); generated **artifacts**
are grouped into subfolders by type so they're easy to find.

## Layout

| Path | What |
|---|---|
| `cases.py` · `runner.py` · `metrics.py` · `skew.py` | The eval harness (imported as `evals.*`; do not move). |
| `fixtures/` | Committed eval fixtures — reverse-match cases + the silver Judge-grade anchor, per dataset (`fixtures/linkedin/`). |
| `results/` | Pipeline run outputs: `pipeline_run_hr_assistant.csv`, `pipeline_vs_llm_hr_assistant.csv`, the RRF deliverable `final_top30_combined.csv`, and the harness baseline `baseline_linkedin.json`. (`*_rerank.csv` = the shelved cross-encoder experiment.) |
| `judgments/` | **Blind LLM Judge-panel outputs** — `blind_judgments_hr_assistant.csv` (per-candidate 0–100 Judge grades, Section grades, evidence flags, credited preferred signals, and rationales) plus `blind_ranking_comparison_hr_assistant.json` (agreement/validation summary). The silver anchor labels come from here. |
| `reports/` | Human-readable markdown writeups. |

## Where each artifact comes from

- `results/pipeline_run_hr_assistant.csv`, `results/pipeline_vs_llm_hr_assistant.csv` ← `scripts/run_hr_assistant.py`
- `judgments/*` ← `scripts/blind_judge_rankings.py` (live LLM; frozen judged cohort, checkpointed batches, staged validation, then explicit promotion)
- `results/baseline_linkedin.json` ← `scripts/run_eval.py --out …`
- `results/t3_c5_reablation.json` records the measured common-control sweep; `results/t3_product_weight_comparison.json` records the later tiny structural product override and original-LLM comparison.
- The silver anchor (legacy identifier `fixtures/linkedin/_gold_hr_assistant.json`) draws its Judge grades from
  `judgments/blind_judgments_hr_assistant.csv` — see `evals/cases.py:build_linkedin_gold_case`.

## Reports

- **`reports/blind_gold_report.md`** — current silver Judge-grade anchor (78-candidate judged cohort) + the
  pipeline-vs-LLM verdict. **← newest.**
- `reports/blind_comparison_report.md` — earlier blind adjudication (pre-Qwen/hybrid; historical).
- `reports/pipeline_improvement_report.md` — the baseline → champion journey.
