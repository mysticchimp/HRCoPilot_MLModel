# Silver Judge-grade anchor — T3 tenure/relevance/language regrade (2026-07-31)

The same frozen **78-candidate judged cohort** was regraded with a neutral, additive rubric that
explicitly covers tenure, career relevance mix, and preferred languages. The original top-50 union
membership was held fixed so label changes measure the rubric/evidence change rather than cohort drift.

> Still silver, circular, and n=1: we told the Judge panel what to weigh, then calibrated components
> to those grades. Recruiter swipes/U2 and multiple human-labeled JDs remain the real validation.

## Method

- **Judged cohort:** frozen pipeline top-50 ∪ original-LLM top-50 → 78 unique candidates.
- **Judges:** `claude-opus-4.8` + `gpt-5.5`, **blind** (candidate keys anonymised + shuffled; source
  ranks/fit labels hidden), scoring nine bounded sections totaling 100. Python derives the total grade.
- **Evidence:** all raw work-history roles/durations/employment types/dates; no pipeline-computed tenure,
  relevance, education tier, component score, or completeness value.
- **Anti-inference:** structured preferred-signal credits fail closed if Tagalog/Filipino is not explicitly
  listed. Both judges credited only `efrelyn-ablay`; Che has no listed language and received no credit.
- **Agreement:** overall Spearman **.9222**. Mapped gates all pass: tenure **.6504** / MAE .77 (n=76),
  career relevance **.8844** / MAE .62 (n=76), preferred signals **.8132** / MAE 1.21 (n=78).

## Headline results

| Ranking | NDCG@10 | NDCG@20 |
|---|---:|---:|
| Frozen pre-reblend pipeline | **0.9490** | **0.9320** |
| Original LLM shortlist | 0.6860 | 0.7271 |

After the preregistered C5 common-control re-ablation, the adopted champion (`language=.15`,
`education=0`) scores **NDCG@10 .9588 / NDCG@5 .9464**. Reverse metrics: MRR .5190,
hit@3 .5789, hit@5 .6316, hit@10 .7895, seed-found 1.0.

## Blind-consensus top 10

| # | candidate | judge score | pipeline # | LLM # |
|--:|---|--:|--:|--:|
| 1 | muhammed-ashar-k | 82.5 | 4 | 54 |
| 2 | sabeeh-ansari | 79.0 | 7 | 5 |
| 3 | ranjani-anthony-raj | 79.5 | 31 | 53 |
| 3 | samasthasunoj | 79.5 | 3 | 20 |
| 5 | megha-p-s | 79.2 | 19 | 25 |
| 6 | harikrishna-r | 79.0 | 10 | 4 |
| 7 | savaf-methiyil | 76.5 | 5 | 39 |
| 7 | shafas-hussain | 76.5 | 21 | 24 |
| 7 | aiswarya-ravindran | 76.5 | 8 | 34 |
| 10 | shahul-p | 77.0 | 20 | 113 |

Consensus rank averages each judge's rank percentile, so tied/nearby scores need not be monotonic.

## Artifacts

- Per-candidate Judge/Section grades, evidence flags, credited signals, and rationales:
  `evals/judgments/blind_judgments_hr_assistant.csv`
- Agreement, validation gates, and consensus top-20: `evals/judgments/blind_ranking_comparison_hr_assistant.json`
- C5 sweeps + selected subset: `evals/results/t3_c5_reablation.json`
- Product structural-weight comparison: `evals/results/t3_product_weight_comparison.json`
- Champion ranking of all 145: `evals/results/pipeline_run_hr_assistant.csv`
- Champion baseline: `evals/results/baseline_linkedin.json`

## Decision

The rubric/JD change resets gold floors from the **new incumbent**, not the adopted gain: raw
NDCG@5 .93449 / NDCG@10 .94831 → floors `.93/.94` (EXCEPTION #7). Reverse floors are unchanged.
C5 measured decision: attrition `0`, experience relevance `0`, education `0`, language `.15`. A subsequent
product override adds attrition `.005`, experience relevance `.015`, and education `.005`: NDCG@5/10 stays
`.94644/.95879`, NDCG@20 improves `.92893→.95175`, and reverse MRR improves `.5190→.5212`. Original-LLM
fit-NDCG@10/top-k overlap is unchanged; rank Spearman slips `.2600→.2568`. Floors were not ratcheted because
the primary NDCG@5/10 metrics are unchanged and this remains circular n=1 silver.
