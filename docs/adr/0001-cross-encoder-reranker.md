---
status: accepted
---

# Cross-encoder reranker as a two-stage top-K rerank, not a blended component

We add a cross-encoder (`Alibaba-NLP/gte-reranker-modernbert-base`) as a **second stage**
that re-scores only the top-50 of the existing pipeline's ranking and reorders that Head,
rather than adding it as another weighted Component over the full pool. The cross-encoder
**supersedes the bi-encoder `similarity_score` (v1 → v2) within the Head only**; Head
membership is frozen (it reorders positions 1..50, never crossing into the tail), and the
tail keeps its Stage-1 order. It is config-gated (`rerank_model_config = None` disables it)
and therefore fully reversible. Architecture is accepted; whether it *ships* is gated on
the measured-adoption test below. Implementation pending.

## Considered options

- **Blended full-pool component (rejected).** Scoring `rerank_score` over all 145 and
  adding it as a 9th weighted Component double-counts the semantic signal, dilutes the
  cross-encoder's precision at a small weight, and isn't "reranking the head" at all.
- **Replacing the bi-encoder entirely (rejected).** The bi-encoder is the cheap, cacheable
  Stage-1 Retriever that selects the Head and orders the tail; a 149M cross-encoder can't
  be the only scorer at scale, and we'd lose recall.
- **Pure cross-encoder Head order (rejected as default).** Letting the CE alone reorder the
  Head discards the recruiter-priority Components (skill, industry, seniority) exactly where
  they matter most; keeping them in the Head fusion (v2 in similarity's 0.45 slot, other
  Components unchanged) is the chosen "B1" shape.

## Consequences

- **Recall ceiling.** The Reranker can only reorder what Stage 1 surfaced; a relevant
  candidate buried below K is never rescued (concretely, the Judge panel's #1, harikrishna,
  once sat at pipeline rank 51). K = 50 is therefore **provisional**, gated on a recall@K
  check against the judge-loved set. A miss beyond K is a *Retriever* recall hole, not the
  Reranker's to fix.
- **Measured, not product-motivated, adoption.** Because we build a frozen,
  pipeline-independent silver anchor (Judge grades), the Reranker ships only if its Head
  beats the bi-encoder Head on Judge NDCG@10 (plus Kendall τ toward the judge order /
  rescued misses) and holds the existing regression guardrails; otherwise it stays
  implemented but off behind the flag.
- **Ground truth is the frozen Judge grades, not `evals/results/final_top30_combined.csv`.** The
  fused file is an RRF over the pipeline's own ranks (circular). Using frontier judges to
  grade a 149M cross-encoder is teacher-grades-student, not the LLM-vs-LLM-label circularity
  the repo warns about for an LLM reranker.

## Outcome (measured 2026-07-16) — SHELVED, kept off behind the flag

`gte-reranker-modernbert-base` **failed the measured-adoption gate and was not adopted.**
Against the frozen blind-Judge grades (interim n=35, all inside the top-50):

- **Head NDCG@10 regressed 0.954 → 0.792** (−0.162; also −0.090 @5, −0.146 @20).
- Raw-score agreement with the judges: **bi-encoder Spearman +0.687 vs cross-encoder +0.188** —
  the CE's ordering is much weaker, so it is not a fusion/temperature artifact (CE scores also
  saturated in [0.815, 0.924]; spreading a +0.19 signal only amplifies the harm). The judges'
  #1 (harikrishna, 82) sank from bi #10 to CE #33.

This is the classic *reranker-hurts-a-strong-retriever* effect (the NVIDIA finding) plus domain
mismatch — a general web-passage reranker on long HR profiles, against a Qwen retriever already at
~0.95 NDCG@10 vs the judges. `rerank_model_config` stays **None (OFF)**; the two-stage machinery
remains implemented and reversible for a future re-test (a domain/instruction reranker, or an
expanded human-graded anchor). The measured gate did its job: it caught a −0.16 regression that
product-motivated adoption would have shipped blind. Status **accepted** = the *architecture/measurement
approach* is accepted; the specific gte model is shelved.
