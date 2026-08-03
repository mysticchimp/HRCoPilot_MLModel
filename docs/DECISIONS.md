# Decisions & Rationale — AI Recruiter Pipeline

Captures **(A)** the original pipeline issues and how they were resolved, and **(B)** the key
design decisions with rationale — so a new session doesn't re-open settled calls. For the metric
journey see `evals/reports/pipeline_improvement_report.md`.

## A. Original pipeline issues (Phase 1) → resolution

| Issue (original pipeline) | Resolution |
|---|---|
| **Degenerate embedding** — candidate vector used responsibilities only, identical within a job title → the 0.45-weight semantic signal gave no within-role discrimination | Enriched candidate embedding = career-objective/summary + responsibilities + **skills** |
| **Hard title gate** — brittle fuzzy cutoff dropped relevant candidates (e.g. "HR Executive" for an "HR Assistant" JD) | Soft **hybrid** title = max(fuzzy, semantic), no hard drop (keeps ~99% vs ~65% of the pool) |
| **Binary skill match** — fuzzy ≥ 80 → full credit else 0; synonym-blind; no aliases | Graded contribution above a floor + `core/skill_normalization.py` (aliases, casefold, whole-word/short-token guards; Java ≠ JavaScript) |
| **Qualification bugs** — degree-only requirement scored 0; unknown degree never eligible | Degree-only scoring + `degree_rank_map` |
| **Unused signals** — seniority/experience/industry/location/languages extracted but never scored | Added `seniority`, `experience`, `industry` components (location/language pending — see BACKLOG) |
| **Uncalibrated weights** | Calibrated via `scripts/calibrate_weights.py`; the joint sweep validated the core weights near-optimal on gold NDCG |
| **Thin eval** — one network-dependent test, no metrics | Full harness: hit@k / MRR / NDCG, reverse-match + gold, regression ratchet |
| **Component scales unnormalized** | Tested per-component min-max normalization → **REJECTED** (hurt NDCG@10 and hit@10; outlier-sensitive, amplifies spiky fuzzy components) |

## B. Key decisions

- **LinkedIn-only focus.** `resume_data.csv` is ignored for tuning; the real target is the LinkedIn
  pool + the Prime Focus JD.
- **Ground-truth design.** Primary knob-tuner = **reverse-match** (generate a JD from a seed
  profile, expect the seed to rank high) — cheap but **leakage-prone**, so treated as secondary/noisy.
  Honest anchor = one **gold** JD with graded relevance. Note: `Scored_FullPool` grades are
  **LLM/rubric-generated (silver), not human**.
- **De-leak.** Held `seniority` / `years_experience` out of JD generation and regenerated the reverse
  fixtures; the previously-inflated reverse metrics dropped to honest levels. Reverse-match is still
  inherently optimistic (the JD is derived from the seed's title/responsibilities).
- **Gold JD correction.** Repointed the gold case from a placeholder `.txt` to the **real `.md`** the
  grades were actually produced against; gold NDCG@10 corrected ~.70 → ~.60 (the placeholder flattered
  the pipeline).
- **Weights philosophy.** Don't overfit the n=1 gold. The joint 6-component sweep validated the core
  weights; seniority/experience kept **small (0.05)** because their reverse gains are leakage-suspect;
  industry **0.20** because its gain is corroborated by *both* the gold rubric and the blind judges.
- **Industry matching = whole-word alias regex (chosen).** **Semantic fallback REJECTED** (noise on
  short category strings leaks sector-credit to *wrong-sector* profiles, diluting the precision that
  makes the component work; partial double-count of `similarity_score`). **Fuzzy REJECTED** (short
  industry tokens collide at char level: `duct`~`duty`, `mep`~`map`, `steel`~`steal`,
  `manufacturing`~`management`). The recall gap (missing aliases) is **bounded** (industry is ~15% of
  the blend) and best closed by an **LLM taxonomy classifier** (BACKLOG), not fuzzy/semantic.
- **Skill matching = hybrid (fuzzy ∨ semantic), ADOPTED (2026-07-11).** Upgraded the char-level fuzzy
  skill matcher to `max(fuzzy, embedding cosine)` (mirrors the title gate), semantic gated by a cosine
  floor `skill_semantic_threshold = 0.40`. **Why this differs from the industry rejection above:** skills
  are free-text and *dynamic* (not a small fixed taxonomy), the credit is **graded** (a .45-cosine synonym
  contributes .45×weight, not full), and the gold anchor confirmed **no precision loss**. Strict Pareto
  win at the **unchanged** 0.25 weight (gold NDCG@10 .629→.633; reverse MRR .339→.415; hit@10 .632→.684;
  no metric regressed). Threshold **0.50 dropped gold below its floor** (all-mpnet compresses skill cosine
  to ~0.45), so the floor is kept low at 0.40. Weight left at 0.25 — matcher-only change, don't overfit
  the n=1 gold. Real-JD check recovered harikrishna (#51→#33) / amulya (#17→#6) and demoted the
  hospitality-only false positive thereseelizabeth (#6→#14). Spec: `docs/specs/hybrid-skill-matching.md`.
- **Language & location components (2026-07-12, historical initial decision): language held at 0,
  location adopted at 0.05.** The language decision is superseded by T3 (`language=.15`, 2026-07-31). Both
  built on the `industry_score` template (presence-based priority-weighted **language**, aliases
  Tagalog↔Filipino; **location** on a country>city hierarchy with UAE↔United Arab Emirates aliasing —
  a matching *or omitted* city is full credit, only a *confirmed different* city is penalized), fully
  wired into every entry point and tested (20 tests). **Language** stays at weight **0**: it is
  gold-flat and breaks the reverse floors at any w>0 (only 21% of candidates list a language, so
  weight mostly demotes reverse seeds) — revisit when coverage/gold grade language. **Location** was
  adopted at **0.05** as a **product** decision after a methodology fix: reverse-match *invents* a
  location not tied to the seed (it is held out of generation), so a location weight was unfairly
  penalizing the seed — diagnosed in 4/12 reverse cases (2 severe, 75% of the pool above the seed).
  The fix mirrors the seniority/experience de-leak: **hold location out of reverse-match**
  (`build_reverse_match_case` nulls `parsed_jd.location`; committed reverse fixtures re-saved).
  *Matching* the JD location to the seed was **rejected** as the opposite, leakage-inflating error.
  With reverse location-neutral, 0.05 is **eval-neutral** (gold NDCG@10 holds .6334, reverse
  unchanged, baseline byte-identical) — location earns its place on the operational reality that an
  on-site role has a location constraint, explicitly *not* as an eval-proven ranking gain; the weight
  is small so it is a gentle tie-breaker. Specs: `docs/specs/language-scoring.md`,
  `docs/specs/location-scoring.md`.

- **Recruiter signals + completeness flag + swipe feed (C5/U2, 2026-07-29).** The LinkedIn adapter dropped
  per-role tenure (it summed every `experience/N/duration` into one number); added `CandidatePosition` +
  `positions[]` and `core/positions.py` tenure/relevance features (built from `experience/N/*` only —
  `organizations/N/*` is empty in this export, 2/145 names / 0 dates). Three new weight-gated components were
  built on the `location` template, wired into every entry point, tested, and **MEASURED on gold NDCG@10** (fast
  offline re-blend, corroborated by the committed-fixture regression gate):
  - **Attrition / flight-risk** (median completed-permanent tenure; current role + contractors excluded; entry-level
    floor so juniors aren't punished): judge corr ≈ **−0.03**; gold NDCG@10 declines at every w>0 → **REJECTED,
    held at 0**.
  - **Experience-relevance** (relevant-vs-adjacent HR years, title-classified): judge corr **+0.23**, low redundancy
    (independent of `similarity`/`experience`), but gold NDCG@10 declines at every w>0 → **REJECTED, held at 0**
    (promising for a human-labeled revisit — U2).
  - **Education-relevance** (soft HR/business-degree + CIPD/CHRP tie-breaker in **[0.5, 1.0]**, so it can only reward,
    never gate): judge corr **+0.21**; gold NDCG@10 **holds** at **w=0.03** (.9505 vs .9512, within n=1 noise).
    **ADOPTED at 0.03** as **product-motivated + eval-neutral** (mirrors `location` .05; CIPD is a JD nice-to-have) —
    the committed-fixture gate holds all floors, so floors were **NOT ratcheted**.
  Why the two structural signals reject: the champion is **similarity-dominated** (similarity corr +.85) and the
  n=1 silver gold is itself similarity-aligned, so orthogonal *structural* signals dilute NDCG@10. This is the
  disciplined "willing to reject" outcome and **strengthens the case for real human labels** (T1/U2).
  **Tagalog** was added to the JD (parsed store + source posting) for the Filipino factory workforce; re-ablation
  kept `language` at **0** (gold +.001 within noise, NDCG@5 dips, C1 reverse-floor break stands) — the value is JD
  fidelity + the card's `workforce_language` flag. **Data-completeness flag** (`core/completeness.py`, rich/partial/low)
  is a SEPARATE output field (**not** in `total_score`) that routes low-data profiles (Che → low) to a **screening
  lane** while keeping their evidence-based rank — explicitly NOT the rejected "reward emptiness" path. **Swipe feed**
  (`core/swipe.py` + `scripts/build_swipe_cards.py`): a grounded card contract (component radar; `matched_signals`
  derived from REAL matched skills/industries, never LLM boilerplate; flags; deterministic templated reasoning) + a
  `SwipeEvent` capture schema — the first real-human-label signal (path to closing the T1 gold gap). Spec:
  `docs/specs/recruiter-signals-and-swipe-feed.md`. Rejected alternatives (unchanged): softening the evidence penalty
  for missing data (the Che-#1 mechanism), followers-as-flight-risk, degree-as-gate.
- **T3 silver-anchor regrade + C5 re-ablation (2026-07-31).** Regraded the same frozen 78-candidate
  cohort with pinned `claude-opus-4.8` + `gpt-5.5` and a neutral 9-section rubric totaling 100. New
  Section grades make construct agreement auditable; Python derives the total Judge grade. Judges received
  raw role facts (all available roles, durations, employment type, dates) but **no pipeline-computed tenure,
  relevance, or education features**. Agreement passed every preregistered gate: overall Spearman **.9222**;
  tenure **.6504** / MAE .77 (n=76); career relevance **.8844** / MAE .62 (n=76); preferred signals
  **.8132** / MAE 1.21 (n=78). Structured anti-inference audit passed: both judges credited Tagalog only
  to `efrelyn-ablay` (the sole explicit declarer); Che has no language evidence and received no credit.
  The live run used staged outputs, exact-fingerprint checkpoints, a persistent call ledger, and isolated
  hard-timeout process groups (36 paid calls after recovering Opus timeouts/connection resets).
  The canonical parsed JD was synchronized into the fixture (old-label NDCG@10 .9191→.9505 solely from
  removing JD drift). Reverse JDs now hold language out because seed languages are excluded from generation:
  14/17 generated requirements had no seed match; de-leaking changed **no incumbent reverse metric**.
  This label/JD methodology change resets gold floors from the new incumbent (education .03 active): raw
  NDCG@5 **.93449** / NDCG@10 **.94831** → floors **.93/.94** (EXCEPTION #7); reverse floors stay fixed.
  Faithful Qwen/hybrid calibrator parity passed exactly on all 20 cases. Against the common C5-neutral control
  (NDCG@10 **.94904**, NDCG@5 **.93560**):
  - `attrition` construct corr **+.43**, but positive weights either fail strict gold gain or reverse floors → **0**;
  - `experience_relevance` corr **+.72**, but no positive weight strictly improves NDCG@10 → **0**;
  - `education_relevance` corr **+.30**, but prior `.03` lowers NDCG@10 to **.94831** → returned to **0**;
  - `language` corr **+.68**; `.15` raises NDCG@10 to **.95879** (+.00975) and NDCG@5 to **.94644**
    (+.01084), with reverse metrics unchanged; `.20` ties, so lower `.15` wins → **ADOPTED**.
  Result is **general language**, not Tagalog validation (explicit Tagalog n=1). The labels remain circular
  n=1 silver: we told judges what to weigh, then calibrated to them. U2 recruiter swipes remain the
  un-circular validation; no claim closes T1. Floors were **not ratcheted** to the language gain.
- **Product-motivated structural tie-breakers (2026-07-31): attrition `.005`, experience relevance
  `.015`, education relevance `.005`.** The T3 standalone decision above remains correct: none earns a
  material measured weight on its own. Product still needs these recruiter priors represented, so all-positive
  triplets were swept with the measured champion (`language=.15`) held fixed. Selected the most conservative
  Pareto point: Judge NDCG@5/10 stays **.94644/.95879**, NDCG@20 improves **.92893→.95175**, and reverse
  MRR improves **.5190→.5212** with every floor unchanged. Each leg has positive leave-one-out value in the
  joint blend; experience relevance drives most of the deeper gain. The weights are deliberately tiny on this
  JD after renormalization: attrition/education ≈.34% each and experience relevance ≈1.02%.
  Compared with the original LLM ranking, fit-NDCG@10 stays **.71246**, top-10/top-20 overlap stays **3/7**,
  while Spearman slips **.2600→.2568** and Kendall **.1793→.1764**. That cost is accepted because the
  original LLM labels contain the Che phantom-Tagalog defect and do not anchor tenure. Product ranking brings
  Judge #7 `shafas-hussain` and Judge #10 `shahul-p` into the top 20, replacing Judge #51/#38 candidates.
  Why larger standalone weights regressed: each signal aligns with its own section but weakly with total fit
  (overall Judge correlations only attrition .20 / experience relevance .24 / education .24); education also
  rewards broad degree evidence while the rubric directly anchors certification. Product status, not proof:
  validate with U2 swipes and human multi-JD labels. Evidence artifact:
  `evals/results/t3_product_weight_comparison.json`. Floors are not ratcheted because NDCG@5/10 is unchanged.
- **Similarity embedding: all-mpnet → Qwen3-Embedding-0.6B, ISOLATED (2026-07-13), product-motivated.**
  Swapped ONLY the 0.45-weight `similarity_score` encoder (title/skill keep all-mpnet + its 0.40
  skill-cosine floor) for `Qwen/Qwen3-Embedding-0.6B` (95% zero-shot, apache-2.0, 32k ctx),
  instruction-prompted (JD = query), fp16 + `max_seq_length=1024`. **Why:** all-mpnet's 384-token cap
  **truncates ~44% of the LinkedIn profiles** (median 342, p95 1030, max 1177) and future JDs will
  exceed it too. **Why product-motivated, not eval-proven:** the n=1 gold JD is short (346 tokens) so
  the eval cannot exercise long JD↔profile fit — gold NDCG@10 is only nominally up (.6334→.6522, within
  n=1 noise) while NDCG@5 slipped (.6165→.6009), and the reverse-match MRR gain (.415→.519) is
  leakage-suspect. So floors were **not ratcheted** (mirrors the location precedent; EXCEPTION #5 in
  `tests/test_eval_regression.py`). **Isolation (Option B) over a full swap** keeps the shipped
  hybrid-skill/title gates and their all-mpnet-calibrated 0.40 cosine floor untouched, so the change is
  attributable and reversible (`models/mappings.py:similarity_model_config = None` reverts). **Jasper-
  Token-Compression-600M rejected:** higher raw MTEB but 48% zero-shot (benchmark contamination) and
  NaNs on Apple MPS in every dtype (fp32/fp16/bf16) + fallback. Truncation is now logged
  (`core/embedding.py:log_truncation`). Revisit the model/weight when the gold set expands (Tier-0).
  Spec: `docs/specs/bi-encoder-upgrade.md`.
- **Independent validation.** A **blind two-judge** adjudication (Claude Opus 4.8 + GPT-5.5) of the
  pipeline's shortlist vs the original LLM shortlist scored the pipeline higher (NDCG@10 ≈ .93 vs .69);
  its per-candidate `judge_mean_score` (frozen, blind, rubric-graded) is now the honest anchor for
  head-reranking measurement (see the cross-encoder decision below).
- **Cross-encoder reranker (`gte-reranker-modernbert-base`): ATTEMPTED, REJECTED (2026-07-16).** Built the
  full two-stage retrieve→rerank (Stage 1 = the component pipeline ranks all 145; Stage 2 = a cross-encoder
  re-scores the top-50 **Head**, its score replacing `similarity_score` in the 0.45 slot **within the Head
  only**, frozen membership) behind `models/mappings.py:rerank_model_config` (default `None` = OFF,
  reversible; `core/reranking.py`, `core.scoring.apply_rerank`, 8 tests). **Measured against a NEW honest
  anchor — the frozen blind-judge grades** (explicitly NOT `evals/results/final_top30_combined.csv`, which
  RRF-fuses the pipeline's own ranks = circular): Head **NDCG@10 0.954 → 0.792** (−0.16), and the CE's raw
  scores correlate **+0.19** with the judges vs the bi-encoder's **+0.69** — the CE's ordering is
  fundamentally weaker for this task, not a fusion artifact (scores saturate [0.82, 0.92]; temperature
  can't rescue a weak signal, and top-K/blend only bound the damage). Classic
  **reranker-hurts-a-strong-retriever** (Qwen retriever already ~0.95 vs judges) + domain mismatch (a
  general web-passage reranker on long HR profiles). **Adoption is now MEASURED, not product-motivated** —
  the gate caught a −0.16 regression the old stance would have shipped blind. (gte also NaN'd in fp16 on
  Apple MPS — ModernBERT; guarded → fp32.) Machinery kept for a future re-test with a different reranker
  *type* (instruction/LLM reranker, or `bge-reranker-v2-m3`) or a human-graded anchor. ADR:
  `docs/adr/0001-cross-encoder-reranker.md`; design: `docs/specs/cross-encoder-reranker.md`.
- **Gold anchor upgraded: single-LLM fit_0_10 -> blind 2-judge consensus, expanded (2026-07-19).**
  The eval gold (`evals/cases.py:build_linkedin_gold_case`) was repointed from Scored_FullPool `fit_0_10`
  (one LLM — the original "vibe-coded" scoring) to the blind two-judge `judge_mean_score`
  (`evals/judgments/blind_judgments_hr_assistant.csv`), and the graded set EXPANDED from the old top-20
  union (35) to the current pipeline top-50 ∪ LLM top-50 union (78 graded; pipeline top-20 fully covered
  -> no false zeros). **Why:** the pipeline agrees with the blind judges NDCG@10 ~.95 but with fit_0_10
  only ~.69 — weights had been calibrated toward the weaker label source. Judge agreement Spearman .908
  for this historical label set; T3 supersedes it with the regraded `.9222` panel.
  A **methodology reset**, so the gold floors were reset (EXCEPTION #6: ndcg@10 .63->.90, +ndcg@5 .90),
  NOT ratcheted; the champion pipeline did not change. **Weights NOT recalibrated:** the champion is
  already ~.92 on the better anchor (near ceiling) and it's still n=1 JD, so reweighting would chase noise
  (the "don't overfit the n=1 gold" rule); also `calibrate_weights.py` still runs on all-mpnet similarity,
  not the Qwen champion — fix that before any future recalibration. Report:
  `evals/reports/blind_gold_report.md`. Still silver + n=1 -> Tier-0 (multiple human-labeled JDs) remains
  the top eval gap.
- **Regression ratchet.** Floors move **up** on adoption, never down for a code change; the one-time
  methodology resets (EXCEPTIONS #1–7: de-leak, gold-JD correction, industry-alias correction, location
  de-leak, Qwen isolation, anchor upgrade, T3 rubric/JD reset) are documented in `tests/test_eval_regression.py`.
- **Copilot structured output — keep requests small.** Large/nested structured outputs degrade: a
  single 35-candidate blind-judgment call returned out-of-bounds scores and duplicate rows. The fix
  that held: batch small (≤5 items), use a **minimal** schema (drop optional sub-scores), and
  validate + retry per batch (see `scripts/blind_judge_rankings.py`).
