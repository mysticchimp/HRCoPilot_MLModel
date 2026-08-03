# Backlog — AI Recruiter Pipeline

Prioritized open work. Each item: **what / why / status**. See `docs/DECISIONS.md` for
settled calls, `docs/specs/` for shipped-feature specs, `ARCHITECTURE.md` for design detail, and
**`AGENTS.md` → "How to test / ablate a new change"** for the step-by-step adopt loop every item
below should follow. This is the single source of truth for "what's next" — update it as items ship.

> **IDs:** items carry a stable section prefix for easy reference — **T**ier 0 · **M**odeling ·
> **C**omponents · **U** infra/UX · **S**hipped (e.g. `C2` = location component).

## Tier 0 — eval integrity (highest leverage)

### T1 · Expand the gold set with HUMAN labels
- **What:** add several real JDs with **human**-graded candidate relevance.
- **Why:** the honest anchor is currently ONE silver (LLM/rubric-graded) JD (n=1) — the binding
  constraint on trusting further tuning and the prerequisite for an LLM re-ranker. Every accuracy
  claim rests on it.
- **How:** collect 3–5 JDs + human relevance; add as gold cases in `evals/`; re-run the joint
  sweep NDCG-led across the set.
- **Status:** parked (needs human input). **Interim (2026-07-31):** the frozen 78-candidate cohort was
  regraded by Opus 4.8 + GPT-5.5 with a 9-section tenure/relevance/language-aware rubric. Overall agreement
  Spearman **.922**; tenure **.650**; career relevance **.884**; preferred signals **.813**. The adopted
  champion reaches NDCG@10 **.9588** / NDCG@5 **.9464** (EXCEPTION #7). This does **not** close T1 —
  still LLM labels on **one** JD; the human, multi-JD set remains the goal. See
  `evals/reports/blind_gold_report.md`, `docs/DECISIONS.md`.

### T2 · Validate the live location weight (fair location eval)
- **What:** get an eval that can actually *reward* location fit — a **location-diverse** candidate
  pool (multi-country JDs where country-level fit discriminates) and/or a **location-graded** gold
  case — then re-ablate `location_score` and tune the weight on evidence.
- **Why:** `location_score` is **live at 0.05 but PRODUCT-motivated, not eval-validated** — this pool
  is 144/145 UAE and the n=1 silver gold doesn't grade location, so no honest metric can currently
  confirm or tune it (reverse-match was de-leaked because it invents a seed-mismatched location). Same
  shape as the gold-set gap above; a location-graded gold would resolve both.
- **Status:** parked (needs a location-diverse pool or location-graded labels).

### T3 · Tenure/relevance/language-aware re-grade of the silver Judge-grade anchor
- **What:** expand the blind-judge rubric (`scripts/blind_judge_rankings.py`) with **tenure & continuity**,
  **relevant-vs-adjacent career mix**, and **Tagalog/workforce-language** sections (rebalanced to 100),
  re-run the live 2-judge grade, rebuild the gold labels, and re-ablate the C5 components on the new anchor.
- **Why:** the current gold *and* the `fit_0_10` source rubric **never score tenure** and only partially
  score role-relevance/Tagalog — so `attrition` (judge corr −0.03) and `experience_relevance` were tested
  against anchors blind to them (an eval blind spot, not necessarily a dead component). Makes the anchor
  *fair* to the C5 signals; does **not** close T1 (still n=1 silver, and re-grading is circular — we steer
  the judge then calibrate to it).
- **How:** design + neutral rubric wording + circularity/floors-reset caveats in
  `docs/specs/anchor-regrade-and-compliance-component.md` §3. Live LLM (Opus 4.8 + GPT-5.5); floors **RESET**
  (methodology change), not ratcheted. Guardrail: **credit Tagalog only on an explicit declaration, never a
  name/nationality inference** (the `fit_0_10` Che defect — its #1 got phantom Tagalog credit from her name).
- **Status:** ✅ **SHIPPED (2026-07-31).** Frozen 78-candidate cohort regraded by both pinned judges;
  all agreement gates pass. Explicit Tagalog credit went only to `efrelyn-ablay` (the sole declarer), never
  Che. Canonical JD/fixture drift fixed; reverse language requirements de-leaked with incumbent reverse
  metrics unchanged. Floors reset to `.93/.94` from the new incumbent (EXCEPTION #7), not ratcheted.

## Modeling upgrades (ranking quality)

### M1 · Cross-encoder re-ranker (retrieve → re-rank) — recommended first
- **What:** re-score the top-K (~30–50) with a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
  over (JD, candidate) pairs; blend with / replace `similarity_score`.
- **Why:** the 0.45-weight bi-encoder embeds JD and candidate independently and misses
  "this candidate for this JD" nuance — the source of the biggest misses (e.g. Harikrishna, now
  partially recovered by hybrid skills, rank 51→33, but still buried below its blind-consensus #1).
  No circularity, tractable at K≤50.
- **Status:** ⛔ **ATTEMPTED → REJECTED (2026-07-16, `gte-reranker-modernbert-base`).** The two-stage
  retrieve→rerank machinery is **shipped behind a flag** (`models/mappings.py:rerank_model_config`,
  default `None` = OFF; `core/reranking.py` + `core.scoring.apply_rerank`, frozen-membership top-50 Head,
  8 tests) but the model **failed the measured-adoption gate**: Head NDCG@10 vs the **frozen blind-judge
  grades** dropped **0.954 → 0.792**, and the CE's raw scores correlate **+0.19** with the judges vs the
  bi-encoder's **+0.69** (reranker-hurts-a-strong-retriever + domain mismatch; a general web-passage
  reranker on long HR profiles). Fully reversible; revisit with a different reranker *type*
  (instruction/LLM reranker, or `bge-reranker-v2-m3`) or a human-labeled anchor. Decision:
  `docs/adr/0001-cross-encoder-reranker.md`; design: `docs/specs/cross-encoder-reranker.md`.

### M2 · Semantic skill matching — ✅ SHIPPED → see `## Shipped`

### M3 · Re-run blind adjudication under hybrid skills
- **What:** re-run `scripts/blind_judge_rankings.py` for an updated independent read now that the
  skill signal is live (unset `COPILOT_SKIP_CLI_DOWNLOAD`).
- **Why:** `evals/blind_comparison_report.md` predates hybrid skills — its pipeline ranks are stale,
  and its component-diagnosis table specifically flagged the (then) dead skill score.
- **Status:** not started (live LLM).

### M4 · LLM re-ranker (listwise, top-K) — LAST
- **What:** an LLM ranks the top ~20 given the JD.
- **Blocker:** **circularity** — cannot be validated against LLM-generated labels, so it needs the
  human gold set (Tier 0) first. Also cost/latency/nondeterminism.
- **Status:** blocked on Tier 0.

## Components (optional, weight-gated; ablate-then-adopt)

### C1 · Language component — ✅ ADOPTED at weight 0.15 (T3) → see `## Shipped`
- candidate `languages` → **exact/normalized** presence match, priority-weighted; aliasing
  (Tagalog↔Filipino). Initially held at 0; T3 added a fair language-aware silver anchor and removed
  ungrounded reverse requirements. Adopted at `.15` on a strict NDCG@10 gain with all floors holding.
  **Follow-up:** human/multi-JD validation; Tagalog-specific evidence remains n=1.

### C2 · Location component — ✅ ADOPTED at weight 0.05 (product-motivated) → see `## Shipped`
- `CandidateLocation` country>city hierarchy (matching *or omitted* city = full credit; confirmed
  different city partial), UAE↔United Arab Emirates aliasing. Adopted at **0.05** after de-leaking
  location from reverse-match (it invents a seed-mismatched location); eval-neutral at 0.05 (gold
  holds .6334, reverse unchanged). **Follow-up:** revisit the weight upward once the eval can *reward*
  location (location-diverse pool / location-graded gold).

### C3 · Industry: taxonomy classifier (scale fix)
- Replace the hand-maintained `INDUSTRY_ALIASES` with **LLM classification into a fixed taxonomy**
  at ingestion (cached per candidate); match by taxonomy (hierarchical). Trigger: a 2nd client in a
  different industry. (Semantic & fuzzy fallbacks were evaluated and **rejected** — see DECISIONS.)
  Not started.

### C4 · Essential-skill emphasis / must-have gate (mostly covered by skill_score — de-prioritized)
- **Finding:** UAE compliance (WPS/MOHRE/visa/PRO) is the heaviest human signal, but the JD extractor emits it
  as `priority=="essential"` skills/technologies and `skill_score` ALREADY scores it priority-weighted
  (essential=1.0) inside a joint-sweep-validated 0.25 weight — so the pipeline is NOT under-weighting it (the
  "gap" was largely illusory).
- **Honest lever (if more emphasis is wanted):** raise `skill_score`'s weight or
  `attribute_weight_by_importance[ESSENTIAL]` and re-run the joint sweep — NOT a new component (which would
  double-count `skill_score` + hand-set an emphasis the sweep didn't support).
- **Only reason to build a component:** a hard must-have GATE (missing an essential ⇒ disqualify) — a
  non-linearity `skill_score` can't express. Expected REDUNDANT on this pre-filtered HR pool; if built, adopt
  only if `--redundancy essential_gate_score skill_score` shows the gate beats the graded blend on gold NDCG@10.
- **How:** `docs/specs/anchor-regrade-and-compliance-component.md` §4. **Status:** 📝 analyzed →
  **de-prioritized** (the "gap" is largely already covered; T3 re-grade is the higher-value eval work).

### C5 · Recruiter signals: tenure/attrition + relevant-experience + education
- **What:** three new weight-gated components — **tenure/attrition** (flight-risk from years-per-company,
  not followers), **relevant-vs-adjacent experience** (real HR years, not total), and **education
  relevance** (soft HR/business-degree + CIPD/CHRP bonus, never a gate) — plus adding **Tagalog** to the
  JD for the Filipino factory workforce and re-measuring `language_score`.
- **Why:** the pipeline drops the signals a recruiter actually uses; the LinkedIn adapter even sums
  per-role durations into one number and **discards tenure** though the raw export carries it. Surfaced
  by the Che / hollow-bullseye / 6-candidate analysis.
- **How:** implementation-ready spec → `docs/specs/recruiter-signals-and-swipe-feed.md` (§4–7 designs,
  §10 adopt-loop). Each component: correlate w/ judges → `--ablate` gold NDCG@10 → `--redundancy` →
  adopt only if floors hold. Add per-role `positions[]` to the adapter first (P1).
- **Status:** ✅ **SHIPPED + RE-ABLATED (2026-07-31).** On the T3 anchor, construct directions are positive
  (`attrition` .43, `experience_relevance` .72, `education` .30, `language` .68), but attrition and
  experience-relevance reject as standalone measured weights, and education .03 did not re-earn weight.
  **Language ADOPTED at .15**. A subsequent explicit product override adds tiny structural tie-breakers:
  attrition `.005`, experience relevance `.015`, education `.005`. Joint NDCG@5/10 holds, NDCG@20
  .9289→.9518, reverse MRR .5190→.5212; original-LLM Spearman slips .0032. General-language result only;
  Tagalog-specific n=1. See `docs/DECISIONS.md` §C5.

## Infra / UX

### U1 · Interactive JD extraction (human-in-the-loop)
- **What:** make JD extraction a back-and-forth Q&A with the recruiter to confirm extracted
  features **and the priority** of each requirement, before any scoring.
- **Why:** JD extraction is the entry point; input quality caps the entire pipeline.
- **Status:** noted; high-impact UX lever.

### U2 · Recruiter swipe feed + low-data flag
- **What:** present scored candidates as a **Tinder-style card feed** (component radar, grounded
  matched-signals, flags, reasoning) the recruiter swipes; emit a **data-completeness flag** that routes
  sparse profiles (e.g. Che) to **screening** instead of discarding them (treat as potentially hire-able).
- **Why:** turns the ranking into fast triage **and** captures swipes as the first **real human labels** —
  the path to closing the **T1** gold-set gap (gold is currently LLM-judged, n=1).
- **How:** card JSON contract + swipe-capture schema in `docs/specs/recruiter-signals-and-swipe-feed.md`
  (§8–9). The flag is a separate field, **not** folded into `total_score`.
- **Status:** ✅ **Backend SHIPPED (2026-07-29).** Data-completeness flag (`core/completeness.py`, rich/partial/low,
  NOT in `total_score`) routes low-data profiles (Che → low) to a screening lane; grounded swipe-card contract +
  `SwipeEvent` capture schema (`core/swipe.py`, `scripts/build_swipe_cards.py` → 145 cards, 25 `screen_me`).
  **Front-end still out of scope.** See `docs/DECISIONS.md` §C5.

## Shipped

### S3 · Similarity bi-encoder upgrade (all-mpnet → Qwen3-Embedding-0.6B, isolated) — ✅ 2026-07-13
Swapped ONLY the 0.45-weight `similarity_score` encoder to `Qwen/Qwen3-Embedding-0.6B` (95% zero-shot,
apache-2.0, instruction-prompted, fp16, `max_seq_length=1024`); title/skill keep all-mpnet.
**Product-motivated:** all-mpnet truncates ~44% of profiles at its 384-token cap (median 342, p95 1030,
max 1177) and future JDs will exceed it. Adopted eval-cautiously — gold NDCG@10 .6334→.6522 is nominal
(n=1 noise), NDCG@5 −.016, and the reverse-MRR gain is leakage-suspect — so floors were NOT ratcheted
(EXCEPTION #5). Isolated (Option B) behind `models/mappings.py:similarity_model_config` (set `None` to
revert). Shipped alongside: a model-safe embedding cache, a NaN guard, per-model instruction support,
and **truncation logging** (`core/embedding.py:log_truncation`). Jasper-600M rejected (48% zero-shot
contamination + NaNs on Apple MPS in every dtype).
- **Decision:** `docs/DECISIONS.md` · **Spec:** `docs/specs/bi-encoder-upgrade.md` · **Config:**
  `models/mappings.py` · **Floors:** EXCEPTION #5 in `tests/test_eval_regression.py`
- **Open follow-up (Tier-0):** the n=1 short gold JD (346 tokens) can't measure the long-context gain —
  revisit the model/weight (and whether to extend Qwen to title/skill) once the gold set expands with
  longer, human-labeled JDs.

### S1 · Language + location components — ✅ 2026-07-12 initial decision (language later adopted by T3)
Two new optional, weight-gated components (`language_score`, `location_score`) built on the
`industry_score` template: presence-based priority-weighted language matching (aliases:
Tagalog↔Filipino) and country>city-hierarchy location matching (UAE aliasing; matching *or omitted*
city = full credit, confirmed different city = partial). Fully wired into every entry point +
`calibrate_weights.py`, with 20 unit tests. **Initial language decision:** held at 0 (gold-flat +
reverse-floor-breaking; 21% coverage), superseded by T3 on 2026-07-31. **Location adopted at 0.05
(product-motivated):** reverse-match invents a
seed-mismatched location, so it was **de-leaked** (held out of reverse — `build_reverse_match_case`
nulls it, committed reverse fixtures re-saved), mirroring the seniority/experience de-leak; at 0.05
the adoption is **eval-neutral** (gold NDCG@10 holds .6334, reverse unchanged, baseline byte-identical)
and reflects the real location constraint of an on-site role, not an eval-proven gain.
- **Specs:** `docs/specs/language-scoring.md`, `docs/specs/location-scoring.md` · **Decision:**
  `docs/DECISIONS.md` · **Tests:** `tests/test_language_score.py`, `tests/test_location_score.py`
- **Open follow-ups (tracked above):** revisit language weight when coverage/gold expand (Tier 0);
  revisit location weight upward on a location-diverse / location-graded eval.

### S2 · Hybrid skill matching — ✅ 2026-07-11
Fuzzy → hybrid skill matcher (per-channel **gated `max(fuzzy, semantic cosine)`**, semantic floor
`skill_semantic_threshold = 0.40`) — a strict Pareto win at the **unchanged** 0.25 skill weight
(gold NDCG@10 .629→.633, reverse MRR .339→.415, hit@10 .632→.684; no metric regressed). Recovered
skill-starved strong candidates on the real JD (harikrishna #51→#33, amulya #17→#6) and demoted a
surface-match false positive (thereseelizabeth #6→#14).
- **Spec:** `docs/specs/hybrid-skill-matching.md` · **Decision:** `docs/DECISIONS.md` · **Floors
  ratcheted:** `tests/test_eval_regression.py`
- **Open follow-ups (tracked above):** *Re-run blind adjudication under hybrid skills*; revisit
  `skill_semantic_threshold` / skill weight when the gold set expands (Tier 0).
