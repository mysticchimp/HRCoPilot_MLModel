# Spec — Recruiter Signals (tenure/attrition, relevant-experience, education, workforce-language) + Low-Data Flag + Swipe Feed

| | |
|---|---|
| **Status** | ✅ **Implemented + T3 re-ablated + product override (2026-07-31).** Language `.15` measured; attrition `.005`, experience relevance `.015`, education `.005` are product-motivated tie-breakers. See `docs/DECISIONS.md` §C5. |
| **Type** | 3 new scoring components + 1 JD/config change + 1 output flag + 1 product surface |
| **Depends on** | The ablate-then-adopt loop in `AGENTS.md`; existing components in `core/scoring.py`; `models/mappings.py` weights |
| **Related decisions** | `docs/DECISIONS.md`; prior component specs `docs/specs/location-scoring.md`, `docs/specs/language-scoring.md` |
| **Rejected alternatives** | "penalize less for missing data" (reintroduces the Che error); followers-as-flight-risk (noisy); degree-as-gate (JD requires no degree) — see §11 |

---

## 1. Summary

Add the recruiter signals the pipeline currently drops, so the ranking captures **how a human recruiter actually reasons** — not just title/skill/similarity. Concretely:

1. **Tenure & attrition (flight-risk)** — reward stable job history, penalize chronic short-tenure. Based on **years-per-company**, *not* follower count.
2. **Relevant / adjacent experience** — distinguish *real HR years* from total years (the "HR-Operations vs process" question), giving partial credit for adjacent roles (admin, PRO, ops).
3. **Education relevance** — a *soft* tie-breaker rewarding HR/business-relevant degrees & certs (CIPD/CHRP). Never a gate (the JD requires no degree).
4. **Workforce-language (Tagalog/Filipino)** — the role supports a Filipino factory workforce; add Tagalog to the JD and re-measure `language_score`.
5. **Low-data flag** — candidates with sparse profiles (e.g. Che) are **flagged for screening, treated as potentially hire-able**, *not* disregarded and *not* rescued by down-weighting evidence.
6. **Swipe feed** — present scored candidates as a **Tinder-style card feed** the recruiter swipes; capture swipes as the **first real human-label signal** (which also fixes the "gold is LLM-judged" gap).

Every scoring change goes through the **ablate-then-adopt loop** (correlate with judges → ablate on gold NDCG@10 → redundancy → joint → adopt only if floors hold → ratchet). Weights are **measured, never hand-set**.

## 2. Motivation & background (what this session established)

- **Che Ibardelosa** (`che-ibardelosa-a538072a1`): original LLM ranked her **#1**, pipeline **#132**, blind judges **#67/78**. Her profile is an *empty skeleton* — bullseye title, Dubai, entry, but **no about, no responsibilities, 4 generic non-HR skills**. The LLM ranked her #1 by **hallucinating** signals (its `matched_signals` are boilerplate copy-pasted across ranks 1–18 of the 2673-pool shortlist — asserted even for a CCTV/typist). The pipeline + judges correctly discount her.
- **Hollow-bullseye cohort:** 34 candidates score high on title+seniority+experience+location but low on skill+similarity. The judges **agree** they're weak (cohort mean judge score 39 vs pool 54). So burying them is correct — *not* a calibration bug.
- **Calibration check:** over the 78 judged, `similarity` correlates **+0.86** with judge scores and `skill` **+0.52** (the two heaviest weights — validated); `title` **+0.02** and `seniority` **−0.08** (range-restricted in a pre-filtered HR pool — they gate, they don't differentiate). So do **not** shift weight toward the surface components.
- **The gap:** a recruiter's real reasoning (from the 6-candidate walk-through in §13) uses **tenure, current-role duration, job-hopping, relevant-vs-adjacent career mix, degree relevance, and workforce-language** — none of which the pipeline models.
- **Concrete data finding:** the LinkedIn adapter's `_years_experience` ([core/adapters/linkedin_adapter.py](../../core/adapters/linkedin_adapter.py#L146-L149)) **sums** every `experience/N/duration` into one number and **discards the per-role structure**, even though the raw export carries per-role `duration`, `organizations/N/startDate`, and `organizations/N/endDate`. **Tenure is in the data; we throw it away.**
- **Caveat that motivates the swipe feed:** the "gold" is a 2-LLM-judge consensus (Claude Opus 4.8 + GPT-5.5), **not human** — a silver anchor, n=1. Recruiter swipes are the path to real human labels.

## 3. Goals / Non-goals

**Goals**
- Model tenure/attrition, relevant-experience, and education as measured, weight-gated components.
- Add Tagalog to the JD and re-measure whether `language_score` now earns weight.
- Emit a **data-completeness flag** (separate from the score) that routes low-data candidates to screening.
- Define the **swipe-feed card data contract** + a feedback-capture design (→ human labels).
- Keep every scoring change honest: measured on gold NDCG@10, floors only ratchet up.

**Non-goals**
- **Not** lowering the evidence penalty for missing data (rejected — it's exactly what made the LLM rank Che #1; the judges penalize thin profiles *harder*, −0.64 richness→judge-rank). Missing *data* → flag + screen; we add missing *metrics*, we don't reward emptiness.
- **Not** followers/connections as a flight-risk signal (noisy; penalizes well-networked people). Flight-risk = tenure only.
- **Not** a degree *requirement* (the JD needs none; education is a soft bonus that never drops a candidate).
- **Not** the cross-encoder reranker (shelved — `docs/adr/0001`), nor front-end implementation (only the card schema).

## 4. Component 1 — Tenure & attrition (flight-risk)

### 4.1 Data & adapter change
Raw fields (present, currently unused per-role): `experience/N/duration` ("3 yrs 10 mos"), `organizations/N/startDate`, `organizations/N/endDate`, `currentPosition/0/*`.

Add a `positions` list to the canonical model and populate it in the adapter:

```python
# models/candidate.py
class CandidatePosition(BaseModel):
    title: Optional[str] = None
    company: Optional[str] = None
    start: Optional[str] = None          # raw "MMM YYYY" or year
    end: Optional[str] = None            # None / "Present" => current
    tenure_months: Optional[int] = None  # parsed from duration OR end-start
    is_current: bool = False
    employment_type: Optional[str] = None  # to exclude contract/intern from hop count

class CandidateProfile(BaseModel):
    ...
    positions: list[CandidatePosition] = Field(default_factory=list)  # enrichment; knob no-ops when empty
```

Adapter: add `_positions(record)` collecting `experience/N/{position,title,companyName,duration,employmentType}` (reuse `_collect`) merged with `organizations/N/{positionHeld,startDate,endDate}`; parse `duration` via the existing `_duration_to_months`. Keep `_years_experience` unchanged (total years still used by `experience_score`).

### 4.2 Derived features
`current_tenure_months`, `median_completed_tenure_months` (exclude the current role — it may be mid-probation), `n_dated_roles`, `n_short_perm_stints` (permanent roles < 12 mo), `total_years`.

### 4.3 Score (proposed — tune on gold)
```
if n_dated_roles < 2                          -> 0.5   (can't assess) + flag "insufficient_history"
else, base on median_completed_tenure_months of PERMANENT roles:
    >= 24 mo                                  -> 1.00
    18–24                                     -> 0.85
    12–18                                     -> 0.65
    < 12 across >= 3 perm roles (chronic hop) -> 0.30
Guards:
  · contract/internship roles are EXCLUDED from the hop count (short contracts are expected, not flight-risk)
  · early-career floor: if total_years < 3 and n_dated_roles <= 2, floor at 0.70 (don't punish juniors — the JD is entry-level, 1–4 yrs)
  · a short *current* role alone is NOT flight-risk if prior roles were stable (probation nuance)
```
Missing dates → `0.5` neutral + completeness flag (never penalize absence).

### 4.4 Active-gating
Product-motivated (like `location_score`) — active whenever the candidate has ≥2 dated roles, independent of the JD. Document as product-motivated in the weight comment.

## 5. Component 2 — Relevant / adjacent experience (YoE refinement)

Total years (existing `experience_score`) ≠ *relevant* years. This properly answers the "HR-Operations vs process" concern (e.g. Amulya) by **measuring** the ratio instead of zeroing the candidate.

### 5.1 Features (from `positions[]`)
Classify each role by title:
- **RELEVANT:** `hr`, `human resource`, `recruit`, `talent`, `payroll`, `people ops`, `personnel`, `hris`
- **ADJACENT:** `admin`, `administrative`, `coordinator`, `operations`, `executive assistant`, `pro`, `government relations`, `office`
- **UNRELATED:** everything else

`relevant_years = Σ tenure(RELEVANT)`, `adjacent_years = Σ tenure(ADJACENT)`.

### 5.2 Score (proposed)
```
relevance_ratio = (relevant_years + 0.5 * adjacent_years) / total_years
experience_relevance_score = clamp(relevance_ratio, 0, 1)
no dated/titled roles -> 0.5 neutral + flag
```
Start with title-keyword classification; upgrade to semantic if noisy (open question §11).

### 5.3 Redundancy watch
`similarity_score` already rewards HR-dense text, and `experience_score` covers total-years-in-range. **Run `--redundancy experience_relevance_score similarity_score` and `--redundancy experience_relevance_score experience_score`** — adopt only if it adds signal beyond them.

## 6. Component 3 — Education relevance (soft tie-breaker)

The JD requires **no** degree, so this is a **bonus only — never a gate, never below neutral**. Distinct from the existing gate-style `qualification_score` (inactive here). Add a separate `education_relevance_score`.

### 6.1 Data
`education[]` (degree, field) + `certifications[]` (both already parsed).

### 6.2 Score (proposed)
```
RELEVANT fields: human resources, psychology, business administration, management, commerce, law
RELEVANT certs:  CIPD, CHRP, SHRM, aPHR
relevant degree OR relevant cert   -> 1.00
business-adjacent (BBA/MBA/commerce, any) -> 0.75
unrelated degree only              -> 0.50   (NEUTRAL floor — absence of a relevant degree is NOT disqualifying)
no education data                  -> 0.50 + flag
```
Small weight; pure tie-breaker. CIPD (a JD nice-to-have) folds in here.

## 7. Component 4 — Workforce-language (Tagalog / Filipino)

The role supports a **Filipino factory workforce**; Tagalog is a genuine nice-to-have. `language_score`
is adopted at **0.15** after T3 regraded explicit language fit and de-leaked reverse language requirements.
This remains a JD/config change + existing component, not a separate Tagalog component.

### 7.1 Change (apply both, for effect + provenance)
- **Parsed JSON** `jd/parsed/hr_assistant_prime_ac.json` → `language_proficiency` add:
  ```json
  { "language": "Tagalog", "level": "conversational", "priority": "valuable" }
  ```
  (Editing the parsed store takes effect immediately — no re-extraction.)
- **Source posting** `jd/HR Assistant — Prime Focus Group (Prime AC).md` → Nice-to-have: *"Tagalog/Filipino, to support the Filipino factory workforce."* (keeps the source and the parse in sync for any future re-extract).

### 7.2 Re-measure
T3 `--c5-reablate` selected `.15`: NDCG@10 .9490→.9588, NDCG@5 .9356→.9464,
reverse unchanged. Treat this as general-language validation only; explicit Tagalog evidence is n=1.
**Confirm the Filipino-workforce assumption with the client.**

## 8. Data-completeness flag (NOT a score)

A separate output field — deliberately **excluded** from `total_score`:
```
data_completeness = {
  level: "rich" | "partial" | "low",
  missing: [ "about", "responsibilities", "skills<5", "no_dated_roles", ... ]
}
low  := (no about AND no responsibilities) OR n_skills < 5 OR no dated roles
```
**Product treatment:** low-data candidates keep their evidence-based rank (they'll rank low) **but** are surfaced with a **"Screen me" badge / lane** and treated as *potentially hire-able*. A screening call/questionnaire gathers the missing info instead of silently discarding them. Rationale: evidence-based ranking is correct, but a bullseye-title thin profile (Che) may be a real hire — route to a human, don't bury.

**Optional:** a secondary "high title/location match + low data" watchlist alongside the main feed.

## 9. Product — recruiter swipe feed (Tinder-style)

### 9.1 Card data contract (per candidate — backend emits JSON; front-end out of scope)
```jsonc
{
  "candidate_id": "muhammed-ashar-k-19b216236",
  "name": "Muhammed Ashar K",
  "title": "Human Resources Assistant",
  "current_company": "AMAN Taxi MENA",
  "location": "Al Ain, UAE",
  "rank": 4,
  "total_score": 0.71,
  "component_breakdown": {           // for a radar/bar chart
    "title": 0.90, "skill": 0.45, "similarity": 0.72, "industry": 0.30,
    "tenure_attrition": 0.65, "experience_relevance": 0.80, "education": 1.0,
    "seniority": 1.0, "experience": 0.98, "location": 0.70, "language": 0.5
  },
  "matched_signals": ["visa+PRO/gov coordination","SAP HRIS","payroll support","blue-collar workforce"], // GROUNDED in matched skills/industries — NOT the LLM boilerplate
  "flags": { "flight_risk": false, "industrial_sector": false, "workforce_language": false,
             "data_completeness": "rich" },
  "reasoning": "Bullseye title + entry level; documents visa/PRO, SAP HRIS, payroll over a blue-collar workforce (the JD's exact context).",
  "linkedin_url": "https://www.linkedin.com/in/muhammed-ashar-k-19b216236"
}
```
`matched_signals` MUST be derived from actual matched skills/industries/keywords — explicitly **not** the LLM's hallucinated boilerplate (see §2).

### 9.2 Swipe semantics & feedback loop
- **right** = shortlist/advance · **left** = pass · **up/star** = send to screening (esp. low-data cards).
- Persist every swipe as `{recruiter_id, candidate_id, jd_id, decision, ts, rank_shown}`.
- **This is the strategic payoff:** swipes are the first **real human labels**. They (a) let us finally validate the LLM-judge gold against humans, and (b) feed a future learn-to-rank / weight-tuning loop. Note **selection bias** (only shown candidates get labeled) when using them.
- Ordering = pipeline `total_score`; low-data cards interleaved with the badge (or a separate "Screen me" stack).

## 10. Scoring integration & evaluation protocol

### 10.1 Wiring (per new component — mirror `location_score`)
| file | change |
|---|---|
| `models/candidate.py` | `CandidatePosition` + `CandidateProfile.positions` |
| `core/adapters/linkedin_adapter.py` | `_positions()`; keep `_years_experience` |
| `core/data.py` | `profiles_to_dataframe` adds the per-component input columns (tenure features, relevant/adjacent years, education fields) |
| `core/scoring.py` | `calculate_attrition_score`, `calculate_experience_relevance_score`, `calculate_education_relevance_score` + active-gate branches in `calculate_total_score` |
| `models/mappings.py` | new weights (init **0.0**) + any thresholds/constants |
| `core/pipeline.py`, `evals/runner.py`, `scripts/run_hr_assistant.py` | call scorers in the **same order** + output columns |
| `scripts/calibrate_weights.py` | add cols to `COMP_COLS` + `precompute_components` (enables `--ablate`) |
| `tests/` | `test_attrition_score.py`, `test_experience_relevance_score.py`, `test_education_relevance_score.py`, adapter tenure tests |

Active-gating uses the existing pattern: `if '<x>_score' in df.columns and <gate>: active_components.append('<x>_score')`; active weights renormalize to sum 1.

### 10.2 Adopt-loop for EACH component (do not hand-set weights)
1. **Correlate** the new component with `judge_mean_score` over the 78 judged (sniff test — like industry's +0.15).
2. **`--ablate <x>_score`** — 1-D weight sweep; read **gold NDCG@10** as the decision metric (reverse-match MRR secondary/noisy).
3. **`--redundancy <x>_score <related>`** — attrition vs experience; experience_relevance vs similarity & experience.
4. **`--joint`** — re-validate the core mix.
5. **Adopt** only if gold NDCG@10 **holds or improves** AND every floor in `tests/test_eval_regression.py` still passes. Be willing to **reject** a component that doesn't clear the bar (per discipline).
6. **Ratchet** floors up; regenerate `evals/results/baseline_linkedin.json`; run the suite.
7. Record in `docs/DECISIONS.md` / `docs/BACKLOG.md`; update `AGENTS.md` champion block.

> **Gold caveat:** it's a 2-LLM-judge consensus, n=1 silver. Treat "holds/improves" as *well-corroborated*, not proven. The swipe feedback (§9.2) is the route to real human validation.

## 11. Risks & open questions

- **Entry-level vs flight-risk:** the JD is entry-level (1–4 yrs) → many candidates have short histories. The early-career floor (§4.3) must not punish juniors. Tune the threshold on gold.
- **Contractors:** exclude `contract`/`internship` from the hop count (via `employment_type`) so genuine contractors aren't mislabeled flight risks. Layoffs/legit moves are unobservable — keep the penalty gentle.
- **Adjacent-experience classification** is title-keyword-based → noisy; may need semantic classification. Risk of misclassifying "HR Operations" (relevant) vs "Operations" (adjacent).
- **Redundancy:** `experience_relevance` may overlap `similarity` (HR-dense text) → could fail the `--redundancy` test and be rejected. That's an acceptable outcome.
- **Language sparsity:** even with Tagalog, 21% coverage may keep `language_score` weight ~0. Measure, don't assume.
- **Education must stay soft** — never a gate, never below the 0.5 neutral floor (the JD requires no degree).
- **Weight budget:** adding 3 components dilutes existing weights; the `--joint` re-validation matters. Some components may not survive.
- **Tenure parsing edge cases:** "Present"/current roles, missing end dates, overlapping/concurrent roles, gaps, inconsistent date formats.
- **Swipe selection bias** when using swipes as labels.

## 12. Implementation phases (checklist)

- [x] **P1 — Adapter/model:** `CandidatePosition` + `positions[]`; `_positions()`; tenure features into `profiles_to_dataframe`. Tests. ✅
- [x] **P2 — Attrition score:** standalone measured weights rejected; product override **`.005`** as a gentle tenure tie-breaker. ✅
- [x] **P3 — Experience-relevance score:** standalone measured weights rejected; product override **`.015`**, the strongest structural leg in the joint triplet. ✅
- [x] **P4 — Education-relevance score:** prior standalone `.03` rejected; product override **`.005`**, minimal and never a gate. ✅
- [x] **P5 — Workforce-language:** explicit-declaration-only, construct corr **+.68**; `.15` raises NDCG@10 .9490→.9588 and NDCG@5 .9356→.9464 with reverse unchanged → **ADOPTED at .15**. Tagalog-specific n=1. ✅
- [x] **P6 — Completeness flag:** rule + `core/completeness.py` output field (NOT in `total_score`); wired into pipeline output + card. Tests. ✅
- [x] **P7 — Swipe feed:** grounded card contract (`core/swipe.py`) + `scripts/build_swipe_cards.py` (145 cards, 25 `screen_me`) + `SwipeEvent` capture schema. Tests. Front-end out of scope. ✅
- [x] **P8 — Land it:** regenerated baseline, floors held (NOT ratcheted — eval-neutral), full suite green (144), updated `AGENTS.md` / `docs/DECISIONS.md` / `docs/BACKLOG.md`. ✅

## 13. Appendix — evidence from this session

### The 6 reference candidates (pipeline / judge / LLM) + the missing signals they expose
| candidate | pipe# | judge# (score) | LLM# (fit) | recruiter signals the pipeline misses |
|---|--:|--:|--:|---|
| muhammed-ashar | 4 | **1** (85) | 54 (4) | Al Ain (far, on-site); 10-mo current tenure |
| harikrishna | 10 | **2** (84) | 4 (9) | mfg sector; short probation (≈ availability); mid/over-title |
| megha-p-s | 19 | **3** (82) | 25 (6) | explicit WPS; precast/building-materials; skills-list generic though duties rich |
| amulya-dattada | **1** | 4 (80) | 42 (5) | "HR-Ops vs pure HR" → relevant-vs-adjacent years; Finance degree |
| samasthasunoj | **3** | 9 (76) | 20 (6) | EPC/construction + Arabic; job-hopping (≤1.4 yr/role); 10k followers (NOISE — ignore) |
| arunmania | **2** | 17 (68) | 37 (5) | hospitality-skewed sector; 1.2-yr tenure (flight-risk signal) |
| **che (anchor)** | 132 | 67 (36) | **1 (9)** | empty profile → **low-data flag + screen**, not disregard |

### Key measured facts (so the next session need not re-derive)
- LLM `matched_signals` are boilerplate/hallucinated (identical across shortlist ranks 1–18; asserted for empty & irrelevant profiles). Do not trust them; card `matched_signals` must be grounded in real matches.
- `similarity` +0.86 and `skill` +0.52 correlate with judges (heavy weights validated); `title` +0.02, `seniority` −0.08 (range-restricted — don't shift weight to the surface).
- Profile richness → judge-rank −0.64 vs pipeline −0.57: **judges penalize thin profiles even harder** → do not soften the evidence penalty.
- Adapter drops per-role tenure (sums durations). Raw export has per-role `duration` + `organizations/N/startDate/endDate`.

### Rejected alternatives (do not re-litigate without new evidence)
- **"Penalize less for missing data" / neutral-default on skill+similarity** — rejected: it's the exact mechanism behind the LLM's Che #1; rewards sparsity; moves against the judges. Use the completeness flag instead.
- **Followers/connections as flight-risk** — rejected: noisy, penalizes networked candidates. Flight-risk = tenure only.
- **Degree as a gate** — rejected: the JD requires no degree; education is a soft bonus that never drops a candidate.
