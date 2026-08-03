# Spec — Anchor Re-grade (tenure/relevance/language-aware blind judges) + Compliance-Gap Analysis (C4)

| | |
|---|---|
| **Status** | (a) ✅ **T3 implemented + measured (2026-07-31)** · (b) 📝 C4 remains separate/de-prioritized |
| **Type** | (a) eval-anchor change: expand the blind-judge rubric · (b) analysis: the compliance "gap" is mostly covered by `skill_score` (backlog **C4** de-prioritized) |
| **Depends on** | C5 components (standalone measured weights rejected; tiny product tie-breakers adopted later) — `docs/specs/recruiter-signals-and-swipe-feed.md`; the `fit_0_10` source rubric (Contra6) |
| **Motivated by** | Gap analysis vs the `fit_0_10` source rubric + the current blind-judge rubric (§2) |
| **Rejected / guardrails** | Name→ethnicity language inference (the `fit_0_10` Che error); injecting the pipeline's computed features into the judge payload (adds circularity) |

---

## 1. Summary

Two changes, both motivated by one finding: **the prior silver labels did not score the signals the new
components model.** (a) Expand the **blind-judge rubric** so the regraded anchor can fairly reward tenure,
relevant-vs-adjacent experience, and workforce-language (Tagalog) — the prerequisite for honestly
re-ablating the C5 components. (b) Add the **essential-skill must-have gate** (backlog **C4**): UAE compliance
is the *single heaviest* human signal, but it is ALREADY in the JD's essential skills — so `skill_score`
already scores it. C4's value is therefore the missing *mechanism* — a JD-driven gate that ENFORCES the
essentials — **not** a hard-coded compliance lexicon (which would double-count `skill_score`).

Both are documented here and implemented **separately**; every new weight stays at **0** until measured.

> **(b) revised (see §4):** the compliance "gap" is **largely illusory** — `skill_score` already carries
> compliance, priority-weighted at the essential max (1.0) inside a joint-sweep-validated 0.25 weight. The
> honest lever for "more compliance weight" is **inside `skill_score`** (its weight or the essential-priority
> weight, re-run on the joint sweep), NOT a new component (which double-counts). A component is justified only
> for a hard must-have **gate**, which is **expected redundant** on this pre-filtered HR pool. C4 is
> **de-prioritized**; the rubric re-grade (a) is the higher-value work.

## 2. Gap analysis — what each "scorer" actually rewards

### 2.1 The `fit_0_10` source rubric (Contra6)

The repo's `fit_0_10` labels (`data/Scored_FullPool_HR_Assistant_v2…csv`) were produced by an LLM
signal-checklist: it marks which **signals** a profile "clearly evidences", then sums tier weights
(`strong=3, normal=2, light=1, exclude=−5`) and normalizes to 0–10. Confirmed — the `matched_signals`
column is verbatim these signals. Retrieval **pre-filters** the pool (function=HR, UAE, 1–5 yrs, HR
titles), so those signals don't discriminate *within* the pool; variance comes from the scored signals:

| Signal | Tier |
|---|---|
| Assistant/coordinator-level fit ("bullseye") | (implicit level signal) |
| Industrial-sector employer / Prime-adjacent (HVAC/MEP/fabrication/mfg) | strong |
| **UAE labour/payroll admin** (WPS, MOHRE, labour card, visa, Tasheel, GDRFA) | **strong** |
| Named HRIS platform (Bayzat/ZenHR) | normal |
| Tagalog (declared) | normal |
| UAE-based HR history | normal |
| Arabic | light |
| CIPD / HR cert | light |
| Northern-Emirates base | light (global) |
| Over-qualified (senior title / 10+ yrs) | **exclude (−5, hard)** |

**Two known defects of this anchor:** (1) it is a coarse binary checklist; (2) it **hallucinates
signals on thin profiles** — its **#1 (fit 9) is Che**, credited with `Tagalog (declared)` although
Tagalog appears **nowhere** in her record. The credit was a **name→ethnicity inference** (Filipino name ⇒
assumed Tagalog) — exactly the demographic inference the blind-judge system prompt forbids. This is why the
pipeline + blind judges correctly bury Che, and why `fit_0_10` is distrusted.

### 2.2 Coverage matrix

| Signal | `fit_0_10` | T3 blind Judge panel | ML pipeline |
|---|---|---|---|
| Level/title "assistant bullseye" | ✅ | ✅ Role&level 0-15 | ✅ title (range-restricted) + seniority |
| **UAE compliance / payroll / PRO** | ✅ strong | ✅ **0-22 (heaviest)** | ⚠️ IN `skill_score` (JD essentials) but **not enforced as a must-have** |
| Industrial sector (mfg/HVAC/MEP) | ✅ strong | ✅ 0-10 | ✅ `industry_score` 0.20 |
| HRIS (Bayzat/ZenHR) | ✅ normal | ✅ 0-8 | ⚠️ skill + similarity |
| Core HR ops | ⚠️ implied | ✅ 0-15 | ⚠️ skill + similarity |
| **Tagalog / Filipino** | ✅ normal (name-inferred for Che) | ✅ preferred, explicit declaration only | ✅ `language` .15, structured-field only |
| Arabic | ✅ light | ✅ preferred | ✅ `language` .15 |
| CIPD / HR cert | ✅ light | ✅ preferred | ⚠️ education component available, weight 0 |
| UAE / Northern-Emirates location | ✅ (normal + light) | ⚠️ context only | ✅ `location` 0.05 (NE nuance unmodeled) |
| Over-qualification ceiling | ✅ **exclude (hard)** | ⚠️ soft-penalize managerial | ⚠️ soft seniority/experience penalties |
| **Tenure / continuity (flight-risk)** | ❌ **never a signal** | ✅ 0-8 | ✅ `attrition` `.005` product tie-breaker |
| Relevant-vs-adjacent HR years | ❌ (pre-filtered to HR) | ✅ 0-7 | ✅ `experience_relevance` `.015` product tie-breaker |
| Evidence quality / anti-thin | ❌ | ✅ 0-5 | ⚠️ completeness flag (not scored) |

### 2.3 The gaps

1. **Tenure was anchored nowhere before T3.** The implemented rubric now scores it explicitly; re-ablation
  still rejected attrition because the measured weight trade-off failed the full adoption gate.
2. **The heaviest human signal (UAE compliance/PRO) is ALREADY scored.** `skill_score` carries it
   priority-weighted (essential=1.0) inside a 0.25 weight, so the pipeline isn't under-weighting it. The only
   thing `skill_score` can't do is *enforce* it as a hard must-have — and per §4 the honest lever for more
   emphasis is **inside `skill_score`**, not a new component. So this "gap" is largely illusory and C4 is
   **de-prioritized**.

## 3. (a) Expanded blind-judge rubric

**Pre-T3 rubric** (historical, 7 sections summing to 100): Role & level
fit 20 · UAE compliance/payroll/PRO 25 · Core HR ops 20 · HR systems/tools 10 · Industrial context 10 ·
Preferred signals (Arabic, CIPD, English, gov portals) 10 · Evidence quality 5.

**Implemented rebalance to 100** (adds tenure, relevance mix, Tagalog; trims the broad sections):

| Section | Now | Proposed | Change |
|---|--:|--:|---|
| Role & level fit | 20 | 15 | −5 (relevance split out) |
| UAE compliance / payroll / PRO depth | 25 | 22 | −3 (still heaviest) |
| Core HR operations | 20 | 15 | −5 |
| HR systems & office tools | 10 | 8 | −2 |
| Industrial / blue-collar context | 10 | 10 | — |
| **Tenure & continuity** | — | **8** | NEW |
| **Career relevance mix (relevant vs adjacent HR yrs)** | — | **7** | NEW |
| Preferred signals (Arabic, CIPD, **Tagalog/Filipino**, English, gov portals) | 10 | 10 | +Tagalog |
| Evidence quality | 5 | 5 | — |
| **Total** | 100 | **100** | |

**Neutral wording for the new sections** (steer *what* to consider, not *our number* — avoids baking in
our prior):
- *Tenure & continuity (0–8):* "Reward a stable, coherent job history as a recruiter would — consider
  average time per role and whether the candidate stayed long enough to deliver. Do **not** over-penalize
  junior candidates (1–4 yrs), contract roles, or a short *current* role (they haven't left yet). Judge
  only from the per-role durations provided."
- *Career relevance mix (0–7):* "Reward time in genuinely HR/people roles over adjacent
  admin/coordination/operations; give partial credit to adjacent roles. Judge from the titles + durations
  provided."
- *Preferred signals — Tagalog:* "Tagalog/Filipino to support the Filipino factory workforce — credit
  **only an explicit declaration** (a listed language or stated proficiency). **Never** infer language or
  nationality from a name." (Reinforces the existing anti-demographic guardrail; directly fixes the Che
  error.)

**Payload:** unchanged — the judges already receive per-role `duration`, `derived_total_years`,
`education`, `certifications`. Do **NOT** inject the pipeline's computed features (median completed-perm
tenure, relevant/adjacent years, education tier) — that would make the calibration circular.

**Re-grade loop:** edit rubric → re-run `scripts/blind_judge_rankings.py` **live** (pinned Opus 4.8 +
GPT-5.5; `COPILOT_SKIP_CLI_DOWNLOAD` unset) → rebuild `evals/judgments/blind_judgments_hr_assistant.csv`
+ the gold fixture relevance → check inter-judge agreement (Spearman) → re-establish the champion baseline
→ re-ablate `attrition` / `experience_relevance` / `education` on the **new** gold.

**Methodology-reset caveat:** changing the rubric changes *every* label, so the champion's own gold NDCG@10
moves. Floors are **RESET** to the new champion (documented as an EXCEPTION, like the `fit_0_10`→blind
repoint), **not** ratcheted. Adoption = "does the component improve the *new* gold vs the *new* baseline."

**Circularity caveat:** we're calibrating to an LLM opinion we just steered → still **silver**. It removes
the "blind anchor" objection but doesn't *prove* the signals matter. The un-circular validation stays
**U2 (real recruiter swipes)**.

## 4. (b) The UAE-compliance "gap" — mostly already covered; a component is optional (backlog C4)

### 4.1 `skill_score` already carries compliance (priority-weighted, heavy)
`weighted_fuzzy_skill_score` computes, over the JD's skills + technologies,
`score = Σ w(s)·strength(cand, s) / Σ w(s)`, where `w` is the priority weight (`essential=1.0, important=0.7,
valuable=0.4`) and `strength = max(fuzzy, semantic)`. For this JD, WPS / visa / MOHRE / UAE-Labour-Law are all
`essential`, so they ALREADY carry the **maximum** per-skill weight (1.0) inside a **joint-sweep-validated 0.25**
component. The pipeline is NOT under-weighting compliance — the "gap" from the source-rubric analysis was
largely **illusory**.

### 4.2 The honest lever for "more compliance weight" is INSIDE `skill_score` — not a new component
If the goal is "compliance should count more", the measurable, non-double-counting levers are:
- raise the **skill component weight** (`candidate_score_weights['skill_score']`, currently 0.25) and re-run the
  **joint sweep**; or
- raise the **essential priority weight** (`attribute_weight_by_importance[ESSENTIAL]`) so essentials dominate
  the skill blend more, and re-run the sweep.
Both are one-knob offline experiments with no double-count — and the joint sweep already validated skill ≈ 0.25.
A separate compliance component that re-detects the same skills and adds its OWN weight would **double-count**
`skill_score` and **hand-set** an emphasis the sweep didn't support (violates "weights are measured, never
hand-set").

### 4.3 The ONLY thing a component adds that `skill_score` can't: a hard must-have GATE
`skill_score` is a **linear graded blend** — a candidate missing an essential just loses that skill's
proportional contribution; there is no "missing an essential ⇒ disqualify". The only non-redundant component is
therefore a **non-linear must-have gate** (JD-driven: `[s for s in jd.skills if s.priority == ESSENTIAL]`,
`essential_coverage = matched/total`, hard-drop/penalize below a threshold; reuse the same hybrid matcher; wire
like location/industry; weight 0.0; tests). This is a MECHANISM, not "more weight".

### 4.4 …but it is EXPECTED REDUNDANT on this pool — measure before building
The signal is already in `skill_score`, and the pool is **pre-filtered to HR-titled UAE candidates**, so almost
everyone has the core compliance skills and the few who don't already rank low. A gate is therefore expected to
be redundant or harmful here. Decision metric if you build it: `--redundancy essential_gate_score skill_score`
— does hard-gating beat `skill_score`'s graded blend on gold NDCG@10? If not (the likely outcome), **REJECT**.
**Prefer the §4.2 skill-weight / essential-priority sweep first**; only build the gate if you specifically need
**disqualification** semantics (e.g. a different, non-pre-filtered pool).

## 5. Sequencing
1. **Rubric re-grade (T3)** is the higher-value eval work — it fixes the tenure blind spot that makes the C5
   ablations unfair. Live LLM cost; then re-ablate `attrition` / `experience_relevance` / `education` (and
   re-check `language`/Tagalog under the explicit-declaration rule) on the tenure-aware gold.
2. **Compliance emphasis (C4)** — if wanted, FIRST try the offline skill-weight / essential-priority sweep
   (§4.2, no new component). Only build the hard must-have gate (§4.3) if you need disqualification semantics,
   and expect it to be rejected as redundant on this pool.
3. **U2 swipes** remain the real, un-circular validation for all of the above.

## 6. Risks / open questions
- **C4 is likely not worth a component.** The compliance signal is already in `skill_score` (priority-weighted,
  essential=1.0, 0.25 weight); "more weight" via a component is a double-count. The honest lever is the
  skill-weight / essential-priority sweep (§4.2); a component only adds a hard gate (§4.3), expected redundant
  on this pre-filtered pool.
- **Re-grade circularity** — steering the judge then calibrating to it; keep wording neutral, keep 2 judges.
- **Tagalog free-text detection** (a separate future pipeline idea): only ever on an **explicit
  declaration**, never name/nationality inference (the Che defect). Structured-first remains safest.
- **Over-qualification**: `fit_0_10` hard-excludes; the pipeline soft-penalizes. A hard manager/director
  gate for an assistant seat is a C4-adjacent decision to make deliberately, not by default.
- **n=1 silver** throughout — none of this closes T1 (human, multi-JD gold); it makes the single anchor
  *fairer*, not *proven*.
