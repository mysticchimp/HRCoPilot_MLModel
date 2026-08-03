# Spec — Language Scoring Component

| | |
|---|---|
| **Status** | ✅ Implemented · ✅ **ADOPTED at `0.15` (T3, 2026-07-31)** |
| **Type** | New optional, weight-gated scoring component (`language_score`) |
| **Decision** | `docs/DECISIONS.md` → "Language & location components" |
| **Tests** | `tests/test_language_score.py` (8) |
| **Baseline** | `evals/results/baseline_linkedin.json` |

---

## 1. Summary

A new optional component that scores candidate **language fit** against the JD's
`language_proficiency` list. Matching is **presence-based** and **priority-weighted** — mirroring
`industry_score` — using normalized-exact matching with a small alias table (e.g. Tagalog ==
Filipino). T3 regraded the silver Judge-grade anchor with explicit language scoring and de-leaked
ungrounded reverse-JD language requirements. At weight `0.15`, NDCG@10 improves `.9490→.9588` and
NDCG@5 improves `.9356→.9464`, while reverse metrics are unchanged. This validates the **general
language component** on one circular silver JD; it does not validate Tagalog specifically (n=1).

## 2. Motivation

`CandidateProfile.languages` (from the LinkedIn adapter) and the JD's `language_proficiency` were
both extracted but never scored (an "unused signal" per `docs/DECISIONS.md`). For the real HR JD,
English is **essential** and Arabic **valuable**, so language is a plausible discriminator worth
measuring.

## 3. Goals / Non-goals

**Goals**
- Score language presence against the JD, priority-weighted, behind a weight-gated seam.
- No-op cleanly (neutral 0.5, dropped from the blend) when the JD names no languages or a candidate
  lists none.
- Adopt a weight **only** if it holds/improves gold NDCG@10 and clears every regression floor.

**Non-goals**
- **No proficiency-level gating** (MVP). Candidate proficiency is noisy free-text
  ("Native or bilingual proficiency") and most list the essential language at a high level, so the
  component scores *presence*, not level. Proficiency-gating is a possible future refinement.
- **No fuzzy / embedding matching.** Language names are short tokens where char-level and semantic
  similarity produce false merges (the same precision concern that ruled out fuzzy/semantic industry
  matching). Genuine synonyms are handled by explicit aliases.

## 4. Design

### Normalization (`core/language_normalization.py`)

- `normalize_language(name)` — casefold + strip, drop trailing qualifiers ("English (US)",
  "Filipino/Tagalog"), then map aliases → canonical. Unknown languages return their own casefolded
  root, so any language still matches when both sides spell it the same.
- `LANGUAGE_ALIASES` — only genuine merges (`filipino ← tagalog/pilipino`, `chinese ← mandarin/…`,
  `persian ← farsi`). Mutually-intelligible-but-distinct languages (Hindi vs Urdu) are kept
  **separate** on purpose.
- `normalize_candidate_languages(names)` — a candidate's raw names → a `frozenset` of canonicals.
- `jd_language_requirements(jd)` — `[(canonical_language, priority)]` from `jd.language_proficiency`
  (deduped; the required `level` is intentionally ignored in the MVP).

### Scorer (`core/scoring.py` → `calculate_language_score`)

Mirrors `calculate_industry_score`:

```
total_weight = Σ priority_weight(req)                       # essential 1.0 / important 0.7 / valuable 0.4 / supplementary 0.2
score(cand)  = Σ priority_weight(req) for req in JD if req ∈ cand_languages   / total_weight
```

- Candidate with **no** listed languages → neutral **0.5**.
- JD with **no** `language_proficiency` → component inert (0.5) and dropped by
  `calculate_total_score`.

### Active-gating (`calculate_total_score`)

```python
if 'language_score' in df.columns and jd.language_proficiency:
    active_components.append('language_score')
```

Active weights are renormalized to sum 1. The adopted raw weight is `0.15` when a JD names languages;
the component remains inert when `language_proficiency` is absent.

## 5. Integration points

| file | change |
|---|---|
| `core/language_normalization.py` | **new** — aliases, `normalize_language`, `normalize_candidate_languages`, `jd_language_requirements` |
| `core/data.py` | `profiles_to_dataframe` adds a `languages` column (raw candidate language names) |
| `core/scoring.py` | `calculate_language_score` + active-gate branch in `calculate_total_score` |
| `models/mappings.py` | `candidate_score_weights['language_score'] = 0.15` |
| `core/pipeline.py`, `evals/runner.py`, `scripts/run_hr_assistant.py` | call the scorer (same order everywhere) + output columns |
| `scripts/calibrate_weights.py` | `language_score` in `COMP_COLS` + `precompute_components` (enables `--ablate language_score`) |
| `tests/test_language_score.py` | 8 tests |

## 6. Data characterization (LinkedIn pool, n = 145)

Only **31/145 (21%)** list any language. The canonical JD names English (essential), Arabic
(valuable), and Tagalog (valuable). Current score distribution: `0.5` (no data) ×114, `0.556`
(English only) ×19, `0.778` (two weighted matches) ×11, and `0.0` (explicit languages but no match) ×1.
Only `efrelyn-ablay` explicitly lists Tagalog; both judges credited it only there.

## 7. Evaluation & decision

Ablation via `scripts/calibrate_weights.py --c5-reablate` against the common C5-neutral control
(decision metric = unrounded silver-anchor NDCG@10; all regression floors binding):

| weight | NDCG@10 | NDCG@5 | reverse MRR | hit@10 |
|---:|---:|---:|---:|---:|
| 0.00 | 0.9490 | 0.9356 | 0.5190 | 0.7895 |
| 0.05 | 0.9526 | 0.9376 | 0.5190 | 0.7895 |
| 0.10 | 0.9572 | 0.9446 | 0.5190 | 0.7895 |
| **0.15** | **0.9588** | **0.9464** | **0.5190** | **0.7895** |
| 0.20 | 0.9588 | 0.9464 | 0.5190 | 0.7895 |

Construct correlation with explicit Judge-panel language credits is **+.68**. Weights `.15` and `.20`
tie; the preregistered tie-break chooses the lower weight. **Decision: adopt `language_score = 0.15`.**
Reverse metrics are unchanged because reverse JDs hold language out: seed languages were excluded from
generation, yet 17/19 reverse JDs invented requirements and 14 had no explicit seed match. Matching the
invented requirement to the seed was rejected as leakage.

## 8. Follow-ups

Revalidate with recruiter swipes and a multi-JD human set before treating `.15` as proven or portable.
Tagalog-specific validation requires more than the single explicit declarer. Improve candidate language
coverage and revisit proficiency-level scoring only with evidence.
