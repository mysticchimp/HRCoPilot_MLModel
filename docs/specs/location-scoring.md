# Spec — Location Scoring Component

| | |
|---|---|
| **Status** | ✅ Implemented + **adopted at weight `0.05`** (product-motivated, 2026-07-12) after de-leaking location from reverse-match |
| **Type** | New optional, weight-gated scoring component (`location_score`) |
| **Decision** | `docs/DECISIONS.md` → "Language & location components" |
| **Tests** | `tests/test_location_score.py` (12) |
| **Baseline** | `evals/baseline_linkedin.json` (unchanged — eval-neutral at 0.05) |

---

## 1. Summary

A new optional component that scores candidate **location fit** against the JD's `location.cities`
and `location.countries`, on a **country > city hierarchy** (country is the primary gate, the city
refines within it). A **city match** is full credit; inside a matching country an **omitted** city is
*not* penalized (benefit of the doubt — it scores equal to a city match), while a **confirmed
different** city earns partial credit; a wrong country is a miss. UAE ↔ United Arab Emirates (and the
`AE` code) are aliased. It is **adopted at weight `0.05`** as a deliberate **product** decision (an
on-site role has a real location constraint), after **de-leaking location from reverse-match** (which
invents a seed-mismatched location it cannot fairly score). At 0.05 the adoption is **eval-neutral**
— gold NDCG@10 holds (`0.6334`) and reverse is unchanged — so it earns its place on operational
grounds, not as an eval-proven ranking gain.

## 2. Motivation

`CandidateLocation` (city / country / country_code) was extracted but never scored. Recruiters care
about location, and the real HR JD names Dubai / United Arab Emirates, so location is a plausible
discriminator worth measuring. Per user intent, the component scores **both city and country** (not
country-only), with country as the fallback when the JD extraction omits a city.

## 3. Goals / Non-goals

**Goals**
- Score city (strongest) → in-country (partial) → mismatch, with a country-only JD treating a country
  match as full credit.
- No-op cleanly (neutral 0.5, dropped from the blend) when the JD names no location or a candidate
  has none.
- Adopt a weight only on **non-regression** of gold NDCG@10 + floors (the n=1 silver gold rubric is
  not expected to grade emirate, so the bar is "does not hurt", not "must improve").

**Non-goals**
- **No fuzzy / embedding matching** (short place tokens collide; same precision rationale as industry
  and language). Aliases are explicit.
- No travel / remote / relocation modeling (only `cities` + `countries` are used).

## 4. Design

### Normalization (`core/location_normalization.py`)

- `normalize_country(value)` — casefold + strip, map `COUNTRY_ALIASES` (`united arab emirates ←
  uae / u.a.e / emirates / …`).
- `normalize_country_code(code)` — 2-letter ISO code → canonical country (`ae → united arab
  emirates`); `''` if unknown.
- `normalize_city(value)` — casefold + strip, map light `CITY_ALIASES` (a few UAE spellings).
- `jd_location_requirements(jd)` → `(cities, countries)` frozensets, normalized, **empty entries
  dropped** (extraction yields `cities: ["Dubai", ""]`; the blank is filtered, so a country-only
  requirement stays country-only). All normalizers guard non-`str` cells (the dataframe stores
  `None` for missing fields).

### Scorer (`core/scoring.py` → `calculate_location_score`)

Country is the **primary gate**; the city **refines** within it. An *omitted* city is not penalized
(absence of evidence ≠ evidence of absence), so it scores equal to a city match; only a *confirmed
different* city earns the partial 0.7.

```
city         = normalize_city(cand.city)
cand_country = normalize_country(cand.country) or normalize_country_code(cand.country_code)

no city AND no country                          -> 0.5   (total unknown)
city ∈ JD cities                                -> 1.0   (definitive — strongest signal)
country ∈ JD countries  (primary gate met):
    · city omitted, or JD names no city          -> 1.0   (benefit of the doubt — omission not penalized)
    · city present but ∉ JD cities               -> 0.7   (confirmed different city, in the right country)
country present but ∉ JD countries               -> 0.0   (confirmed wrong country)
otherwise (country unconfirmable)               -> 0.0 if a non-matching city, else 0.5
```

Constants live in `models/mappings.py` (`location_city_match=1.0`, `location_country_match=0.7`,
`location_mismatch=0.0`).

### Active-gating (`calculate_total_score`)

```python
if 'location_score' in df.columns and jd.location and (jd.location.cities or jd.location.countries):
    active_components.append('location_score')
```

A `0.0` weight adds nothing to the weighted sum or the normalizer, so `total_score` is byte-identical
to the champion while the weight is 0.

## 5. Integration points

| file | change |
|---|---|
| `core/location_normalization.py` | **new** — country/city aliases + code map, normalizers, `jd_location_requirements` |
| `core/data.py` | `profiles_to_dataframe` adds `location_city` / `location_country` / `location_country_code` columns |
| `core/scoring.py` | `calculate_location_score` + active-gate branch in `calculate_total_score` |
| `models/mappings.py` | `location_score` weight (`0.0`) + `location_city_match` / `location_country_match` / `location_mismatch` |
| `core/pipeline.py`, `evals/runner.py`, `scripts/run_hr_assistant.py` | call the scorer (same order everywhere) + output columns |
| `scripts/calibrate_weights.py` | `location_score` in `COMP_COLS` + `precompute_components` (enables `--ablate location_score`) |
| `tests/test_location_score.py` | 12 tests |

## 6. Data characterization (LinkedIn pool, n = 145)

**133/145 (92%)** have a city; **144/145 (99%)** are United Arab Emirates. City split: Dubai 67%,
Abu Dhabi 17%, no-city 8%, Sharjah 3%, Ajman 2%, Ras Al Khaimah 1%, Fujairah 1%, + 1 India. On the
real JD (`cities: ["Dubai"]`, `countries: ["United Arab Emirates"]`) the refined score distribution
is `1.0` (Dubai + no-city UAE) ×109, `0.7` (confirmed other emirate) ×35, `0.0` (India) ×1 — the
component discriminates by *emirate*, but an **omitted** city gets the benefit of the doubt (full
credit), not a demotion.

> **Known limitation:** LinkedIn appends "City" to some parsed cities ("Ajman City",
> "Ras Al Khaimah City"), which do not normalize to the bare city name. Irrelevant for a Dubai JD
> (those candidates score the 0.7 country match regardless), but add aliases before scoring a JD
> that targets one of those emirates.

## 7. Evaluation & decision

### The reverse-match artifact (and why location was de-leaked)

Reverse-match holds the seed's real fields out of JD generation (leakage control), but the LLM still
**invents** a location (e.g. "Dubai") that isn't tied to the seed. Diagnosing the committed reverse
cases: 12 carried a location, and in 4 the invented requirement scored the seed *below* the pool
(2 severely — 75% of the pool outranked the seed). So a nonzero location weight dropped reverse MRR
purely because the seed didn't match a made-up requirement — an **eval artifact, not a quality loss**
(confirmed by the gold case, where the location is real, not regressing).

The fix mirrors the existing seniority/experience de-leak: **hold location out of reverse-match**
(`build_reverse_match_case` now sets `parsed_jd.location = None`; the committed reverse fixtures were
re-saved likewise). Matching the JD's location *to* the seed was rejected — that is the opposite,
leakage-inflating error. With reverse made location-neutral, the sweep is clean:

| weight | reverse MRR (de-leaked) | gold NDCG@10 |
|---:|---:|---:|
| **0.00** | 0.4150 | 0.6334 |
| **0.05** | **0.4150** | **0.6334** |
| 0.10 | 0.4150 | 0.6398 |
| 0.20 | 0.4150 | 0.6714 |

Reverse MRR is now perfectly flat (location does no harm); gold holds at 0.05 and rises after (the
rise is **n=1, not robust**).

### Decision — adopted at 0.05 (product-motivated)

At **0.05** nothing regresses (gold holds `0.6334`, reverse unchanged, all floors pass, baseline
byte-identical). Gold shows no *measurable* gain at this weight, so location is adopted as a
deliberate **product** decision — an on-site Dubai role has a real location constraint recruiters act
on — explicitly **eval-neutral, not an eval-proven ranking gain**. The weight is kept small (0.05) so
it is a gentle tie-breaker (on the real JD it nudges Dubai/UAE candidates up without demoting strong
non-Dubai ones — e.g. `muhammed-ashar-k` at 0.7 still holds a top-5 spot on similarity). Higher
weights (0.10–0.20) only chase the non-robust n=1 gold rise and are avoided.

## 8. Follow-ups

Revisit the weight (upward) once the eval can actually *reward* location — a **location-diverse** pool
(multi-country JDs where country fit discriminates) or a location-graded gold set. Add the "… City"
city aliases (§6) before scoring a non-Dubai-emirate JD.
