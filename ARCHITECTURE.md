# AI Recruiter Pipeline - Project Architecture

## 📁 **core/** - Core Processing Engine
The main processing pipeline containing all core functionality modules.

- **`data.py`** - Data structures and candidate/job data handling
- **`email_generator.py`** - Automated email generation for candidate outreach
- **`embedding.py`** - Text embedding generation using vector models for semantic matching
- **`filtering.py`** - Initial candidate filtering logic
- **`jd_extraction.py`** - Job description parsing and feature extraction
- **`matching.py`** - Core matching algorithms between candidates and job requirements
- **`normalization.py`** - Data normalization and standardization processes
- **`pipeline.py`** - Main pipeline orchestration and workflow management
- **`scoring.py`** - Candidate scoring and ranking algorithms

## 📁 **data/** - Data Storage
Contains structured data files for the application.

- **`df_candidates.pkl`** - Serialized DataFrame containing candidate profiles and CVs
- **`resume_data.csv`** - Raw resume data in CSV format for processing

## 📁 **jd/** - Job Description Management
Handles job description processing and sample outputs.

- **`sample-outputs/`** - Directory containing sample job description processing results
- **`sample_jd_01.txt`** - Sample job description file for testing and development

## 📁 **models/** - Data Models & Configuration
Defines data structures and model configurations.

- **`data_models.py`** - Pydantic models and data classes for candidates, jobs, and results
- **`enums.py`** - Job category definitions and skill importance levels (e.g., required, preferred, nice-to-have)
- **`mappings.py`** - Field mappings and data transformation configurations

## 📁 **prompts/** - AI Prompt Templates
Contains prompt templates for AI operations.

- **`email_generation.py`** - Prompt templates for generating recruitment emails
- **`jd_extraction.py`** - Prompt templates for extracting structured data from job descriptions

## 📁 **tests/** - Test Suite
Unit and integration tests for the pipeline components.

- **`test_eval_pipeline.py`** - Tests for pipeline evaluation and performance metrics
- **`test_matching_pipeline.py`** - Tests for the core matching algorithm functionality

## 📁 **utils/** - Utility Functions
Helper functions and utilities.

- **`parsing.py`** - Text parsing utilities

## 📚 **Main Application Files**
- **`ai_recruiter.ipynb`** - Jupyter notebook for interactive development and testing
- **`main.py`** - Main application entry point and CLI interface

## 🏗️ **System Architecture Overview**

> **📊 Detailed current-state pipeline diagram + stage-by-stage walkthrough:**
> see **[`docs/PIPELINE.md`](docs/PIPELINE.md)**. The list below is only a high-level summary;
> for *how the pipeline evolved* see `evals/pipeline_improvement_report.md`.

The pipeline follows this general flow:

1. **Job Description Processing** → Extract structured requirements from the free-text JD via an LLM (`JobRoleSchema`)
2. **Data Ingestion** → Load candidates from CSV through an adapter → canonical `CandidateProfile`
3. **Embedding Generation** → Vectorize candidate profiles (and the JD) for semantic matching
4. **Title Gate** → Soft hybrid title score (scores, no hard drop)
5. **Component Scoring** → title · skill · qualification · seniority · experience · industry · similarity
6. **Scoring & Ranking** → Active-gated, weight-renormalized fusion → rank by `total_score`
7. **Output Generation** → Top-N shortlist / outreach emails for recruiters

## 🔄 **Key Components Integration**

- **Core Pipeline** orchestrates the entire process
- **Models** define data structures used throughout
- **Prompts** provide LLM templates for intelligent text processing
- **Utils** support parsing and data manipulation
- **Tests** ensure pipeline reliability and performance

## ⚠️ **Limitations & Known Gaps (TBD)**

- **Location scoring not implemented (deferred).** The canonical `CandidateProfile`
  already carries structured `location`, but the current LinkedIn evaluation pool is
  144/145 UAE-based, so a location component has no candidates to discriminate between
  and cannot be validated on this data. It is intentionally skipped for now and should
  be added (as another optional, weight-gated component) once a location-diverse
  dataset / evaluation set is available.
- **Reverse-match evaluation is optimistic (partially mitigated).** The primary eval
  synthesises a job description from a seed candidate's profile and checks that the seed
  ranks highly. Explicit `skills`, `education`, `seniority`, and `years_experience` are
  now **held out of JD generation** (fixtures were regenerated), which removes the direct
  metadata copy that previously inflated `seniority_score` / `experience_score`. Residual
  optimism remains because the JD is still derived from the seed's title / responsibilities,
  so reverse-match `MRR` / `hit@k` are treated as **secondary, noisy** signals.
- **The gold case is a silver, single-JD anchor.** The leakage-free cross-check is one
  real posting (`jd/HR Assistant — Prime Focus Group (Prime AC).md`) with graded relevance
  scored across the 145-candidate pool → `NDCG@10`. Two caveats: (1) the `fit_0_10` grades
  are **LLM/rubric-generated** ("Contra6 / RUBRIC v2"), **not human** — a *silver*, not
  gold, standard; (2) it is **n=1**. The pipeline is now fed the exact JD the grades were
  produced against (previously a placeholder, which inflated `NDCG@10` from a true ~.60 to
  ~.70). Because the rubric emphasises *level-fit*, this single JD pulls weights toward
  seniority/experience while reverse-match pulls toward title/skill/similarity — a tension a
  single case cannot authoritatively resolve.
- **Expanding the graded set beyond n=1 is the highest-leverage next step.** It is
  currently the binding constraint on confidently calibrating weights (in particular, how
  much to reward level-fit). Adding several human-graded JDs would let the honest metric
  drive weight decisions rather than a single silver-labelled case.

## 🚧 **Planned Improvements (design — not yet implemented)**

The pipeline-vs-LLM agreement analysis on the HR-Assistant pool (Spearman ρ ≈ 0.20,
top-10 overlap 1/10, skill scores mostly < 0.35) shows the **final ranking quality** is the
main headroom. Candidate upgrades, each to be adopted only if it clears the eval
harness's regression floors on the (expanded) honest set:

1. **Cross-encoder re-ranker (retrieve → re-rank).** Keep the bi-encoder (`all-mpnet`) as
   a fast recall + component-scoring stage, then re-score the top-K (~30–50) with a
   cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly attends over
   the `(JD, candidate)` pair. Replace or blend with `similarity_score` (the 0.45-weight
   component). *Pro:* far better pairwise relevance than independent embeddings; O(K)
   passes is tractable at K ≤ 50. *Con:* slower; depends on good JD/candidate text.
2. **LLM re-ranker at the head (listwise, top-K).** After component scoring narrows to
   ~top-20, pass the JD + candidate profiles to an LLM for a listwise ranking / 0–10 score
   (essentially what "Contra6" did to build the silver labels). *Pro:* captures level-fit,
   sector, and language nuance the current components miss. *Con:* cost/latency/
   nondeterminism, and **circularity** — it cannot be validated against LLM-generated
   labels, so it **requires human-labelled JDs** first. Do last.
3. **Semantic component scoring (replace/augment fuzzy).** Skills: the specific-JD run
   showed fuzzy matching underperforms on specialised terminology (WPS, MOHRE, Bayzat) —
   add embedding-based skill matching (cosine between JD-skill and candidate-skill
   embeddings, thresholded) or a small skill-pair cross-encoder; ablate against the current
   `rapidfuzz` + alias approach and guard against semantic false positives on short skill
   strings. Title is already hybrid (fuzzy ∨ semantic); consider dropping the fuzzy leg.

**Recommended sequence:** (0) expand the gold set with **human** labels (prerequisite for
validating any re-ranker, mandatory before the LLM re-ranker) → (1) cross-encoder re-ranker
(no circularity, likely solid NDCG gain, moderate effort) → (2) semantic skill matching
(ablate carefully) → (3) LLM re-ranker (highest cost; only once human labels exist).

### Backlog — noted observations (not yet scheduled)

- **Interactive JD extraction (human-in-the-loop).** JD extraction is the pipeline's *entry
  point* and is currently a single LLM pass. Make it a back-and-forth Q&A with the recruiter to
  confirm the extracted features and — critically — the **priority** of each requirement, so the
  structured JD captures hiring intent fully before any scoring happens. Input quality here caps
  the entire pipeline.
- **Location scoring component.** `CandidateLocation` (city/country) is extracted but never
  scored. Add an optional, weight-gated `location_score` (city match > country match > mismatch).
  Deferred earlier only because this pool is 144/145 UAE; still worth adding for
  location-diverse JDs.
- **Industry / sector scoring — scale beyond the hardcoded alias list.** `industry_score`
  (alias-based, `core/industry_normalization.py`) is implemented and adopted (weight 0.20) and
  measurably helps (gold NDCG@10 +~5%, blind-corroborated). BUT `INDUSTRY_ALIASES` is a
  hand-maintained, **non-exhaustive** dict — fine for the single-client MVP (Prime Focus =
  HVAC / manufacturing) and useful to *prove* sector matching helps, yet it does **not scale**
  across clients/industries (O(clients × industries) upkeep, always lagging). The LinkedIn export
  has **no structured industry field** (confirmed), so the candidate sector is derived from
  employer/role text. Scale path — do NOT hand-grow the list: classify candidate sector into a
  **fixed taxonomy** (~25 sectors, or GICS / LinkedIn's ~150 industries) via a cached LLM call at
  ingestion; map the LLM-extracted JD `industry` into the same taxonomy; match by taxonomy
  (hierarchical, e.g. HVAC ⊂ manufacturing) instead of aliases. Alternative: enrich at source
  (scrape LinkedIn `companyIndustry`). Keep the alias dict only as a cheap offline fallback.
  - **Matching approaches evaluated (2026-07):** current = **whole-word alias regex** (chosen;
    high precision — a match means confident sector evidence, which is *why* the component works).
    **Rejected — semantic fallback** (embed JD industry labels vs candidate text when no alias
    matches): short category strings embed unreliably and its noise leaks partial sector-credit to
    *wrong-sector* profiles, diluting precision; also partly double-counts `similarity_score`.
    **Rejected — fuzzy match**: short industry tokens collide at char level against free text
    (`duct`~`duty`, `mep`~`map`, `steel`~`steal`, `manufacturing`~`management`), reintroducing the
    substring false positives the whole-word fix removed. Both are net-negative on precision; the
    recall gap (missing aliases) is bounded (industry is ~15% of the blend, a dent not a knockout)
    and best closed by the taxonomy classifier above — not fuzzy/semantic. Missing morphological
    variants should be added as explicit aliases (deterministic), not matched fuzzily.

## Scoring API on Render (memory & ops — 2026-08-04)

Working production config for **contra6-scoring-api** on Render Standard (2GB):

| Knob | Value | Why |
|------|-------|-----|
| `BASE_EMBEDDING_DTYPE` | `fp16` | Halves mpnet resident weights vs fp32 |
| `SIMILARITY_MODEL` | `mpnet-only` | Drops Qwen (~1.1GB weights); title/skill/similarity share mpnet |
| Idle RSS (measured) | **~964 MB** | Post warm + encode-prime on Linux |
| Successful `/score` peak (n=10) | **~1017 MB** | ~+53 MB climb |

**Dual-model “champion” (Qwen similarity) is not viable on this tier** — local idle for fp32 mpnet+Qwen was ~2.1GB before any request. Keep Qwen for larger plans / eval machines only (`SIMILARITY_MODEL=qwen`).

### Candidate-count ceiling (local load test, 2026-08-04)

Fresh-process runs of the instrumented pipeline (fp16 mpnet-only, real LinkedIn-length profiles cloned to N):

| N | Local climb (peak − baseline) | Embed stage Δ |
|--:|------------------------------:|--------------:|
| 10 | ~43 MB | ~34 MB |
| 25 | ~48 MB | ~42 MB |
| 40 | ~48 MB | ~39 MB |

Climb is **sub-linear** in N: encoder workspace is mostly a fixed cost; retaining more embedding vectors adds little. Soft limit uses a **conservative** linearisation of the production climb (`53 MB / 10 ≈ 5.3 MB/cand`), which would approach 2GB−150MB reserve near **n≈176**. Product soft max is **`SCORE_MAX_CANDIDATES=100`** (HTTP 422 on overflow) so a large batch fails cleanly instead of OOM-killing the instance.

### Startup race — ruled out

Uvicorn 0.52 runs `lifespan` startup (including `warm_scoring_models()`) **before** `create_server()` binds the listen port. Port-open cannot overlap model load/encode-prime. `/score` also gates on `_models_ready` (503 while warming). Do not re-investigate “request arrived during warm-up” unless the uvicorn version or start command changes.

### JD parse cache

`process_jd` writes/reads **sha256(jd_text)** files under `JD_CACHE_DIR` (default `.ai-recruiter/jd_cache`). Changing the JD text misses the cache. Optional request field `parsed_jd` lets Sourcing_Apify skip Claude entirely on repeat scores. Render disk is ephemeral across deploys — fine for same-instance re-scores.

### Alerting / watchdog (ops checklist)

1. **Render Dashboard → Service → Settings → Notifications** — enable deploy failure + service failure emails (or Slack webhook).
2. **External uptime** (UptimeRobot or similar, outside this repo): poll `GET /health` every 5 minutes. Treat as down unless JSON has `"status":"ok"` **and** `"models_ready":true`. `startup_process_rss_mb` is informational (baseline after warm); a sudden jump toward ~1800+ is an early warning, not a hard fail.
3. Prefer clean **422 batch-too-large** over instance OOM — OOMs force multi-minute redeploys and drop in-flight work.
