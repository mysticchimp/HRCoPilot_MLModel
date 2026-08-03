import glob
import os

from sentence_transformers import SentenceTransformer

from core.adapters.linkedin_adapter import LinkedInAdapter
from core.embedding import build_similarity_spec
from evals.cases import load_fixture
from evals.runner import PipelineConfig, aggregate, evaluate_cases
from models.mappings import similarity_model_config

LINKEDIN = "data/Raw_dataset_linkedin-profile-search_HR_Assistant_2026-07-06_16-31-55-822.csv"

# The champion pipeline config + metric FLOORS on the committed LinkedIn fixtures.
# RATCHET: whenever an adopted improvement raises the metrics, bump these floors UP
# so the gain can't silently regress. Never lower a floor for a code change.
# EXCEPTIONS (one-time methodology resets, 2026-07, NOT code regressions):
#   1. Reverse-match fixtures regenerated to hold seniority/years out of JD
#      generation (leakage removal) -> reverse metrics legitimately dropped.
#   2. Gold JD repointed from a placeholder to the real posting the fit_0_10 grades
#      were produced against -> gold NDCG@10 corrected from an inflated .70 to the
#      honest .60. Floors were reset once to the de-leaked, gold-corrected champion.
#   3. Skill matcher upgraded fuzzy -> hybrid (semantic cosine floor 0.40). At the
#      unchanged skill weight (0.25) gold NDCG@10 held and reverse MRR/hit@10 rose,
#      so these floors were ratcheted UP to the hybrid champion.
#   4. Location component adopted (weight 0.05, product-motivated) with a reverse-match
#      de-leak: reverse JDs invent a location NOT tied to the seed (it is held out of
#      generation), which unfairly penalized the seed on location_score, so the reverse
#      fixtures were re-saved with parsed_jd.location=null (reverse-match cannot fairly
#      evaluate location; the gold posting is the anchor). At 0.05 gold NDCG@10 holds
#      (.6334) and reverse is unchanged, so NO floor moved -> this records the
#      methodology change, not a floor reset.
#   5. Similarity embedding upgraded all-mpnet -> Qwen3-Embedding-0.6B (2026-07-13),
#      ISOLATED to similarity_score (title/skill keep all-mpnet). PRODUCT-MOTIVATED:
#      all-mpnet truncates 44% of profiles at its 384-token cap and future JDs will
#      exceed it too. The n=1 gold JD is short (346 tokens) so the eval cannot fairly
#      measure the long-context benefit: gold NDCG@10 is only nominally up (.6334->.6522,
#      within n=1 noise) while NDCG@5 slipped (.6165->.6009), and the reverse-match MRR/
#      hit gains are leakage-suspect. So floors are NOT ratcheted (mirrors #4) — the
#      Qwen champion simply must keep clearing the established honest floors.
#   6. Gold labels repointed single-LLM fit_0_10 (Scored_FullPool) -> blind 2-judge
#      consensus (judge_mean_score, evals/judgments/blind_judgments_hr_assistant.csv), expanded to
#      the blind top-50 union (78 graded; pipeline top-20 fully covered). Stronger silver
#      anchor: 2 independent frontier judges (agreement Spearman .908) vs one LLM; the
#      pipeline agrees with it NDCG@10 ~.92 vs only ~.65 on the old fit_0_10 labels. The
#      label SCALE (0-10 -> 0-100) AND the anchor changed, so the gold floors are RESET
#      (methodology change, NOT a code gain): ndcg@10 .63 -> .90 (+ ndcg@5 added at .90).
#      Reverse fixtures/floors are untouched; the champion pipeline itself did NOT change.
#   7. T3 regraded the same frozen 78-candidate cohort with a neutral 9-section rubric
#      that explicitly covers tenure, career relevance, and workforce language; the
#      canonical parsed JD was also synchronized (fixing Tagalog and other stale fields).
#      This changes every silver Judge grade, so gold floors RESET from the NEW incumbent
#      (education .03 still active): raw ndcg@5 .93449 / ndcg@10 .94831 -> floors .93/.94
#      (truncate to .01). Reverse language requirements were de-leaked because seed
#      languages are held out; incumbent reverse metrics were unchanged, so reverse
#      floors stay fixed. C5 then selected language .15 and removed education .03, but
#      floors are NOT ratcheted to that circular n=1 gain.
# Product override (2026-07-31, not a methodology reset): attrition .005,
# experience_relevance .015, and education_relevance .005 were added as gentle
# recruiter tie-breakers. NDCG@5/10 is unchanged, NDCG@20 improves .9289->.9518,
# and reverse MRR improves .5190->.5212, so floors do not move.
CHAMPION_CONFIG = PipelineConfig(title_mode="hybrid", title_hard=False, skill_mode="hybrid")
FLOORS = {
    "seed_found_rate": 1.0,
    "hit@3": 0.42,
    "hit@5": 0.47,
    "hit@10": 0.68,
    "mrr": 0.41,
    "ndcg@5": 0.93,
    "ndcg@10": 0.94,
}


def _load_committed_cases():
    keys = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob("evals/fixtures/linkedin/*.json")]
    return [case for case in (load_fixture("linkedin", key) for key in keys) if case]


def test_linkedin_regression_gate():
    cases = _load_committed_cases()
    assert len(cases) >= 15, "committed LinkedIn fixtures are missing"

    profiles = LinkedInAdapter().to_profiles(LINKEDIN)
    model = SentenceTransformer("all-mpnet-base-v2")
    # Guard the ACTUAL champion: title/skill on all-mpnet + similarity on the isolated
    # champion model (Qwen3-Embedding-0.6B). sim_spec is None if the config is disabled.
    sim_spec = build_similarity_spec(similarity_model_config, base_model=model)
    summary = aggregate(evaluate_cases(cases, profiles, model, CHAMPION_CONFIG, sim_spec=sim_spec))

    for metric, floor in FLOORS.items():
        assert summary[metric] >= floor - 1e-9, (
            f"REGRESSION: {metric}={summary[metric]:.4f} dropped below floor {floor}"
        )
