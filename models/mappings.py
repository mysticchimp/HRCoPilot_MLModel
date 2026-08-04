from models.enums import ImportanceLevel

# Shared encode micro-batch for title / skill / profile / similarity paths.
# ST pads to the longest sequence in a batch — keep this small on 2GB hosts.
encode_batch_size = 2

# Title/skill (and mpnet-only similarity) encoder dtype. fp16 via torch_dtype at
# construction — same pattern as Qwen — roughly halves resident weight RSS vs the
# previous default SentenceTransformer() fp32 load. Override with BASE_EMBEDDING_DTYPE.
base_embedding_dtype = "fp16"

attribute_weight_by_importance = {
    ImportanceLevel.ESSENTIAL: 1.0,
    ImportanceLevel.IMPORTANT: 0.7,
    ImportanceLevel.VALUABLE: 0.4,
    ImportanceLevel.SUPPLEMENTARY: 0.2
}

degree_norm_map = {
    "bachelor": [
        "b.e", "be", "b.tech", "btech", "b.sc", "bsc", "b.a", "ba", "b.com", "bcom",
        "bba", "bca", "b.arch", "barch", "b.pharm", "bpharm", "b.ed", "bed", "bfa",
        "b.des", "bdes", "b.lit", "blit", "b.s", "bs", "b.eng", "beng", "b.engg", "bengg",
        "bachelors", "bachelor", "undergraduate", "ug", "licentiate", "bacharelado"
    ],
    "master": [
        "m.e", "me", "m.tech", "mtech", "m.sc", "msc", "m.a", "ma", "m.com", "mcom",
        "mba", "mca", "m.arch", "march", "m.pharm", "mpharm", "m.ed", "med", "mfa",
        "m.des", "mdes", "m.lit", "mlit", "m.s", "ms", "m.eng", "meng", "m.engg", "mengg",
        "masters", "master", "postgraduate", "pg", "post grad", "post-graduate", "magister"
    ],
    "phd": [
        "ph.d", "phd", "d.phil", "dphil", "doctorate", "doctoral", "dr.", "dr",
        "sc.d", "scd", "eng.d", "engd", "edd", "dba"
    ],
    "associate": [
        "associate", "a.a", "aa", "a.s", "as", "a.sc", "asc", "a.a.s", "aas"
    ],
    "diploma": [
        "diploma", "advanced diploma", "postgraduate diploma", "pg diploma", "pgd", "post diploma",
        "polytechnic", "certificate", "post-baccalaureate diploma", "post baccalaureate diploma"
    ],
    "highschool": [
        "high school", "secondary school", "hsc", "ssc", "10th", "12th", "intermediate", "matriculation",
        "matric", "senior secondary", "pre-university", "pu", "pu college", "a-levels", "alevels", "o-levels", "olevels"
    ],
    "postdoc": [
        "postdoc", "post doctoral", "post-doctoral", "post doctorate", "post-doctorate"
    ],
    "vocational": [
        "certificate course", "vocational", "trade school", "training", "bootcamp"
    ]
}

# ordinal rank of each normalized degree category (higher = more advanced).
# categories at the same level share a rank (e.g. vocational ~ diploma).
# unknown / unmapped degrees resolve to -1 and never satisfy a requirement.
degree_rank_map = {
    "highschool": 0,
    "diploma": 1,
    "vocational": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
    "postdoc": 6,
}

# need to adjust weights after eval
candidate_score_weights = {
    'title_score': 0.25,
    'skill_score': 0.25,
    'qualification_score': 0.05,
    'similarity_score': 0.45,
    'seniority_score': 0.05,  # ADOPTED (Phase 4 #4a): strict Pareto win at 0.05 on LinkedIn eval
    'experience_score': 0.05,  # ADOPTED (Phase 4 #4b): NDCG@10 gain saturates at 0.05; higher only chases leaked MRR
    'industry_score': 0.20,  # ADOPTED: sector-fit (whole-word alias match); gold NDCG@10 peak (.6020->.6288) at 0.20 post alias-correction
    'language_score': 0.15,  # T3 ADOPTED (2026-07-31): explicit-declaration-only language signal. New silver-anchor NDCG@10 .9490->.9588 (+.00975) vs C5-neutral control; NDCG@5 .9356->.9464; reverse unchanged after language de-leak. General-language result only (Tagalog-specific n=1).
    'location_score': 0.05,  # ADOPTED (product-motivated, 2026-07-12): country>city fit for an on-site role. Reverse-match de-leaked for location (it invents a seed-mismatched location, can't fairly test it), so at 0.05 gold NDCG@10 holds (.6334) and reverse is unchanged -> eval-neutral; weight reflects a real recruiter constraint, not an eval-proven gain.
    'attrition_score': 0.005,  # PRODUCT-MOTIVATED (2026-07-31): gentle tenure tie-breaker. In the joint product triplet, Judge NDCG@5/10 holds, NDCG@20 improves, reverse MRR rises .5190->.5212; normalized share is ~0.34% on this JD.
    'experience_relevance_score': 0.015,  # PRODUCT-MOTIVATED: strongest structural leg (construct corr +.72); joint triplet preserves NDCG@5/10 and drives the NDCG@20 gain. Normalized share ~1.02% on this JD.
    'education_relevance_score': 0.005,  # PRODUCT-MOTIVATED: minimal soft tie-breaker, not a gate. Prior standalone .03 was rejected; joint .005 has positive leave-one-out contribution without changing NDCG@5/10.
}

# candidate seniority vs JD required level share this ordinal scale.
seniority_rank_map = {
    "entry": 0,
    "mid": 1,
    "senior": 2,
    "executive": 3,
    "c_level": 4,
}
# per-level penalty for a seniority gap; under-qualification hurts more than
# over-qualification.
seniority_under_penalty = 0.4
seniority_over_penalty = 0.25

# per-year penalty for a years-of-experience gap vs the JD range; a shortfall
# (under the minimum) hurts more than being over the top of the range.
experience_under_penalty_per_year = 0.15
experience_over_penalty_per_year = 0.05

# location scoring (country = primary gate, city = refinement). Full credit for a
# matching city OR an in-country candidate whose city is unspecified (an omitted
# city is not penalized — benefit of the doubt once the country matches). A
# confirmed DIFFERENT city inside the right country earns partial credit; a wrong
# country is a miss. A country-only JD gives any in-country candidate full credit.
location_city_match = 1.0
location_country_match = 0.7  # right country, confirmed different city
location_mismatch = 0.0

# tenure / attrition (flight-risk) scoring (C5 P2). Product-motivated and
# JD-independent (like location): job stability always matters. The score is based
# on the MEDIAN completed-PERMANENT tenure in months (the current role and
# contractors/interns are excluded upstream in core/positions.py, so a short
# *current* stint never reads as flight-risk and short contracts are expected).
# The bands are a documented starting shape; the ablation tunes the WEIGHT, not
# these numbers. First band whose threshold the median meets wins.
attrition_tenure_bands = [
    (24, 1.00),   # >= 24 mo median -> very stable
    (18, 0.85),
    (12, 0.65),
]
attrition_short_stint_months = 12       # a permanent role under this is a "short stint"
attrition_chronic_min_short_roles = 3   # median < 12 AND >= this many short perm roles => chronic hopper
attrition_chronic_hop_score = 0.30      # chronic job-hopper
attrition_short_stint_score = 0.50      # median < 12 but not chronic (a one-off short stint)
attrition_neutral = 0.5                 # too little history to assess (< 2 dated roles / no completed-perm role)
attrition_early_career_years = 3        # JD is entry-level (1-4 yrs): short histories are normal for juniors
attrition_early_career_max_roles = 2    # the early-career floor only applies to genuinely short histories
attrition_early_career_floor = 0.70     # don't punish juniors below this

# relevant-vs-adjacent experience classification (C5 P3). Each role's TITLE is
# classified RELEVANT (full credit), ADJACENT (partial credit) or UNRELATED (none),
# word-boundary matched with precedence RELEVANT > ADJACENT. The score is the
# tenure-weighted ratio (relevant + adjacent_credit*adjacent) / total dated tenure.
# Keyword-based first cut (spec 5.1); the ablation tunes the WEIGHT, not these lists.
relevant_role_keywords = [
    "hr", "human resource", "human resources", "recruit", "talent",
    "payroll", "people ops", "personnel", "hris",
]
adjacent_role_keywords = [
    "admin", "administrative", "coordinator", "operations", "executive assistant",
    "pro", "government relations", "office",
]
experience_relevance_adjacent_credit = 0.5   # partial credit for an adjacent role
experience_relevance_neutral = 0.5           # no dated/titled roles -> neutral

# education relevance (C5 P4) — a SOFT tie-breaker, never a gate and never below the
# 0.5 neutral floor (the JD requires no degree). score = max credential tier across
# the candidate's degree fields + certs. Substring-matched (field-of-study phrases);
# precedence RELEVANT (directly HR) > BUSINESS (adjacent) > neutral. Distinct from the
# gate-style qualification_score. Weight tuned by ablation.
education_relevant_fields = [
    "human resource", "human resources", "personnel", "psycholog", "law", "legal", "labour", "labor law",
]
education_business_fields = [
    "business administration", "business management", "management", "commerce", "business",
    "bba", "mba", "economic", "finance", "accounting", "international business",
]
education_hr_certs = ["cipd", "chrp", "shrm", "aphr", "phr", "sphr"]
education_relevant_score = 1.00   # directly-HR degree or an HR cert
education_business_score = 0.75   # business-adjacent degree
education_neutral = 0.50          # unrelated degree only, or no education data (never below neutral)

# data-completeness flag (C5 P6) — a separate output annotation, NOT part of the score.
completeness_min_skills = 5       # < this many skills contributes to a 'low' data flag

# skill matcher: semantic cosine floor for the hybrid skill matcher. Below this a
# semantic skill pair contributes nothing, guarding against embedding false
# positives (the same precision concern that ruled out semantic industry matching).
# Tuned via scripts/calibrate_weights.py --skill-mode hybrid on gold NDCG@10:
# all-mpnet compresses skill-phrase similarity (genuine synonyms ~0.45), and 0.50
# dropped gold below its floor, so 0.40 is the adopted floor.
skill_semantic_threshold = 0.40

# Isolated similarity_score embedding model (Option B, adopted 2026-07-13). None =>
# use the base model (all-mpnet) for similarity too (pre-upgrade behavior). This
# overrides ONLY the 0.45-weight similarity component; title/skill keep all-mpnet.
# Qwen3-Embedding-0.6B (95% zero-shot, apache-2.0) captures the ~44% of LinkedIn
# profiles that exceed all-mpnet's 384-token cap; the query instruction is its trained
# asymmetric-retrieval prompt (JD=query). fp16 via torch_dtype at load + L1024 keep it
# within a 16GB MPS budget (no post-load .half() RSS spike).
# batch_size=2: real LinkedIn about+experience texts often hit the 1024-token cap;
# encoding a full request (10+) in one padded batch spikes RSS past a 2GB Render box.
#
# Runtime toggle (scoring API / warm_scoring_models): env SIMILARITY_MODEL=
#   qwen|champion (default) → this dict
#   mpnet-only|mpnet|none    → None (single-model baseline; drops Qwen from RSS)
similarity_model_config = {
    "model_name": "Qwen/Qwen3-Embedding-0.6B",
    "query_instruction": "Instruct: Given a job description, retrieve candidate profiles that best match the role.\nQuery: ",
    "doc_instruction": None,
    "dtype": "fp16",
    "device": None,
    "max_seq_length": 1024,
    "batch_size": encode_batch_size,  # see encode_batch_size above
}

# Stage-2 cross-encoder reranker (docs/adr/0001-cross-encoder-reranker.md). Two-stage
# retrieve→rerank: Stage 1 (the component pipeline) ranks the full pool; the top
# `rerank_top_k` Head is then re-scored by this cross-encoder, whose score REPLACES the
# bi-encoder similarity_score in the 0.45 fusion slot WITHIN the Head only (frozen
# membership — the tail keeps its Stage-1 order). See core.scoring.apply_rerank.
#
# rerank_model_config is None (reranking OFF) until it clears MEASURED adoption: the CE
# Head must beat the bi-encoder Head on the frozen blind-judge NDCG@10 (+ Kendall tau)
# without regressing the guardrails. To A/B it, pass RERANK_GTE to build_rerank_model
# (run_hr_assistant / run_eval / calibrate --rerank); once it wins, set
# rerank_model_config = RERANK_GTE. gte-reranker-modernbert-base: 149M ModernBERT
# (MPS-safe), 8192 ctx, apache-2.0. temperature = sigmoid(logit / T); raise if the Head saturates.
RERANK_GTE = {
    "model_name": "Alibaba-NLP/gte-reranker-modernbert-base",
    "max_length": 2048,
    "dtype": "fp32",   # fp16 NaNs on Apple MPS for ModernBERT (guarded); fp32 runs clean
    "device": None,
    "batch_size": 16,
    "temperature": 1.0,
}
rerank_model_config = None      # champion: OFF (measured 2026-07-16: gte HURT head NDCG@10 .95->.79 vs judges; SHELVED)
rerank_top_k = 50               # Head size the reranker reorders (validated via recall@K; K=30 misses judge-graded cands)
