import numpy as np
from rapidfuzz import fuzz, process, utils

from core.normalization import normalize_degree
from core.skill_normalization import normalize_skill, skill_similarity
from models.data_models import Education, Skill
from models.mappings import attribute_weight_by_importance, degree_rank_map


def fuzzy_match(set1: list[str], set2: list[str], scorer=fuzz.token_set_ratio):
    return process.cdist(
        set1, 
        set2, 
        scorer=scorer,
        processor=utils.default_process
    )


def weighted_fuzzy_skill_score(
    candidate_id: str,
    jd_skills: list[Skill],
    candidate_skills: list[str],
    score_threshold: float = 70,
    semantic_index=None,
    semantic_threshold: float = 0.55,
    include_fuzzy: bool = True,
):
    if not jd_skills or not candidate_skills:
        return {
            "score": 0.0,
            "matched_skills": []
        }

    # Collapse duplicate JD skills/technologies after canonicalization, keeping
    # the highest priority. This avoids counting the same requirement twice.
    requirements = {}
    for skill in jd_skills:
        canonical = normalize_skill(skill.skill)
        if not canonical:
            continue
        weight = attribute_weight_by_importance.get(skill.priority, 0.0)
        if canonical not in requirements or weight > requirements[canonical]:
            requirements[canonical] = weight

    normalized_candidates = [
        (original, normalize_skill(original))
        for original in candidate_skills
        if isinstance(original, str) and normalize_skill(original)
    ]

    candidate_score = 0.0
    matched_skills = []

    total_possible_score = sum(requirements.values())

    if total_possible_score == 0:
        return {
            "score": 0.0,
            "matched_skills": []
        }

    def _match_strength(jd_skill_name: str, candidate_norm: str) -> float:
        # Hybrid per-pair strength in [0, 1]. Fuzzy (char-level, guarded) covers
        # spelling/aliases; semantic (embedding cosine) covers synonyms the fuzzy
        # matcher is blind to (talent acquisition~recruitment, WPS~wage protection).
        # max() mirrors the title matcher; each channel is gated by its own floor.
        strength = 0.0
        if include_fuzzy:
            fuzzy = skill_similarity(jd_skill_name, candidate_norm)
            if fuzzy >= score_threshold:
                strength = fuzzy / 100.0
        if semantic_index is not None:
            sem = semantic_index.similarity(jd_skill_name, candidate_norm)
            if sem >= semantic_threshold:
                strength = max(strength, sem)
        return strength

    for jd_skill_name, jd_weight in requirements.items():
        strengths = [_match_strength(jd_skill_name, candidate) for _, candidate in normalized_candidates]
        if not strengths:
            continue
        best_idx = int(np.argmax(strengths))
        best_strength = strengths[best_idx]

        if best_strength > 0.0:
            # Graded contribution: a 0.79-strength match contributes 0.79 of the
            # skill's priority weight (fuzzy ratio/100 or semantic cosine).
            candidate_score += jd_weight * best_strength
            matched_skill = normalized_candidates[best_idx][0]
            if matched_skill not in matched_skills:
                matched_skills.append(matched_skill)

    return {
        "score": candidate_score / total_possible_score,
        "matched_skills": matched_skills
    }


def weighted_fuzzy_qualification_score(candidate_id: str, jd_qualifications: list[Education] | None, candidate_qualifications, score_threshold=60):
    if not jd_qualifications or not candidate_qualifications:
        return {
            "score": 0.0,
            "matched_qualifications": []
        }

    def degree_rank(degree):
        return degree_rank_map.get(degree, -1)  # unknown -> not eligible

    matched_qualifications = []
    candidate_score = 0.0
    total_possible_score = sum(attribute_weight_by_importance[s.priority] for s in jd_qualifications)
    if total_possible_score == 0:
        return {
            "score": 0.0,
            "matched_qualifications": []
        }
    
    candidate_degrees = candidate_qualifications["degrees"]
    candidate_fields = candidate_qualifications["fields"]
    candidate_degrees_norm = [normalize_degree(deg) for deg in candidate_degrees]

    for jd_qualification in jd_qualifications:
        jd_degree_norm = normalize_degree(jd_qualification.degree)
        jd_degree_rank = degree_rank(jd_degree_norm)

        # find candidate degrees that meet or exceed the min required degree
        eligible_indices = [
            idx for idx, cand_deg_norm in enumerate(candidate_degrees_norm)
            if degree_rank(cand_deg_norm) >= jd_degree_rank and jd_degree_rank != -1
        ]

        if not eligible_indices:
            continue

        jd_weight = attribute_weight_by_importance.get(jd_qualification.priority, 0.0)

        # Compare JD field against eligible candidate fields
        eligible_fields = [candidate_fields[idx] for idx in eligible_indices]

        if jd_qualification.field:
            match_scores = fuzzy_match([jd_qualification.field], eligible_fields)
            scores = match_scores[0]

            best_idx = np.argmax(scores)
            best_score = scores[best_idx]

            if best_score >= score_threshold:
                candidate_score += jd_weight

                matched_degree = candidate_degrees[eligible_indices[best_idx]]
                matched_field = candidate_fields[eligible_indices[best_idx]]

                matched_qualifications.append(f"{matched_degree} in {matched_field}")
        else:
            # degree-only requirement (no field specified): meeting the required
            # degree level is sufficient. Credit the highest-ranked eligible degree.
            candidate_score += jd_weight

            best_eligible_idx = max(
                eligible_indices,
                key=lambda idx: degree_rank(candidate_degrees_norm[idx])
            )
            matched_degree = candidate_degrees[best_eligible_idx]
            matched_field = candidate_fields[best_eligible_idx]

            if matched_field and str(matched_field) != 'N/A':
                matched_qualifications.append(f"{matched_degree} in {matched_field}")
            else:
                matched_qualifications.append(f"{matched_degree}")

    return {
        "score": candidate_score / total_possible_score,
        "matched_qualifications": matched_qualifications
    }
