import re

import numpy as np
from rapidfuzz import fuzz


# Deliberately small, explicit taxonomy. Add aliases only when they are genuine
# equivalents; broader relationships (e.g. recruitment vs talent management)
# should be learned/handled by a semantic matcher, not silently conflated here.
SKILL_ALIASES = {
    "hr": "human resources",
    "human resource": "human resources",
    "human resources hr": "human resources",
    "human resource information system": "hris",
    "human resources information system": "hris",
    "human resource information systems": "hris",
    "human resources information systems": "hris",
    "human resources information systems hris": "hris",
    "hris platform": "hris",
    "hris platforms": "hris",
    "ms office": "microsoft office",
    "microsoft office suite": "microsoft office",
    "ms excel": "microsoft excel",
    "excel": "microsoft excel",
    "ms word": "microsoft word",
    "word": "microsoft word",
    "ms powerpoint": "microsoft powerpoint",
    "powerpoint": "microsoft powerpoint",
    "ms outlook": "microsoft outlook",
    "outlook": "microsoft outlook",
    "recruiting": "recruitment",
    "organisational skills": "organizational skills",
    "organization skills": "organizational skills",
    "payroll administration": "payroll",
    "payroll processing": "payroll",
    "employee record management": "employee records management",
    "admin assistance": "administrative assistance",
    "admin support": "administrative support",
}


def normalize_skill(skill: str) -> str:
    """Canonicalize a skill while preserving meaningful symbols (C++, C#, .NET)."""
    if not isinstance(skill, str):
        return ""
    value = skill.casefold().strip().replace("&", " and ")
    value = value.replace("labour", "labor").replace("organisational", "organizational")
    value = re.sub(r"[()\[\]{},:/_-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    # "X knowledge" is the same skill requirement as "X"; jurisdictional words
    # remain (UAE labor law does not collapse to generic labor law).
    value = re.sub(r"\s+knowledge$", "", value).strip()
    return SKILL_ALIASES.get(value, value)


def skill_similarity(skill_a: str, skill_b: str) -> float:
    """Similarity in [0,100], with exact aliases and short-token safeguards."""
    left = normalize_skill(skill_a)
    right = normalize_skill(skill_b)
    if not left or not right:
        return 0.0
    if left == right:
        return 100.0

    # Acronyms/languages/tools of <=3 chars are too collision-prone for fuzzy
    # matching (R/HR/Go/SQL/AWS). They must be exact or explicitly aliased.
    if min(len(left), len(right)) <= 3:
        return 0.0

    # Prevent a single-token skill from matching a longer token merely because it
    # is a substring (Java vs JavaScript). Character ratio alone is safer too, but
    # this makes the intended invariant explicit.
    if " " not in left and " " not in right and (left in right or right in left):
        return 0.0

    return float(fuzz.ratio(left, right))


class SkillSemanticIndex:
    """Precomputed unit-norm skill embeddings, keyed by normalized skill string.

    Built once per candidate pool so the hybrid skill matcher never calls the
    embedding model inside its per-candidate loop; matching is a cheap dot product
    (cosine, because vectors are unit-normalized).
    """

    def __init__(self, embeddings: dict[str, np.ndarray]):
        self._embeddings = embeddings

    def similarity(self, skill_a: str, skill_b: str) -> float:
        """Cosine similarity in [0, 1] between two skills (0.0 if either is unknown)."""
        va = self._embeddings.get(normalize_skill(skill_a))
        vb = self._embeddings.get(normalize_skill(skill_b))
        if va is None or vb is None:
            return 0.0
        return max(0.0, float(np.dot(va, vb)))


def build_skill_semantic_index(skills, model) -> SkillSemanticIndex:
    """Embed every unique normalized skill once (unit-normalized) for cosine lookups."""
    unique = sorted({
        normalize_skill(s)
        for s in skills
        if isinstance(s, str) and normalize_skill(s)
    })
    if not unique:
        return SkillSemanticIndex({})
    vectors = model.encode(unique, convert_to_numpy=True, normalize_embeddings=True)
    return SkillSemanticIndex({skill: vec for skill, vec in zip(unique, vectors)})
