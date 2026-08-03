from rapidfuzz import process, fuzz
from models.mappings import degree_norm_map

TERM_TO_CATEGORY = {
    term: category
    for category, terms in degree_norm_map.items()
    for term in terms
}

all_terms = list(TERM_TO_CATEGORY.keys())


def normalize_degree(degree: str, threshold=80):
    if not isinstance(degree, str):
        return degree

    degree_lower = degree.lower().strip()
    if not degree_lower:
        return degree_lower

    # exact match (case-insensitive) first
    category = TERM_TO_CATEGORY.get(degree_lower)
    if category:
        return category

    # very short, non-exact inputs (e.g. "ba", "be") are too ambiguous to
    # fuzzy-match safely, so leave them unnormalized (lowercased for consistency).
    if len(degree_lower) <= 3:
        return degree_lower

    # fuzzy fallback on whole tokens. token_set_ratio matches subsets
    # ("btech engineering" -> "btech") without the substring false positives of
    # partial_ratio (which e.g. matches "ba" inside "mba").
    match, score, _ = process.extractOne(
        degree_lower, all_terms, scorer=fuzz.token_set_ratio
    )
    if match and score >= threshold:
        return TERM_TO_CATEGORY[match]

    # unrecognized: return the normalized-case string (rank stays -1 either way)
    return degree_lower
