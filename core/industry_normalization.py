"""Industry / sector matching (alias-based).

Candidates have no structured industry field, so sector fit is inferred by
checking whether a JD industry's alias keywords appear in the candidate's
derived `sector_text` (employers + title + role descriptions). This is
deliberately alias/keyword matching, not embedding similarity, to stay
interpretable and avoid double-counting the semantic `similarity_score`.
"""

import re
from functools import lru_cache

from models.enums import ImportanceLevel

# canonical sector -> alias keywords/phrases to look for in candidate text.
# Matching is WHOLE-WORD (with an optional trailing 's'), so short roots are safe
# from substring collisions (e.g. "duct" no longer matches "product"/"conducted").
# Curated toward this client's world (HVAC / manufacturing / MEP / construction /
# building materials); other sectors are lighter. This dict is an MVP — see
# ARCHITECTURE.md for the taxonomy-classification scale path.
INDUSTRY_ALIASES = {
    "hvac": [
        "hvac", "hvacr", "ductwork", "ducting", "ducts", "sheet metal", "air conditioning",
        "air-conditioning", "air handling", "ventilation", "refrigeration", "chiller", "chillers",
        "climate control", "fan coil", "vrf", "vrv", "ahu", "fahu",
    ],
    "manufacturing": [
        "manufacturing", "manufacturer", "manufacture", "factory", "factories", "fabrication",
        "fabricator", "assembly line", "production line", "industrial", "machining", "cnc",
        "welding", "welder", "metal works", "metalwork", "foundry", "mfg",
    ],
    "mep": [
        "mep", "electromechanical", "mechanical electrical", "mechanical and electrical",
        "plumbing", "mechanical contracting", "electrical contracting",
    ],
    "construction": [
        "construction", "contracting", "contractor", "subcontractor", "sub-contractor",
        "civil engineering", "civil works", "infrastructure", "structural", "site works",
        "fit-out", "fitout", "joinery", "scaffolding", "formwork", "turnkey", "epc",
        "building construction", "building maintenance",
    ],
    "building materials": [
        "building materials", "precast", "pre-cast", "ready mix", "readymix", "ready-mix",
        "concrete", "cement", "aggregates", "rebar", "structural steel", "steel fabrication",
        "aluminium", "aluminum", "gypsum",
    ],
    "facilities management": [
        "facilities management", "facility management", "fm services", "soft services",
        "hard services", "mep maintenance",
    ],
    "oil and gas": ["oil and gas", "oil & gas", "petroleum", "refinery", "petrochemical", "offshore", "drilling"],
    "logistics": ["logistics", "supply chain", "freight", "shipping", "warehousing", "forwarding"],
    "retail": ["retail", "fmcg", "consumer goods", "supermarket", "hypermarket"],
    "hospitality": ["hospitality", "hotel", "hotels", "restaurant", "food and beverage", "catering"],
    "healthcare": ["healthcare", "hospital", "hospitals", "clinic", "pharmaceutical", "medical center", "medical centre"],
    "real estate": ["real estate", "property management", "realty", "developer"],
}

_STOPWORDS = {"the", "and", "in", "of", "for", "uae", "with"}


@lru_cache(maxsize=512)
def industry_keywords(jd_industry: str) -> frozenset:
    """Alias keywords/phrases to search for a given JD industry string."""
    norm = jd_industry.casefold()
    keywords = {tok for tok in re.split(r"[^a-z&]+", norm)
                if len(tok) >= 4 and tok not in _STOPWORDS}
    for canonical, aliases in INDUSTRY_ALIASES.items():
        if canonical in norm or any(alias in norm for alias in aliases):
            keywords.update(aliases)
            keywords.add(canonical)
    return frozenset(keywords)


@lru_cache(maxsize=512)
def _industry_regex(jd_industry: str):
    keywords = industry_keywords(jd_industry)
    if not keywords:
        return None
    # longest-first so multi-word phrases win; whole-word + optional plural 's'
    # so short roots (duct, mep) don't collide with longer words (product, example).
    alternation = "|".join(re.escape(k) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})s?\b", re.IGNORECASE)


def industry_present(jd_industry: str, text: str) -> bool:
    """True if any alias keyword for `jd_industry` appears as a whole word in `text`."""
    if not text:
        return False
    regex = _industry_regex(jd_industry)
    return bool(regex and regex.search(text))


def jd_industry_requirements(jd) -> list[tuple[str, ImportanceLevel]]:
    """Collect (industry, priority) requirements from the JD (dedup, priority-aware).

    experience.industry_experience carries priorities; the top-level industry list
    does not, so it defaults to VALUABLE.
    """
    requirements: list[tuple[str, ImportanceLevel]] = []
    seen: set[str] = set()

    experience = getattr(jd, "experience", None)
    if experience and getattr(experience, "industry_experience", None):
        for item in experience.industry_experience:
            key = (item.industry or "").casefold().strip()
            if key and key not in seen:
                seen.add(key)
                requirements.append((item.industry, item.priority))

    for industry in getattr(jd, "industry", None) or []:
        key = str(industry).casefold().strip()
        if key and key not in seen:
            seen.add(key)
            requirements.append((industry, ImportanceLevel.VALUABLE))

    return requirements
