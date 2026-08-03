"""Location matching (exact / alias-based).

Candidate location (`CandidateLocation`: city / country / country_code) is scored
against the JD's `location.cities` + `location.countries`. Matching is
normalized-exact with a small alias table (UAE == United Arab Emirates, and a few
UAE city spellings), deliberately avoiding fuzzy/semantic matching for the same
short-token precision reasons as the industry and language matchers.

Scoring intent (encoded in `core.scoring.calculate_location_score`):
  * city match          -> strongest signal (exact place)
  * in-country only      -> partial credit (right country, other/unknown city)
  * country-only JD      -> a country match is full credit (no city was required)
  * out-of-scope         -> miss
"""

from functools import lru_cache

# canonical country -> alternate names denoting the SAME country.
COUNTRY_ALIASES = {
    "united arab emirates": [
        "united arab emirates", "uae", "u.a.e", "u a e", "emirates", "the emirates",
    ],
}

# 2-letter ISO country code -> canonical country name (kept small; extend as needed).
COUNTRY_CODE_ALIASES = {
    "ae": "united arab emirates",
}

# canonical city -> alternate spellings.
CITY_ALIASES = {
    "abu dhabi": ["abu dhabi", "abudhabi", "abu-dhabi"],
    "ras al khaimah": ["ras al khaimah", "ras al-khaimah", "rak"],
    "umm al quwain": ["umm al quwain", "umm al-quwain"],
}

_COUNTRY_ALIAS_TO_CANONICAL = {
    alias: canonical for canonical, aliases in COUNTRY_ALIASES.items() for alias in aliases
}
_CITY_ALIAS_TO_CANONICAL = {
    alias: canonical for canonical, aliases in CITY_ALIASES.items() for alias in aliases
}


@lru_cache(maxsize=1024)
def normalize_country(value: str) -> str:
    """Casefold + strip a country name to its canonical form (aliases mapped)."""
    if not isinstance(value, str) or not value:
        return ""
    norm = value.casefold().strip().strip(".")
    return _COUNTRY_ALIAS_TO_CANONICAL.get(norm, norm)


@lru_cache(maxsize=1024)
def normalize_country_code(code: str) -> str:
    """Map a 2-letter ISO country code to its canonical country name ('' if unknown)."""
    if not isinstance(code, str) or not code:
        return ""
    return COUNTRY_CODE_ALIASES.get(code.casefold().strip(), "")


@lru_cache(maxsize=1024)
def normalize_city(value: str) -> str:
    """Casefold + strip a city name to its canonical form (aliases mapped)."""
    if not isinstance(value, str) or not value:
        return ""
    norm = value.casefold().strip()
    return _CITY_ALIAS_TO_CANONICAL.get(norm, norm)


def jd_location_requirements(jd) -> tuple[frozenset, frozenset]:
    """Return (cities, countries) required by the JD, normalized and empty-dropped.

    Extraction sometimes yields blank city entries (e.g. cities: ["Dubai", ""]);
    those are filtered out, so a country-only requirement stays country-only.
    """
    location = getattr(jd, "location", None)
    if not location:
        return frozenset(), frozenset()
    cities = frozenset(
        c for c in (normalize_city(x) for x in (location.cities or []) if isinstance(x, str)) if c
    )
    countries = frozenset(
        c for c in (normalize_country(x) for x in (location.countries or []) if isinstance(x, str)) if c
    )
    return cities, countries
