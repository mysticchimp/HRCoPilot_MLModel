"""Language matching (exact / alias-based).

Candidate languages and the JD's `language_proficiency` both carry structured
language names, so fit is scored by normalized-exact matching (with a small alias
table for the same language under different names, e.g. Tagalog == Filipino).

This deliberately avoids fuzzy / embedding matching: language names are short
tokens where char-level and semantic similarity produce false merges (the same
precision concern that ruled out fuzzy/semantic industry matching). Genuine
synonyms are handled by explicit aliases instead.

Proficiency levels are NOT gated in the MVP: candidate proficiency is noisy
free-text and most candidates list the essential language at a high level, so the
component scores language *presence*, priority-weighted by the JD.
"""

import re
from functools import lru_cache

from models.enums import ImportanceLevel

# canonical language -> alternate names that denote the SAME language.
# Only genuine merges belong here; mutually-intelligible-but-distinct languages
# (e.g. Hindi vs Urdu) are kept separate on purpose.
LANGUAGE_ALIASES = {
    "filipino": ["filipino", "tagalog", "pilipino"],
    "chinese": ["chinese", "mandarin", "cantonese", "putonghua"],
    "persian": ["persian", "farsi"],
}

_ALIAS_TO_CANONICAL = {
    alias: canonical for canonical, aliases in LANGUAGE_ALIASES.items() for alias in aliases
}


@lru_cache(maxsize=1024)
def normalize_language(name: str) -> str:
    """Casefold + strip a language name to its canonical form.

    Drops trailing qualifiers ("English (US)", "Filipino/Tagalog") and maps known
    aliases to a canonical language. Unknown languages return their own casefolded
    root, so any language still matches when both sides spell it the same way.
    """
    if not name:
        return ""
    root = re.split(r"[(/,;]", name.casefold(), maxsplit=1)[0].strip()
    return _ALIAS_TO_CANONICAL.get(root, root)


def normalize_candidate_languages(names) -> frozenset:
    """Normalize a candidate's raw language names into a set of canonical languages."""
    if not isinstance(names, (list, tuple)):
        return frozenset()
    return frozenset(n for n in (normalize_language(x) for x in names if isinstance(x, str)) if n)


def jd_language_requirements(jd) -> list[tuple[str, ImportanceLevel]]:
    """Collect (canonical_language, priority) requirements from the JD (deduped).

    Reads `jd.language_proficiency`; the required proficiency `level` is ignored in
    the presence-based MVP. First occurrence of a language wins on dedup.
    """
    requirements: list[tuple[str, ImportanceLevel]] = []
    seen: set[str] = set()
    for item in getattr(jd, "language_proficiency", None) or []:
        canonical = normalize_language(item.language)
        if canonical and canonical not in seen:
            seen.add(canonical)
            requirements.append((canonical, item.priority))
    return requirements
