from abc import ABC, abstractmethod
from typing import Any, Iterable

from models.candidate import CandidateProfile


class CandidateAdapter(ABC):
    """Translate a raw candidate source into canonical `CandidateProfile` objects.

    A concrete adapter owns ALL source-specific parsing (column names, nested
    JSON, date math, list-string parsing, ...). The scoring pipeline depends only
    on `CandidateProfile`, so adding a new data source means adding a new adapter
    with zero changes to filtering / scoring / ranking.

    Contract for `to_profile`:
      * MUST populate core fields: candidate_id, job_title, skills, education.
        SHOULD populate summary / responsibilities / location when available.
      * Enrichment fields are best-effort: leave them None when the source lacks
        them (the pipeline treats None as "signal absent" and renormalizes the
        component weights, preserving backward compatibility).
      * Adapters DO NOT compute embeddings. They set the raw canonical fields and
        leave `profile_text` / `profile_embedding` for the embedding step.
    """

    #: short identifier stamped onto each produced profile, e.g. "resume", "linkedin"
    source_name: str = "base"

    @abstractmethod
    def load(self, source: Any) -> Iterable[dict]:
        """Read the raw source (a path, DataFrame, ...) into per-candidate records."""
        raise NotImplementedError

    @abstractmethod
    def to_profile(self, record: dict, index: int) -> CandidateProfile:
        """Map ONE raw record into a `CandidateProfile`.

        `index` is provided so adapters can synthesize a stable candidate_id when
        the source has no natural identifier.
        """
        raise NotImplementedError

    def to_profiles(self, source: Any) -> list[CandidateProfile]:
        """Load and map every record. Concrete adapters rarely need to override."""
        profiles: list[CandidateProfile] = []
        for index, record in enumerate(self.load(source)):
            profile = self.to_profile(record, index)
            if profile.source is None:
                profile.source = self.source_name
            profiles.append(profile)
        return profiles
