"""Fighter-name normalization and conservative fuzzy matching.

Both data sources key fights on fighter *names*, not ids, so names are the join. That
makes name handling a correctness concern rather than a formatting one: attaching one
fighter's closing line to another fighter's result would corrupt the market baseline
without changing any row count.

The policy throughout is that a dropped row beats a wrong pairing:

* exact match on normalized names first;
* then a hand-reviewed alias table;
* then fuzzy matching, which must clear a high similarity floor on *both* fighters
  **and** beat the runner-up candidate by a margin, otherwise the row is dropped as
  ambiguous rather than guessed.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# Suffixes that appear inconsistently between sources and carry no identity.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: object) -> str:
    """Fold a fighter name to a comparable key.

    Strips accents (``José`` -> ``jose``), lowercases, removes punctuation such as the
    hyphen in ``Cortes-Acosta`` and the period in ``Jr.``, drops generational suffixes,
    and collapses whitespace.

    Accepts any cell value: Parquet round-trips missing names as float NaN, which is
    truthy, so the null check cannot rely on falsiness alone.

    >>> normalize_name("José Aldo Jr.")
    'jose aldo'
    >>> normalize_name("Waldo Cortes-Acosta")
    'waldo cortes acosta'
    >>> normalize_name(float("nan"))
    ''
    """
    if name is None or not isinstance(name, str):
        if name is None or name != name:  # NaN is the only value unequal to itself
            return ""
        name = str(name)
    if not name:
        return ""
    folded = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    folded = folded.lower()
    folded = re.sub(r"[^a-z\s]", " ", folded)
    tokens = [t for t in folded.split() if t and t not in _SUFFIXES]
    return " ".join(tokens)


def similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two already-normalized names.

    Identical token multisets score 1.0. The two sources disagree on name order for
    many Chinese and Mongolian fighters -- ``zhang weili`` vs ``weili zhang``,
    ``batgerel danaa`` vs ``danaa batgerel`` -- and those are the same person with the
    parts transposed, not a near-miss. A character ratio alone scores such pairs around
    0.45 and would reject them, so word order is normalized away here rather than
    compensated for by lowering the acceptance threshold.

    Otherwise, a character-level ratio is blended with token overlap. The character
    ratio dominates because token overlap alone is far too permissive for shared
    surnames: two unrelated fighters both named "silva" would score 0.5 on tokens.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = a.split(), b.split()
    if sorted(ta) == sorted(tb):
        return 1.0
    char_ratio = SequenceMatcher(None, a, b).ratio()
    sa, sb = set(ta), set(tb)
    token_ratio = len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0
    return 0.65 * char_ratio + 0.35 * token_ratio


@dataclass(frozen=True)
class MatchResult:
    """Outcome of resolving one odds row against the UFCStats bout index."""

    fight_id: str | None
    method: str  # "exact" | "alias" | "fuzzy" | "unmatched_no_candidate" | "unmatched_ambiguous"
    score: float = 1.0
    runner_up: float = 0.0
    matched_bout: str = ""

    @property
    def is_match(self) -> bool:
        return self.fight_id is not None


def load_aliases(path: Path) -> dict[str, str]:
    """Load the hand-reviewed alias table: odds-source name -> UFCStats name.

    Returns an empty mapping if the file does not exist yet, so the pipeline runs
    before any aliases have been reviewed. Both sides are normalized on load, so the
    file may be written with natural spelling.
    """
    if not path.is_file():
        return {}
    aliases: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            src = normalize_name(row.get("odds_name"))
            dst = normalize_name(row.get("ufcstats_name"))
            if src and dst:
                aliases[src] = dst
    return aliases


class BoutIndex:
    """Index of UFCStats bouts, queryable by (event date, fighter pair).

    Matching is always scoped to a single event date. That is justified empirically:
    across the 6,937 priced fights, every join failure was a name variant and none was
    a date mismatch, so widening the date window would only add false-pairing risk
    without recovering rows.
    """

    def __init__(self, bouts: list[tuple[str, object, str, str]]) -> None:
        """
        Args:
            bouts: ``(fight_id, event_date, fighter_a_name, fighter_b_name)`` tuples.
                Names may be raw; they are normalized here.
        """
        self._by_date: dict[object, list[tuple[str, str, str]]] = {}
        self._exact: dict[tuple[object, frozenset[str]], str] = {}
        for fight_id, date, a, b in bouts:
            na, nb = normalize_name(a), normalize_name(b)
            self._by_date.setdefault(date, []).append((fight_id, na, nb))
            self._exact[(date, frozenset((na, nb)))] = fight_id

    def __len__(self) -> int:
        return len(self._exact)

    def match(
        self,
        date: object,
        name_a: str,
        name_b: str,
        *,
        aliases: dict[str, str] | None = None,
        threshold: float = 0.92,
        ambiguity_margin: float = 0.05,
    ) -> MatchResult:
        """Resolve one odds row to a fight_id, or explain why it cannot be resolved."""
        na, nb = normalize_name(name_a), normalize_name(name_b)

        # 1. Exact match on the normalized pair.
        hit = self._exact.get((date, frozenset((na, nb))))
        if hit is not None:
            return MatchResult(hit, "exact")

        # 2. Alias table (hand-reviewed), then retry exact.
        if aliases:
            aa, ab = aliases.get(na, na), aliases.get(nb, nb)
            if (aa, ab) != (na, nb):
                hit = self._exact.get((date, frozenset((aa, ab))))
                if hit is not None:
                    return MatchResult(hit, "alias")
                na, nb = aa, ab

        # 3. Fuzzy, scoped to the same date. Score a candidate by its weaker side, so
        #    one strong name cannot carry a mismatched partner.
        candidates = self._by_date.get(date, [])
        scored: list[tuple[float, str, str]] = []
        for fight_id, ca, cb in candidates:
            direct = min(similarity(na, ca), similarity(nb, cb))
            swapped = min(similarity(na, cb), similarity(nb, ca))
            score = max(direct, swapped)
            scored.append((score, fight_id, f"{ca} vs {cb}"))

        if not scored:
            return MatchResult(None, "unmatched_no_candidate")

        scored.sort(reverse=True)
        best_score, best_id, best_bout = scored[0]
        runner_up = scored[1][0] if len(scored) > 1 else 0.0

        if best_score < threshold:
            return MatchResult(
                None,
                "unmatched_no_candidate",
                score=best_score,
                runner_up=runner_up,
                matched_bout=best_bout,
            )
        if best_score - runner_up < ambiguity_margin:
            # Two plausible bouts on the same card: guessing risks a wrong pairing.
            return MatchResult(
                None,
                "unmatched_ambiguous",
                score=best_score,
                runner_up=runner_up,
                matched_bout=best_bout,
            )
        return MatchResult(
            best_id, "fuzzy", score=best_score, runner_up=runner_up, matched_bout=best_bout
        )
