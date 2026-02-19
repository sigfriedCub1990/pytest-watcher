from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Default glob patterns used to discover test files
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")


def fuzzy_match(query: str, text: str) -> Tuple[bool, int]:
    """Return (matched, score) for *query* against *text*.

    The algorithm checks whether every character of *query* appears in *text*
    in order (case-insensitive).  The score rewards:
    * consecutive character runs  (+3 each)
    * matches at the start of a path segment or after ``_``  (+2 each)
    * shorter remaining tails after the last match  (+1 per saved char)

    A higher score means a better match.
    """
    query_lower = query.lower()
    text_lower = text.lower()

    qi = 0  # index into query
    score = 0
    prev_match_idx = -2  # used to detect consecutive matches

    for ti, ch in enumerate(text_lower):
        if qi < len(query_lower) and ch == query_lower[qi]:
            # consecutive bonus
            if ti == prev_match_idx + 1:
                score += 3
            # word-boundary bonus (start, after / or _)
            if ti == 0 or text_lower[ti - 1] in (os.sep, "/", "_"):
                score += 2
            prev_match_idx = ti
            qi += 1

    matched = qi == len(query_lower)
    if matched:
        # shorter tail bonus
        score += max(0, len(text_lower) - prev_match_idx)

    return matched, score


def find_test_files(
    root: Path,
    patterns: Sequence[str] = TEST_FILE_PATTERNS,
) -> List[str]:
    """Walk *root* and return relative paths of files matching *patterns*."""
    results: List[str] = []
    for pattern in patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                try:
                    rel = str(path.relative_to(root))
                except ValueError:
                    rel = str(path)
                results.append(rel)
    # deduplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for r in results:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return sorted(deduped)


def fuzzy_filter(
    query: str,
    candidates: Sequence[str],
) -> List[str]:
    """Return *candidates* that fuzzy-match *query*, sorted best-first."""
    if not query:
        return list(candidates)

    scored: List[Tuple[int, str]] = []
    for c in candidates:
        matched, score = fuzzy_match(query, c)
        if matched:
            scored.append((score, c))

    scored.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in scored]
