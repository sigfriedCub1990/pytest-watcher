from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from pytest_watcher.fuzzy import find_test_files, fuzzy_filter, fuzzy_match


# ---------------------------------------------------------------------------
# fuzzy_match
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    def test_exact_substring(self):
        matched, score = fuzzy_match("foo", "foobar")
        assert matched is True
        assert score > 0

    def test_case_insensitive(self):
        matched, _ = fuzzy_match("FOO", "foobar")
        assert matched is True

    def test_characters_in_order(self):
        matched, _ = fuzzy_match("tmd", "test_my_decorator.py")
        assert matched is True

    def test_no_match(self):
        matched, score = fuzzy_match("xyz", "foobar")
        assert matched is False

    def test_empty_query_always_matches(self):
        matched, _ = fuzzy_match("", "anything")
        assert matched is True

    def test_query_longer_than_text(self):
        matched, _ = fuzzy_match("abcdef", "abc")
        assert matched is False

    def test_consecutive_bonus(self):
        _, score_consecutive = fuzzy_match("ab", "ab_cd")
        _, score_spread = fuzzy_match("ab", "a___b")
        assert score_consecutive > score_spread

    def test_boundary_bonus(self):
        # 'c' at word boundary after '_' should score higher
        _, score_boundary = fuzzy_match("tc", "test_cache.py")
        _, score_mid = fuzzy_match("tc", "test_factory.py")
        assert score_boundary >= score_mid


# ---------------------------------------------------------------------------
# fuzzy_filter
# ---------------------------------------------------------------------------


class TestFuzzyFilter:
    CANDIDATES = [
        "tests/test_auth.py",
        "tests/test_cache.py",
        "tests/test_commands.py",
        "tests/unit/test_models.py",
        "tests/unit/test_views.py",
    ]

    def test_empty_query_returns_all(self):
        result = fuzzy_filter("", self.CANDIDATES)
        assert result == list(self.CANDIDATES)

    def test_single_character(self):
        result = fuzzy_filter("m", self.CANDIDATES)
        assert "tests/test_commands.py" in result
        assert "tests/unit/test_models.py" in result

    def test_filters_non_matching(self):
        result = fuzzy_filter("auth", self.CANDIDATES)
        assert result == ["tests/test_auth.py"]

    def test_order_by_score(self):
        result = fuzzy_filter("model", self.CANDIDATES)
        assert result[0] == "tests/unit/test_models.py"

    def test_no_matches(self):
        result = fuzzy_filter("zzzzz", self.CANDIDATES)
        assert result == []


# ---------------------------------------------------------------------------
# find_test_files
# ---------------------------------------------------------------------------


class TestFindTestFiles:
    def test_discovers_test_files(self, tmp_path: Path):
        # Create a small directory tree
        (tmp_path / "test_alpha.py").write_text("")
        (tmp_path / "beta_test.py").write_text("")
        (tmp_path / "helper_utils.py").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir(exist_ok=True)
        (sub / "test_gamma.py").write_text("")

        files = find_test_files(tmp_path)

        assert "test_alpha.py" in files
        assert "beta_test.py" in files
        assert str(Path("sub/test_gamma.py")) in files
        assert "helper_utils.py" not in files

    def test_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir(exist_ok=True)

        assert find_test_files(empty) == []


# ---------------------------------------------------------------------------
# Integration-style: filter real-ish candidates
# ---------------------------------------------------------------------------


class TestFuzzyFilterIntegration:
    """Simulate the workflow: discover → filter → pick."""

    def test_filter_from_discovered(self, tmp_path: Path):
        (tmp_path / "test_login.py").write_text("")
        (tmp_path / "test_logout.py").write_text("")
        (tmp_path / "test_profile.py").write_text("")

        files = find_test_files(tmp_path)
        result = fuzzy_filter("login", files)

        assert result == ["test_login.py"]

    def test_fuzzy_partial(self, tmp_path: Path):
        (tmp_path / "test_user_authentication.py").write_text("")
        (tmp_path / "test_user_authorization.py").write_text("")
        (tmp_path / "test_payment.py").write_text("")

        files = find_test_files(tmp_path)
        result = fuzzy_filter("uauth", files)

        # Both user_auth* files should match; payment should not
        assert len(result) == 2
        assert "test_payment.py" not in result
