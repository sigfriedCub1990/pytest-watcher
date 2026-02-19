from __future__ import annotations

import os
from typing import List, Optional, Sequence
from unittest.mock import patch

import pytest

from pytest_watcher.picker import (
    KEY_BACKSPACE,
    KEY_CHAR,
    KEY_DOWN,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_UP,
    MAX_VISIBLE_RESULTS,
    KeyEvent,
    PickerState,
    _read_key_event,
    make_raw_reader,
    render,
    run_picker,
    update_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANDIDATES = [
    "tests/test_auth.py",
    "tests/test_cache.py",
    "tests/test_commands.py",
    "tests/unit/test_models.py",
    "tests/unit/test_views.py",
]


def _identity_filter(query: str, candidates: Sequence[str]) -> List[str]:
    """Trivial filter that returns all candidates (for state-only tests)."""
    if not query:
        return list(candidates)
    return [c for c in candidates if query.lower() in c.lower()]


def _make_char_reader(chars: str):
    """Return a callable that yields one char at a time from *chars*."""
    it = iter(chars)

    def read_char() -> Optional[str]:
        return next(it, None)

    return read_char


def _make_gapped_reader(tokens: Sequence[Optional[str]]):
    """Return a reader that yields from *tokens*, where ``None`` simulates
    a ``select()`` timeout (no data available).

    This reproduces what happens in a real terminal when Python's
    ``BufferedReader`` eagerly consumes bytes from the fd, causing
    subsequent ``select()`` calls to report no data even though
    continuation bytes of an escape sequence are already in Python's
    internal buffer.
    """
    it = iter(tokens)

    def read_char() -> Optional[str]:
        return next(it, None)

    return read_char


# ---------------------------------------------------------------------------
# _read_key_event
# ---------------------------------------------------------------------------


class TestReadKeyEvent:
    def test_printable_char(self):
        ev = _read_key_event(_make_char_reader("a"))
        assert ev == KeyEvent(KEY_CHAR, "a")

    def test_enter_cr(self):
        ev = _read_key_event(_make_char_reader("\r"))
        assert ev == KeyEvent(KEY_ENTER)

    def test_enter_lf(self):
        ev = _read_key_event(_make_char_reader("\n"))
        assert ev == KeyEvent(KEY_ENTER)

    def test_backspace_del(self):
        ev = _read_key_event(_make_char_reader("\x7f"))
        assert ev == KeyEvent(KEY_BACKSPACE)

    def test_backspace_bs(self):
        ev = _read_key_event(_make_char_reader("\x08"))
        assert ev == KeyEvent(KEY_BACKSPACE)

    def test_arrow_up(self):
        ev = _read_key_event(_make_char_reader("\x1b[A"))
        assert ev == KeyEvent(KEY_UP)

    def test_arrow_down(self):
        ev = _read_key_event(_make_char_reader("\x1b[B"))
        assert ev == KeyEvent(KEY_DOWN)

    def test_bare_escape(self):
        reader = _make_char_reader("\x1b")
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_ESCAPE)

    def test_unknown_escape_seq(self):
        # ESC [ C is right-arrow — not handled, treated as escape
        reader = _make_char_reader("\x1b[C")
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_ESCAPE)

    def test_none_when_no_input(self):
        ev = _read_key_event(lambda: None)
        assert ev is None

    def test_arrow_down_with_none_after_esc(self):
        """select() may timeout between ESC and '[' due to Python buffering."""
        reader = _make_gapped_reader(["\x1b", None, "[", "B"])
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_DOWN)

    def test_arrow_up_with_none_after_esc(self):
        reader = _make_gapped_reader(["\x1b", None, "[", "A"])
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_UP)

    def test_arrow_down_with_none_after_bracket(self):
        """select() may timeout between '[' and direction byte."""
        reader = _make_gapped_reader(["\x1b", "[", None, "B"])
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_DOWN)

    def test_arrow_up_with_nones_throughout(self):
        """Multiple select() timeouts within a single escape sequence."""
        reader = _make_gapped_reader(["\x1b", None, None, "[", None, "A"])
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_UP)

    def test_arrow_down_application_mode(self):
        """Some terminals send ESC O B instead of ESC [ B."""
        reader = _make_char_reader("\x1bOB")
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_DOWN)

    def test_arrow_up_application_mode(self):
        """Some terminals send ESC O A instead of ESC [ A."""
        reader = _make_char_reader("\x1bOA")
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_UP)

    def test_arrow_application_mode_with_none_gaps(self):
        reader = _make_gapped_reader(["\x1b", None, "O", None, "A"])
        ev = _read_key_event(reader)
        assert ev == KeyEvent(KEY_UP)


# ---------------------------------------------------------------------------
# update_state
# ---------------------------------------------------------------------------


class TestUpdateState:
    def _new_state(self) -> PickerState:
        return PickerState(
            results=list(CANDIDATES),
            total=len(CANDIDATES),
        )

    def test_char_appends_to_query(self):
        state = self._new_state()
        update_state(state, KeyEvent(KEY_CHAR, "a"), _identity_filter, CANDIDATES)
        assert state.query == "a"
        assert state.cursor == 0

    def test_typing_filters_results(self):
        state = self._new_state()
        update_state(state, KeyEvent(KEY_CHAR, "a"), _identity_filter, CANDIDATES)
        update_state(state, KeyEvent(KEY_CHAR, "u"), _identity_filter, CANDIDATES)
        update_state(state, KeyEvent(KEY_CHAR, "t"), _identity_filter, CANDIDATES)
        update_state(state, KeyEvent(KEY_CHAR, "h"), _identity_filter, CANDIDATES)
        assert state.query == "auth"
        assert state.results == ["tests/test_auth.py"]

    def test_backspace_removes_last_char(self):
        state = self._new_state()
        state.query = "ab"
        state.results = []  # simulate empty results for "ab"
        update_state(state, KeyEvent(KEY_BACKSPACE), _identity_filter, CANDIDATES)
        assert state.query == "a"
        assert state.cursor == 0

    def test_backspace_on_empty_is_noop(self):
        state = self._new_state()
        update_state(state, KeyEvent(KEY_BACKSPACE), _identity_filter, CANDIDATES)
        assert state.query == ""
        # results unchanged
        assert state.results == list(CANDIDATES)

    def test_cursor_down(self):
        state = self._new_state()
        update_state(state, KeyEvent(KEY_DOWN), _identity_filter, CANDIDATES)
        assert state.cursor == 1

    def test_cursor_down_clamps(self):
        state = self._new_state()
        for _ in range(50):
            update_state(state, KeyEvent(KEY_DOWN), _identity_filter, CANDIDATES)
        assert state.cursor == len(CANDIDATES) - 1

    def test_cursor_up(self):
        state = self._new_state()
        state.cursor = 3
        update_state(state, KeyEvent(KEY_UP), _identity_filter, CANDIDATES)
        assert state.cursor == 2

    def test_cursor_up_clamps_at_zero(self):
        state = self._new_state()
        update_state(state, KeyEvent(KEY_UP), _identity_filter, CANDIDATES)
        assert state.cursor == 0

    def test_enter_selects_current(self):
        state = self._new_state()
        state.cursor = 2
        update_state(state, KeyEvent(KEY_ENTER), _identity_filter, CANDIDATES)
        assert state.done is True
        assert state.selected == CANDIDATES[2]

    def test_enter_with_no_results(self):
        state = PickerState(results=[], total=0)
        update_state(state, KeyEvent(KEY_ENTER), _identity_filter, [])
        assert state.done is True
        assert state.selected is None

    def test_escape_cancels(self):
        state = self._new_state()
        update_state(state, KeyEvent(KEY_ESCAPE), _identity_filter, CANDIDATES)
        assert state.done is True
        assert state.selected is None

    def test_typing_resets_cursor_to_zero(self):
        state = self._new_state()
        state.cursor = 3
        update_state(state, KeyEvent(KEY_CHAR, "x"), _identity_filter, CANDIDATES)
        assert state.cursor == 0

    def test_backspace_resets_cursor_to_zero(self):
        state = self._new_state()
        state.query = "abc"
        state.cursor = 2
        update_state(state, KeyEvent(KEY_BACKSPACE), _identity_filter, CANDIDATES)
        assert state.cursor == 0


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


class TestRender:
    def test_shows_query(self):
        state = PickerState(
            query="foo", results=["a.py", "b.py"], total=5
        )
        output = render(state)
        assert "foo" in output

    def test_shows_match_count(self):
        state = PickerState(results=["a.py"], total=10)
        output = render(state)
        assert "1/10" in output

    def test_highlights_cursor_row(self):
        state = PickerState(
            results=["first.py", "second.py"], total=2, cursor=1
        )
        output = render(state)
        # The highlighted row should contain the item
        assert "second.py" in output
        # '❯' marker on the cursor row
        lines = output.split("\n")
        cursor_lines = [l for l in lines if "❯" in l]
        assert len(cursor_lines) == 1
        assert "second.py" in cursor_lines[0]

    def test_no_matches_message(self):
        state = PickerState(query="zzz", results=[], total=5)
        output = render(state)
        assert "no matches" in output.lower()

    def test_limits_visible_rows(self):
        many = [f"test_{i}.py" for i in range(30)]
        state = PickerState(results=many, total=30)
        output = render(state)
        # Should only show MAX_VISIBLE_RESULTS items
        visible_count = sum(
            1 for line in output.split("\n")
            if line.strip().startswith(("❯", "test_"))
        )
        assert visible_count <= MAX_VISIBLE_RESULTS


# ---------------------------------------------------------------------------
# run_picker  (end-to-end with injected I/O)
# ---------------------------------------------------------------------------


class TestRunPicker:
    def _simulate(self, keys: str) -> tuple[Optional[str], str]:
        """Run the picker with simulated keystrokes.

        Returns (selected, captured_output).
        """
        output_buf: list[str] = []

        def write(s: str) -> None:
            output_buf.append(s)

        reader = _make_char_reader(keys)

        selected = run_picker(
            CANDIDATES,
            _identity_filter,
            _read_char=reader,
            _write=write,
        )
        return selected, "".join(output_buf)

    def test_immediate_enter_selects_first(self):
        selected, _ = self._simulate("\r")
        assert selected == CANDIDATES[0]

    def test_arrow_down_then_enter(self):
        selected, _ = self._simulate("\x1b[B\r")  # ↓ Enter
        assert selected == CANDIDATES[1]

    def test_arrow_down_down_up_enter(self):
        selected, _ = self._simulate("\x1b[B\x1b[B\x1b[A\r")  # ↓↓↑ Enter
        assert selected == CANDIDATES[1]

    def test_type_query_then_enter(self):
        # Typing "auth" should filter to just test_auth.py
        selected, _ = self._simulate("auth\r")
        assert selected == "tests/test_auth.py"

    def test_type_query_backspace_then_enter(self):
        # Type "authx", backspace, then enter
        selected, _ = self._simulate("authx\x7f\r")
        assert selected == "tests/test_auth.py"

    def test_escape_cancels(self):
        selected, _ = self._simulate("\x1b")
        assert selected is None

    def test_escape_after_typing(self):
        selected, _ = self._simulate("au\x1b")
        assert selected is None

    def test_no_match_enter_returns_none(self):
        selected, _ = self._simulate("zzzzz\r")
        assert selected is None

    def test_output_contains_candidates(self):
        _, output = self._simulate("\r")
        for c in CANDIDATES:
            assert c in output

    def test_output_shows_match_count(self):
        _, output = self._simulate("auth\r")
        assert "1/" in output

    def test_navigate_past_bottom_stays_clamped(self):
        # Press down 20 times, then enter — should select the last item
        downs = "\x1b[B" * 20
        selected, _ = self._simulate(f"{downs}\r")
        assert selected == CANDIDATES[-1]

    def test_navigate_past_top_stays_at_zero(self):
        ups = "\x1b[A" * 5
        selected, _ = self._simulate(f"{ups}\r")
        assert selected == CANDIDATES[0]


# ---------------------------------------------------------------------------
# run_picker — real-terminal simulation (gapped reads)
# ---------------------------------------------------------------------------


class TestRunPickerGappedReads:
    """Simulate the real terminal where select() can timeout between bytes
    of an escape sequence due to Python's BufferedReader consuming from
    the fd eagerly.
    """

    def _simulate_gapped(
        self, tokens: Sequence[Optional[str]]
    ) -> tuple[Optional[str], str]:
        output_buf: list[str] = []

        def write(s: str) -> None:
            output_buf.append(s)

        reader = _make_gapped_reader(tokens)
        selected = run_picker(
            CANDIDATES,
            _identity_filter,
            _read_char=reader,
            _write=write,
        )
        return selected, "".join(output_buf)

    def test_arrow_down_gapped_then_enter(self):
        """Down arrow with None gaps, then Enter — must select second item."""
        tokens: list[Optional[str]] = ["\x1b", None, "[", None, "B", "\r"]
        selected, _ = self._simulate_gapped(tokens)
        assert selected == CANDIDATES[1]

    def test_two_downs_gapped_then_enter(self):
        tokens: list[Optional[str]] = [
            "\x1b", None, "[", "B",  # ↓
            "\x1b", "[", None, "B",  # ↓
            "\r",
        ]
        selected, _ = self._simulate_gapped(tokens)
        assert selected == CANDIDATES[2]

    def test_down_up_gapped_then_enter(self):
        tokens: list[Optional[str]] = [
            "\x1b", None, "[", "B",  # ↓
            "\x1b", None, "[", "A",  # ↑
            "\r",
        ]
        selected, _ = self._simulate_gapped(tokens)
        assert selected == CANDIDATES[0]

    def test_application_mode_arrows_gapped(self):
        tokens: list[Optional[str]] = [
            "\x1b", None, "O", None, "B",  # ↓ (app mode)
            "\x1b", None, "O", "B",        # ↓ (app mode)
            "\r",
        ]
        selected, _ = self._simulate_gapped(tokens)
        assert selected == CANDIDATES[2]

    def test_type_then_gapped_arrow_then_enter(self):
        """Type a query to filter, then navigate with gapped arrows."""
        tokens: list[Optional[str]] = [
            "c",                             # filter to cache + commands
            "\x1b", None, "[", None, "B",    # ↓
            "\r",
        ]
        selected, _ = self._simulate_gapped(tokens)
        # "c" matches cache and commands; ↓ selects the second match
        assert selected is not None
        assert selected != CANDIDATES[0]  # not the first overall candidate


# ---------------------------------------------------------------------------
# make_raw_reader — verifying the unbuffered fd reader
# ---------------------------------------------------------------------------


class TestMakeRawReader:
    """Verify that ``make_raw_reader`` uses ``os.read`` on the raw fd
    so that ``select()`` and reads operate on the same OS-level buffer.
    """

    def test_reads_single_byte_via_os_read(self):
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"x")
            reader = make_raw_reader(r_fd)
            assert reader() == "x"
        finally:
            os.close(r_fd)
            os.close(w_fd)

    def test_returns_none_when_no_data(self):
        r_fd, w_fd = os.pipe()
        try:
            reader = make_raw_reader(r_fd)
            assert reader() is None
        finally:
            os.close(r_fd)
            os.close(w_fd)

    def test_reads_escape_sequence_byte_by_byte(self):
        """All 3 bytes of an arrow sequence must be individually readable."""
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"\x1b[B")
            reader = make_raw_reader(r_fd)
            assert reader() == "\x1b"
            assert reader() == "["
            assert reader() == "B"
            assert reader() is None
        finally:
            os.close(r_fd)
            os.close(w_fd)

    def test_arrow_key_parsed_correctly_via_raw_reader(self):
        """End-to-end: raw reader → _read_key_event → correct arrow event."""
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"\x1b[A")
            reader = make_raw_reader(r_fd)
            ev = _read_key_event(reader)
            assert ev == KeyEvent(KEY_UP)
        finally:
            os.close(r_fd)
            os.close(w_fd)

    def test_multiple_arrow_keys_via_raw_reader(self):
        """Two arrow sequences written at once are parsed individually."""
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, b"\x1b[B\x1b[A")  # ↓ then ↑
            reader = make_raw_reader(r_fd)
            ev1 = _read_key_event(reader)
            ev2 = _read_key_event(reader)
            assert ev1 == KeyEvent(KEY_DOWN)
            assert ev2 == KeyEvent(KEY_UP)
        finally:
            os.close(r_fd)
            os.close(w_fd)

    def test_run_picker_with_raw_reader_on_pipe(self):
        """Full picker run using a raw reader on a real fd (pipe)."""
        r_fd, w_fd = os.pipe()
        try:
            # Write: ↓ ↓ Enter
            os.write(w_fd, b"\x1b[B\x1b[B\r")
            output_buf: list[str] = []
            reader = make_raw_reader(r_fd)
            selected = run_picker(
                CANDIDATES,
                _identity_filter,
                _read_char=reader,
                _write=lambda s: output_buf.append(s),
            )
            assert selected == CANDIDATES[2]
        finally:
            os.close(r_fd)
            os.close(w_fd)
