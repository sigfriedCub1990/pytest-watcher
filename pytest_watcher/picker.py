"""Interactive fuzzy picker for selecting test files.

Renders an fzf-style interface in the terminal:

    Filter > qu|
    3/5 matches
    ❯ tests/test_query.py
      tests/test_request.py
      tests/unit/test_sql_query.py

Keys:
    typing      – refine the query
    ↑ / ↓       – move the selection cursor
    Enter       – accept the highlighted item
    Escape      – cancel and return to the watcher
"""

from __future__ import annotations

import logging
import os
import select
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ANSI helpers
_CSI = "\033["
_CLEAR_LINE = f"{_CSI}2K"
_CURSOR_UP = f"{_CSI}1A"
_HIDE_CURSOR = f"{_CSI}?25l"
_SHOW_CURSOR = f"{_CSI}?25h"
_BOLD = f"{_CSI}1m"
_CYAN = f"{_CSI}36m"
_REVERSE = f"{_CSI}7m"
_RESET = f"{_CSI}0m"

# Maximum number of result rows to display at once
MAX_VISIBLE_RESULTS = 15


@dataclass
class PickerState:
    """Mutable state for the fuzzy picker."""

    query: str = ""
    cursor: int = 0  # index into `results`
    results: List[str] = field(default_factory=list)
    total: int = 0  # total candidates before filtering
    done: bool = False
    selected: Optional[str] = None  # the accepted result (None = cancelled)


# -- Key constants -----------------------------------------------------------

KEY_ENTER = "enter"
KEY_ESCAPE = "escape"
KEY_BACKSPACE = "backspace"
KEY_UP = "up"
KEY_DOWN = "down"
KEY_CHAR = "char"  # regular printable character


@dataclass
class KeyEvent:
    kind: str
    char: str = ""


_ESC_READ_RETRIES = 4  # max None returns to tolerate inside an escape sequence


def _read_continuation(
    read_char: Callable[[], Optional[str]],
) -> Optional[str]:
    """Read the next real character, skipping up to *_ESC_READ_RETRIES* ``None``
    returns.

    In a real terminal the default ``read_char`` uses ``select()`` on the
    OS file descriptor, but Python's ``BufferedReader`` may have already
    pulled the continuation bytes into its internal buffer.  ``select()``
    then reports *no data* even though the bytes are available — so
    ``read_char()`` returns ``None``.  Retrying a few times fixes this.
    """
    for _ in range(_ESC_READ_RETRIES):
        ch = read_char()
        if ch is not None:
            return ch
    return None


def _parse_arrow(direction: Optional[str]) -> Optional[KeyEvent]:
    if direction == "A":
        return KeyEvent(KEY_UP)
    if direction == "B":
        return KeyEvent(KEY_DOWN)
    return None


def _read_key_event(read_char: Callable[[], Optional[str]]) -> Optional[KeyEvent]:
    """Read one logical key event using *read_char* (a single-char reader).

    Handles multi-byte escape sequences for arrow keys in both normal
    mode (``ESC [ A/B``) and application mode (``ESC O A/B``).
    Tolerates ``None`` gaps between bytes (see :func:`_read_continuation`).
    """
    ch = read_char()
    if ch is None:
        return None

    if ch == "\x1b":  # ESC – might be an arrow-key sequence
        seq1 = _read_continuation(read_char)
        if seq1 in ("[", "O"):
            seq2 = _read_continuation(read_char)
            arrow = _parse_arrow(seq2)
            if arrow is not None:
                return arrow
        if seq1 is None:
            # No follow-up byte at all → genuine Escape press
            return KeyEvent(KEY_ESCAPE)
        # Unrecognised sequence → treat as Escape
        return KeyEvent(KEY_ESCAPE)

    if ch in ("\n", "\r"):
        return KeyEvent(KEY_ENTER)

    if ch in ("\x7f", "\x08"):  # DEL / Backspace
        return KeyEvent(KEY_BACKSPACE)

    if ch.isprintable():
        return KeyEvent(KEY_CHAR, ch)

    return None  # ignore other control characters


# -- Rendering ---------------------------------------------------------------


def render(state: PickerState) -> str:
    """Return the full screen content for the current state."""
    lines: List[str] = []

    # Header: query line
    lines.append(f"{_BOLD}{_CYAN}Filter >{_RESET} {state.query}")

    # Match count
    lines.append(
        f"  {_CYAN}{len(state.results)}/{state.total} matches{_RESET}"
    )

    # Result rows
    visible = state.results[:MAX_VISIBLE_RESULTS]
    for i, item in enumerate(visible):
        if i == state.cursor:
            lines.append(f"  {_REVERSE}{_BOLD}❯ {item}{_RESET}")
        else:
            lines.append(f"    {item}")

    if not visible:
        lines.append(f"  {_CYAN}(no matches){_RESET}")

    return "\n".join(lines)


def _printed_line_count(text: str) -> int:
    return text.count("\n") + 1


# -- Core loop ---------------------------------------------------------------


def update_state(
    state: PickerState,
    event: KeyEvent,
    filter_fn: Callable[[str, Sequence[str]], List[str]],
    candidates: Sequence[str],
) -> None:
    """Apply *event* to *state*, recomputing results when the query changes."""
    if event.kind == KEY_ESCAPE:
        state.done = True
        state.selected = None
        return

    if event.kind == KEY_ENTER:
        state.done = True
        if state.results and 0 <= state.cursor < len(state.results):
            state.selected = state.results[state.cursor]
        else:
            state.selected = None
        return

    if event.kind == KEY_BACKSPACE:
        if state.query:
            state.query = state.query[:-1]
            state.results = filter_fn(state.query, candidates)
            state.cursor = 0
        return

    if event.kind == KEY_UP:
        state.cursor = max(0, state.cursor - 1)
        return

    if event.kind == KEY_DOWN:
        limit = min(len(state.results), MAX_VISIBLE_RESULTS) - 1
        state.cursor = min(limit, state.cursor + 1)
        return

    if event.kind == KEY_CHAR:
        state.query += event.char
        state.results = filter_fn(state.query, candidates)
        state.cursor = 0
        return


def run_picker(
    candidates: Sequence[str],
    filter_fn: Callable[[str, Sequence[str]], List[str]],
    *,
    _read_char: Optional[Callable[[], Optional[str]]] = None,
    _write: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """Run the interactive picker and return the selected path, or ``None``.

    *_read_char* and *_write* are injectable for testing; when ``None`` they
    default to reading from ``sys.stdin`` (in cbreak mode) and writing to
    ``sys.stdout``.
    """
    if _write is None:

        def _write(s: str) -> None:
            sys.stdout.write(s)
            sys.stdout.flush()

    if _read_char is None:

        def _read_char() -> Optional[str]:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                return sys.stdin.read(1)
            return None

    state = PickerState(
        results=list(candidates),
        total=len(candidates),
    )

    _write(_HIDE_CURSOR)
    prev_lines = 0

    try:
        while not state.done:
            # Erase previous frame
            if prev_lines:
                _write(_CURSOR_UP * (prev_lines - 1))
                _write("\r")
                for _ in range(prev_lines):
                    _write(f"{_CLEAR_LINE}\n")
                _write(_CURSOR_UP * prev_lines)
                _write("\r")

            frame = render(state)
            _write(frame)
            prev_lines = _printed_line_count(frame)

            event = _read_key_event(_read_char)
            if event is None:
                continue
            update_state(state, event, filter_fn, candidates)
    finally:
        _write(_SHOW_CURSOR)
        # Move below the rendered frame so the next output starts clean
        _write("\n")

    return state.selected
