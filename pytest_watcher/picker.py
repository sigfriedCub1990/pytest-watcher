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
import shutil
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

try:
    import termios
    import tty
except ImportError:
    pass

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
    max_visible: int = MAX_VISIBLE_RESULTS  # result rows that fit the terminal


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


def _truncate(text: str, width: int) -> str:
    """Clip *text* to *width* visible columns, marking the cut with an ellipsis."""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "…"


def render(state: PickerState, width: int = 80) -> str:
    """Return the full screen content for the current state.

    Every line is clipped to ``width - 1`` visible columns so it never wraps:
    the erase loop in :func:`run_picker` counts logical lines, and a wrapped
    line would occupy more physical rows than counted, leaving stale frames
    on screen.
    """
    lines: List[str] = []
    max_cols = max(1, width - 1)

    # Header: query line.  Keep the tail of an over-long query so the user
    # always sees what they are typing.
    query = state.query
    query_budget = max(0, max_cols - len("Filter > "))
    if len(query) > query_budget:
        if query_budget > 0:
            query = "…" + query[len(query) - query_budget + 1 :]
        else:
            query = ""
    lines.append(f"{_BOLD}{_CYAN}Filter >{_RESET} {query}")

    # Match count
    counts = _truncate(f"  {len(state.results)}/{state.total} matches", max_cols)
    lines.append(f"{_CYAN}{counts}{_RESET}")

    # Result rows
    visible = state.results[: state.max_visible]
    item_budget = max_cols - 4  # 4-column prefix: "  ❯ " / "    "
    for i, item in enumerate(visible):
        item = _truncate(item, max(1, item_budget))
        if i == state.cursor:
            lines.append(f"  {_REVERSE}{_BOLD}❯ {item}{_RESET}")
        else:
            lines.append(f"    {item}")

    if not visible:
        lines.append(f"{_CYAN}{_truncate('  (no matches)', max_cols)}{_RESET}")

    return "\r\n".join(lines)


def _printed_line_count(text: str) -> int:
    return len(text.splitlines())


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
        limit = min(len(state.results), state.max_visible) - 1
        state.cursor = min(limit, state.cursor + 1)
        return

    if event.kind == KEY_CHAR:
        state.query += event.char
        state.results = filter_fn(state.query, candidates)
        state.cursor = 0
        return


def make_raw_reader(fd: int) -> Callable[[], Optional[str]]:
    """Create a single-byte reader that operates directly on a file descriptor.

    Unlike ``sys.stdin.read(1)``, this uses ``os.read(fd, 1)`` so that
    ``select()`` and reads share the same OS-level buffer.  This prevents
    Python's ``BufferedReader`` from eagerly consuming bytes and hiding
    them from ``select()``, which caused arrow-key escape sequences to be
    misinterpreted as bare Escape.
    """

    def read_char() -> Optional[str]:
        if select.select([fd], [], [], 0.05)[0]:
            data = os.read(fd, 1)
            if data:
                return data.decode("utf-8", errors="replace")
        return None

    return read_char


def _terminal_size() -> Tuple[int, int]:
    """Return the terminal size as ``(columns, lines)``."""
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def run_picker(
    candidates: Sequence[str],
    filter_fn: Callable[[str, Sequence[str]], List[str]],
    *,
    _read_char: Optional[Callable[[], Optional[str]]] = None,
    _write: Optional[Callable[[str], None]] = None,
    _get_size: Optional[Callable[[], Tuple[int, int]]] = None,
) -> Optional[str]:
    """Run the interactive picker and return the selected path, or ``None``.

    *_read_char*, *_write* and *_get_size* are injectable for testing; when
    ``None`` they default to reading from ``sys.stdin`` (in cbreak mode),
    writing to ``sys.stdout`` and querying the real terminal size.
    """
    if _get_size is None:
        _get_size = _terminal_size

    if _write is None:

        def _write(s: str) -> None:
            sys.stdout.write(s)
            sys.stdout.flush()

    manage_terminal = _read_char is None

    if _read_char is None:
        _read_char = make_raw_reader(sys.stdin.fileno())

    state = PickerState(
        results=list(candidates),
        total=len(candidates),
    )

    # Switch to raw mode to suppress character echo during interactive input.
    # cbreak mode (used by the watcher) leaves echo enabled, which causes
    # typed characters to appear at the cursor position before the picker
    # redraws the frame — leading to duplicated first letters on screen.
    old_attrs = None
    if manage_terminal:
        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        tty.setraw(fd)

    _write(_HIDE_CURSOR)
    prev_lines = 0

    try:
        while not state.done:
            # Erase previous frame
            if prev_lines:
                _write(_CURSOR_UP * (prev_lines - 1))
                _write("\r")
                for _ in range(prev_lines):
                    _write(f"{_CLEAR_LINE}\r\n")
                _write(_CURSOR_UP * prev_lines)
                _write("\r")

            cols, rows = _get_size()
            # 2 header lines + 1 spare row so a full frame never scrolls
            state.max_visible = max(1, min(MAX_VISIBLE_RESULTS, rows - 3))
            state.cursor = min(state.cursor, state.max_visible - 1)

            frame = render(state, width=cols)
            _write(frame)
            prev_lines = _printed_line_count(frame)

            event = _read_key_event(_read_char)
            if event is None:
                continue
            update_state(state, event, filter_fn, candidates)
    finally:
        _write(_SHOW_CURSOR)
        # Move below the rendered frame so the next output starts clean
        _write("\r\n")
        # Restore the previous terminal mode
        if old_attrs is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    return state.selected
