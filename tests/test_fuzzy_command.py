from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pytest_watcher.commands import FuzzyFilterCommand
from pytest_watcher.config import Config
from pytest_watcher.terminal import Terminal
from pytest_watcher.trigger import Trigger


@pytest.fixture
def command():
    return FuzzyFilterCommand()


@pytest.fixture
def trigger():
    return Trigger()


@pytest.fixture
def config(tmp_path: Path):
    return Config(path=tmp_path, runner_args=["-v", "--tb=short"])


@pytest.fixture
def mock_terminal():
    return MagicMock(spec=Terminal)


class TestFuzzyFilterCommand:
    def test_no_test_files_shows_message(
        self, command, trigger, config, mock_terminal, capsys
    ):
        """When the directory has no test files, show a message and don't trigger."""
        # Point config at an empty directory so no test files are found
        empty = config.path / "empty_subdir"
        empty.mkdir(exist_ok=True)
        config.path = empty

        command.run(trigger, mock_terminal, config)

        captured = capsys.readouterr()
        assert "No test files found" in captured.out
        assert not trigger.is_active()

    def test_selecting_a_file_sets_runner_args(
        self, command, trigger, config, mock_terminal, tmp_path
    ):
        (tmp_path / "test_foo.py").write_text("")
        (tmp_path / "test_bar.py").write_text("")

        with patch(
            "pytest_watcher.picker.run_picker", return_value="test_foo.py"
        ):
            command.run(trigger, mock_terminal, config)

        assert "test_foo.py" in config.runner_args
        assert "-v" in config.runner_args
        assert "--tb=short" in config.runner_args
        assert "test_bar.py" not in config.runner_args
        assert trigger.is_active()

    def test_cancelling_picker_does_not_trigger(
        self, command, trigger, config, mock_terminal, tmp_path
    ):
        (tmp_path / "test_foo.py").write_text("")

        with patch(
            "pytest_watcher.picker.run_picker", return_value=None
        ):
            command.run(trigger, mock_terminal, config)

        assert not trigger.is_active()
        # runner_args should remain unchanged
        assert config.runner_args == ["-v", "--tb=short"]

    def test_preserves_flags_removes_old_files(
        self, command, trigger, config, mock_terminal, tmp_path
    ):
        (tmp_path / "test_alpha.py").write_text("")
        (tmp_path / "test_beta.py").write_text("")

        config.runner_args = ["-v", "test_old.py"]

        with patch(
            "pytest_watcher.picker.run_picker", return_value="test_alpha.py"
        ):
            command.run(trigger, mock_terminal, config)

        assert config.runner_args == ["-v", "test_alpha.py"]

    def test_command_metadata(self):
        assert FuzzyFilterCommand.character == "t"
        assert FuzzyFilterCommand.show_in_menu is True

    def test_clears_screen_before_and_after(
        self, command, trigger, config, mock_terminal, tmp_path
    ):
        (tmp_path / "test_one.py").write_text("")

        with patch(
            "pytest_watcher.picker.run_picker", return_value="test_one.py"
        ):
            command.run(trigger, mock_terminal, config)

        # clear() should be called at least twice (before picker, after picker)
        assert mock_terminal.clear.call_count >= 2
