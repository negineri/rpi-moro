"""Test suite for the CLI module."""

from unittest.mock import patch

import click
from click.testing import CliRunner

from moro.cli._utils import AliasedGroup
from moro.cli.cli import cli
from moro.config.settings import ConfigRepo


def test_example_command() -> None:
    """Test the example command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["example", "echo"])
    assert result.exit_code == 0
    assert "This is an example command." in result.output


def test_main_entry_point() -> None:
    """Test the CLI entry point in __main__.py."""
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0


def test_aliased_group() -> None:
    """Test the AliasedGroup."""
    group = AliasedGroup()

    @group.command("hello")
    def hello() -> None:
        click.echo("Hello, World!")

    ctx = click.Context(group)
    command = group.get_command(ctx, "he")
    assert command is not None
    assert command.name == "hello"
    command = group.get_command(ctx, "a")
    assert command is None


def test_config_repo_read_file_not_found() -> None:
    """Test ConfigRepo.read raises FileNotFoundError when logging config is missing."""
    repo = ConfigRepo()
    with patch("pathlib.Path.exists", return_value=False):
        try:
            repo.read()
        except FileNotFoundError as e:
            assert "Logging configuration file not found" in str(e)
