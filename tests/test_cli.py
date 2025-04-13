"""Test suite for the CLI module."""

from click.testing import CliRunner

from moro.cli.cli import cli


def test_example_command() -> None:
    """Test the example command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["example", "echo"])
    assert result.exit_code == 0
    assert "This is an example command." in result.output
