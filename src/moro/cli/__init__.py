"""CLI module for the moro package."""

from logging import getLogger
from logging.config import dictConfig

import click

from moro.cli._utils import AliasedGroup
from moro.config.settings import ConfigRepo

logger = getLogger(__name__)


@click.group(cls=AliasedGroup)
def cli() -> None:
    """Entry point for the CLI."""
    config = ConfigRepo().read()
    dictConfig(config.logging_config)


@cli.command()
def example() -> None:
    """Example command."""
    click.echo("This is an example command.")
