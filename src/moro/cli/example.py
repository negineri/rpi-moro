"""Example command group."""

from logging import getLogger

import click

from moro.cli._utils import AliasedGroup

logger = getLogger(__name__)


@click.group(cls=AliasedGroup)
def example() -> None:
    """Example command group."""
    pass


@example.command()
def echo() -> None:
    """Echo command."""
    click.echo("This is an example command.")
