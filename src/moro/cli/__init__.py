"""CLI module for the moro package."""

import click

from moro.cli._utils import AliasedGroup


@click.group(cls=AliasedGroup)
def cli() -> None:
    """Entry point for the CLI."""
    pass


@cli.command()
def example() -> None:
    """Example command."""
    click.echo("This is an example command.")
