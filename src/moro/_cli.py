"""CLI module for the moro package."""

import click


@click.group()
def cli() -> None:
    """Entry point for the CLI."""
    pass


@cli.command()
def example() -> None:
    """Example command."""
    click.echo("This is an example command.")
