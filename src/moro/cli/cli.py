"""Entry point for the CLI."""

from logging import getLogger
from logging.config import dictConfig

import click

from moro.cli._utils import AliasedGroup
from moro.cli.example import example
from moro.config.settings import ConfigRepo

logger = getLogger(__name__)


@click.group(cls=AliasedGroup)
@click.version_option()
def cli() -> None:
    """Entry point for the CLI."""
    config = ConfigRepo().read()
    dictConfig(config.logging_config)


cli.add_command(example)
