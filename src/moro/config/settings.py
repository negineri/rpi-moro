"""
Configuration module for managing application settings.

This module provides classes and methods to handle application configuration,
including reading environment variables and setting up logging.
"""

from dataclasses import dataclass
from os import getenv
from os.path import dirname, join
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from yaml import safe_load

ENV_PREFIX = "MORO_"

# Domain Object


@dataclass
class Config:
    """
    Configuration class for the application.

    Attributes:
        jobs (int): Number of jobs for processing.
        logging_config_path (str): Path to the logging configuration file.
    """

    jobs: int  # Number of jobs for processing
    logging_config: dict[str, Any]  # Logging configuration


# Repository Implementation


@dataclass
class ConfigRepo:
    """Repository for configuration."""

    def read(self) -> Config:
        """
        Read configuration.

        Returns:
            Config: Configuration object

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        load_dotenv()

        jobs = int(getenv(f"{ENV_PREFIX}JOBS", "16"))
        logging_config_path = Path(
            getenv(f"{ENV_PREFIX}LOGGING_CONFIG_PATH", join(dirname(__file__), "logging.yml"))
        )

        if logging_config_path.exists():
            with open(logging_config_path) as f:
                logging_config = safe_load(f)
        else:
            raise FileNotFoundError(f"Logging configuration file not found: {logging_config_path}")

        return Config(
            jobs=jobs,
            logging_config=logging_config,
        )
