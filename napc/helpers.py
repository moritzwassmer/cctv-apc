import os
from typing import Any, Dict

import yaml


def setup_mlflow(path: str) -> Dict[str, Any]:
    """Load MLFlow credentials from YAML file and set environment variables.

    Reads a YAML credentials file, validates required keys, and configures
    environment variables for MLFlow tracking server authentication.

    Args:
        path: Path to credentials YAML file containing 'uri', 'username', and 'password' keys.

    Returns:
        Dictionary containing the loaded configuration with keys 'uri', 'username', 'password'.

    Raises:
        FileNotFoundError: If credentials file does not exist at the given path.
        KeyError: If required keys ('uri', 'username', 'password') are missing from the file.
    """
    # Validate credentials file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Credentials file not found: {path}")

    # Load configuration from YAML file
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required configuration keys
    required_keys = ["uri", "username", "password"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing '{key}' in credentials YAML file.")

    # Set environment variables for MLFlow tracking
    os.environ["MLFLOW_TRACKING_URI"] = config["uri"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = config["username"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config["password"]

    return config