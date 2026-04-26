"""Utility functions for working with Hydra configurations and OmegaConf."""

from typing import Any, Dict, List, Optional, Tuple, Union

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


def extract_values(**kwargs: Any) -> List[Any]:
    """
    Extract dictionary values as a list.
    
    Converts keyword arguments into a list of their values
    
    Args:
        **kwargs: Arbitrary keyword arguments to extract values from.
    
    Returns:
        List[Any]: List containing all values from the keyword arguments.
    
    Example:
        >>> extract_values(a=1, b=2, c=3)
        [1, 2, 3]
    """
    return list(kwargs.values())


def extract_tuple(**kwargs: Any) -> Tuple[Any, ...]:
    """
    Extract dictionary values as a tuple.
    
    Converts keyword arguments into a tuple of their values, preserving the order
    of insertion (Python 3.7+).
    
    Args:
        **kwargs: Arbitrary keyword arguments to extract values from.
    
    Returns:
        Tuple[Any, ...]: Tuple containing all values from the keyword arguments.
    
    Example:
        >>> extract_tuple(a=1, b=2, c=3)
        (1, 2, 3)
    """
    return tuple(kwargs.values())


def to_container(x: DictConfig) -> Union[Dict[str, Any], List[Any], Any]:
    """
    Convert OmegaConf configuration to a standard Python container.
    
    Recursively converts a DictConfig or ListConfig to standard Python dict/list,
    resolving all interpolations in the process.
    
    Args:
        x: OmegaConf configuration object to convert.
    
    Returns:
        Union[Dict[str, Any], List[Any], Any]: Standard Python container with all
            interpolations resolved. Returns dict for DictConfig, list for ListConfig,
            or primitive types for scalar values.
    """
    return OmegaConf.to_container(x, resolve=True)


def get_config(
    path: str,
    name: str,
    overrides: Optional[List[str]] = None,
    verbose: bool = True,
    instance: bool = False
) -> Union[DictConfig, Any]:
    """
    Load and compose a Hydra configuration from a specified path.
    
    Initializes a Hydra context, composes a configuration from the specified
    config file, optionally prints it, and either returns the config object
    or an instantiated object from the config.
    
    Args:
        path: Relative or absolute path to the configuration directory.
        name: Name of the configuration file (without .yaml extension).
        overrides: Optional list of configuration overrides in the format
            ["key=value", "nested.key=value"]. Defaults to None.
        verbose: Whether to print the composed configuration as YAML.
            Defaults to True.
        instance: Whether to instantiate the configuration using Hydra's
            instantiate function. If True, returns an instantiated object;
            if False, returns the configuration object. Defaults to False.
    
    Returns:
        Union[DictConfig, Any]: Either a DictConfig object (if instance=False)
            or an instantiated object from the configuration (if instance=True).
    
    """
    with initialize(version_base=None, config_path=path, job_name="test_app"):
        cfg = compose(config_name=name, overrides=overrides)

        if verbose:
            print(OmegaConf.to_yaml(cfg))

        return instantiate(cfg) if instance else cfg


def recurse_to_cfg(cfg_om: DictConfig, name: str, index: int) -> DictConfig:
    """
    Navigate through nested configuration hierarchy using a path string.
    
    Recursively traverses a configuration object following a slash-separated path
    for a specified number of levels. Useful for accessing deeply nested configuration
    sections.
    
    Args:
        cfg_om: Root configuration object to traverse.
        name: Slash-separated path string (e.g., "model/optimizer/lr").
        index: Number of path levels to traverse from the beginning.
    
    Returns:
        DictConfig: Configuration object at the specified nested level.
    
    """
    for key in name.split("/")[:index]:
        cfg_om = cfg_om[key]
    return cfg_om