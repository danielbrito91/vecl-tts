"""
Factory for creating model strategies based on the application configuration.
"""

from vecl.config import AppConfig
from vecl.model.strategy.base import BaseModelStrategy
from vecl.model.strategy.vecl_strategy import VeclStrategy
from vecl.model.strategy.yourtts_strategy import YourTTSStrategy


def get_model_strategy(config: AppConfig) -> BaseModelStrategy:
    """
    Returns the appropriate model strategy based on the model type specified
    in the configuration.

    Args:
        config (AppConfig): The application configuration object.

    Returns:
        BaseModelStrategy: An instance of a concrete model strategy.

    Raises:
        ValueError: If the model type in the config is unsupported.
    """
    model_type = config.model.type
    if model_type == 'vecl':
        return VeclStrategy(config)
    if model_type == 'yourtts':
        return YourTTSStrategy(config)
    raise ValueError(f"Unsupported model type in config: '{model_type}'")
