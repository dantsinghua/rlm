"""
RLM Configuration Module.

Provides configuration loading, model registry, and model selection.
"""

from rlm.config.loader import ConfigLoader, load_config
from rlm.config.model_registry import ModelRegistry
from rlm.config.model_selector import ModelSelector
from rlm.config.types import (
    DefaultsConfig,
    IterationDiversityConfig,
    ModelConfig,
    ProviderConfig,
    RLMConfig,
    RoleConfig,
)

__all__ = [
    # Loader
    "ConfigLoader",
    "load_config",
    # Registry and Selector
    "ModelRegistry",
    "ModelSelector",
    # Config Types
    "RLMConfig",
    "ProviderConfig",
    "ModelConfig",
    "RoleConfig",
    "DefaultsConfig",
    "IterationDiversityConfig",
]
