from rlm.core.rlm import RLM, create_rlm

# Config-driven initialization
from rlm.config import (
    ConfigLoader,
    DefaultsConfig,
    IterationDiversityConfig,
    ModelConfig,
    ModelRegistry,
    ModelSelector,
    ProviderConfig,
    RLMConfig,
    RoleConfig,
    load_config,
)

__all__ = [
    "RLM",
    "create_rlm",
    # Config types
    "RLMConfig",
    "ProviderConfig",
    "ModelConfig",
    "RoleConfig",
    "DefaultsConfig",
    "IterationDiversityConfig",
    # Config utilities
    "ConfigLoader",
    "load_config",
    "ModelRegistry",
    "ModelSelector",
]
