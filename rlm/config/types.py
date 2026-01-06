"""
Configuration data types for RLM.

Defines dataclasses for provider, model, role, and overall RLM configuration.
All models are assumed to be OpenAI API compatible.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ProviderConfig:
    """Configuration for a model provider (e.g., OpenAI, Doubao, Azure)."""

    name: str
    base_url: str
    api_key: str | None = None
    api_key_env: str | None = None  # Environment variable name for API key
    api_version: str | None = None  # For Azure OpenAI
    extra: dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> str | None:
        """Get API key from direct value or environment variable."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            import os

            return os.getenv(self.api_key_env)
        return None


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str  # Alias name in config (e.g., "gpt-4o", "doubao-seed-1.6")
    provider: str  # Provider name reference
    model_name: str  # Actual model name sent to API
    params: dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, etc.
    extra_body: dict[str, Any] = field(default_factory=dict)  # Provider-specific params

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a parameter value with optional default."""
        return self.params.get(key, default)


@dataclass
class RoleConfig:
    """Configuration for a role (completion_turn, sub_query, etc.)."""

    model: str  # Primary model name
    fallback: str | None = None  # Fallback model name


@dataclass
class IterationDiversityConfig:
    """Configuration for iteration diversity (model rotation)."""

    enabled: bool = False
    strategy: Literal["round_robin", "random", "weighted"] = "round_robin"
    models: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)  # For weighted strategy


@dataclass
class DefaultsConfig:
    """Default parameters for all models."""

    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    timeout: int = 300
    max_retries: int = 3


@dataclass
class RLMConfig:
    """Complete RLM configuration."""

    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    models: dict[str, ModelConfig] = field(default_factory=dict)
    roles: dict[str, RoleConfig] = field(default_factory=dict)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    iteration_diversity: IterationDiversityConfig = field(
        default_factory=IterationDiversityConfig
    )

    def get_model(self, name: str) -> ModelConfig:
        """Get model config by name."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in configuration")
        return self.models[name]

    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider config by name."""
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not found in configuration")
        return self.providers[name]

    def get_provider_for_model(self, model_name: str) -> ProviderConfig:
        """Get provider config for a model."""
        model = self.get_model(model_name)
        return self.get_provider(model.provider)

    def get_role(self, name: str) -> RoleConfig:
        """Get role config by name."""
        if name not in self.roles:
            raise ValueError(f"Role '{name}' not found in configuration")
        return self.roles[name]

    def get_model_for_role(self, role: str) -> str:
        """Get primary model name for a role."""
        return self.get_role(role).model

    def get_fallback_for_role(self, role: str) -> str | None:
        """Get fallback model name for a role."""
        return self.get_role(role).fallback
