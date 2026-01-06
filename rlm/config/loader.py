"""
Configuration loader for RLM.

Loads YAML configuration files and resolves environment variables.
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml

from rlm.config.types import (
    DefaultsConfig,
    IterationDiversityConfig,
    ModelConfig,
    ProviderConfig,
    RLMConfig,
    RoleConfig,
)

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


class ConfigLoader:
    """Loads and parses RLM configuration from YAML files."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the config loader.

        Args:
            config_path: Path to config file. If None, uses default config.
        """
        if config_path is None:
            self.config_path = DEFAULT_CONFIG_PATH
        else:
            self.config_path = Path(config_path)

    def load(self) -> RLMConfig:
        """
        Load and parse the configuration file.

        Returns:
            RLMConfig object with all configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config is invalid.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raise ValueError("Empty configuration file")

        # Resolve environment variables in the config
        resolved_config = self._resolve_env_vars(raw_config)

        # Validate and parse config
        return self._parse_config(resolved_config)

    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        Recursively resolve ${VAR_NAME} patterns in config values.

        Args:
            obj: Config object (dict, list, or value)

        Returns:
            Object with environment variables resolved.
        """
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Match ${VAR_NAME} or ${VAR_NAME:default}
            pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

            def replace_env(match):
                var_name = match.group(1)
                default = match.group(2)
                value = os.getenv(var_name)
                if value is not None:
                    return value
                if default is not None:
                    return default
                # Return original if env var not set and no default
                return match.group(0)

            return re.sub(pattern, replace_env, obj)
        else:
            return obj

    def _parse_config(self, config: dict) -> RLMConfig:
        """
        Parse raw config dict into RLMConfig object.

        Args:
            config: Raw config dictionary

        Returns:
            RLMConfig object
        """
        # Parse providers
        providers = {}
        for name, provider_data in config.get("providers", {}).items():
            providers[name] = ProviderConfig(
                name=name,
                base_url=provider_data.get("base_url", ""),
                api_key=provider_data.get("api_key"),
                api_key_env=provider_data.get("api_key_env"),
                api_version=provider_data.get("api_version"),
                extra=provider_data.get("extra", {}),
            )

        # Parse defaults
        defaults_data = config.get("defaults", {})
        defaults = DefaultsConfig(
            temperature=defaults_data.get("temperature", 0.7),
            max_tokens=defaults_data.get("max_tokens", 4096),
            top_p=defaults_data.get("top_p", 0.95),
            timeout=defaults_data.get("timeout", 300),
            max_retries=defaults_data.get("max_retries", 3),
        )

        # Parse models
        models = {}
        for name, model_data in config.get("models", {}).items():
            # Merge defaults with model params
            params = {
                "temperature": defaults.temperature,
                "max_tokens": defaults.max_tokens,
                "top_p": defaults.top_p,
            }
            params.update(model_data.get("params", {}))

            models[name] = ModelConfig(
                name=name,
                provider=model_data.get("provider", "openai"),
                model_name=model_data.get("model_name", name),
                params=params,
                extra_body=model_data.get("extra_body", {}),
            )

        # Parse roles
        roles = {}
        for name, role_data in config.get("roles", {}).items():
            if isinstance(role_data, str):
                # Simple format: role: model_name
                roles[name] = RoleConfig(model=role_data)
            else:
                roles[name] = RoleConfig(
                    model=role_data.get("model", ""),
                    fallback=role_data.get("fallback"),
                )

        # Parse iteration diversity
        diversity_data = config.get("iteration_diversity", {})
        iteration_diversity = IterationDiversityConfig(
            enabled=diversity_data.get("enabled", False),
            strategy=diversity_data.get("strategy", "round_robin"),
            models=diversity_data.get("models", []),
            weights=diversity_data.get("weights", []),
        )

        return RLMConfig(
            providers=providers,
            models=models,
            roles=roles,
            defaults=defaults,
            iteration_diversity=iteration_diversity,
        )

    @classmethod
    def load_from_path(cls, config_path: str | Path | None = None) -> RLMConfig:
        """
        Convenience method to load config from path.

        Args:
            config_path: Path to config file

        Returns:
            RLMConfig object
        """
        return cls(config_path).load()


def load_config(config_path: str | Path | None = None) -> RLMConfig:
    """
    Load RLM configuration from file.

    Args:
        config_path: Path to config file. Uses default if None.

    Returns:
        RLMConfig object
    """
    return ConfigLoader(config_path).load()
