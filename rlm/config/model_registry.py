"""
Model Registry for RLM.

Manages client instances based on configuration.
Provides lazy loading and caching of clients.
"""

from typing import TYPE_CHECKING

from rlm.config.types import RLMConfig

if TYPE_CHECKING:
    from rlm.clients.base_lm import BaseLM


class ModelRegistry:
    """
    Registry that manages model clients based on configuration.

    Provides lazy loading of clients and role-based model access.
    """

    def __init__(self, config: RLMConfig):
        """
        Initialize the registry with configuration.

        Args:
            config: RLM configuration object
        """
        self.config = config
        self._clients: dict[str, "BaseLM"] = {}

    def get_client(self, model_name: str) -> "BaseLM":
        """
        Get or create a client for the specified model.

        Args:
            model_name: Name of the model (as defined in config)

        Returns:
            BaseLM client instance

        Raises:
            ValueError: If model not found in configuration
        """
        if model_name not in self._clients:
            self._clients[model_name] = self._create_client(model_name)
        return self._clients[model_name]

    def get_client_for_role(self, role: str) -> "BaseLM":
        """
        Get client for a specific role.

        Args:
            role: Role name (completion_turn, sub_query, etc.)

        Returns:
            BaseLM client instance
        """
        model_name = self.config.get_model_for_role(role)
        return self.get_client(model_name)

    def get_fallback_client(self, role: str) -> "BaseLM | None":
        """
        Get fallback client for a role.

        Args:
            role: Role name

        Returns:
            BaseLM client instance or None if no fallback
        """
        fallback_name = self.config.get_fallback_for_role(role)
        if fallback_name:
            return self.get_client(fallback_name)
        return None

    def _create_client(self, model_name: str) -> "BaseLM":
        """
        Create a new client instance for a model.

        Args:
            model_name: Name of the model

        Returns:
            BaseLM client instance
        """
        from rlm.clients.openai_compatible import OpenAICompatibleClient

        model_config = self.config.get_model(model_name)
        provider_config = self.config.get_provider(model_config.provider)

        return OpenAICompatibleClient(
            model_name=model_config.model_name,
            base_url=provider_config.base_url,
            api_key=provider_config.get_api_key(),
            params=model_config.params,
            extra_body=model_config.extra_body,
            api_version=provider_config.api_version,
        )

    def register_all(self) -> None:
        """Pre-create all clients defined in configuration."""
        for model_name in self.config.models:
            self.get_client(model_name)

    def get_all_clients(self) -> dict[str, "BaseLM"]:
        """
        Get all created clients.

        Returns:
            Dictionary of model_name -> client
        """
        return self._clients.copy()

    def clear(self) -> None:
        """Clear all cached clients."""
        self._clients.clear()
