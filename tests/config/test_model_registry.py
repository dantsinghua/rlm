"""Tests for ModelRegistry."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from rlm.clients.base_lm import BaseLM
from rlm.config.loader import load_config
from rlm.config.model_registry import ModelRegistry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "test-key",
                },
                "doubao": {
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    "api_key": "test-ark-key",
                },
            },
            "models": {
                "gpt-4o": {
                    "provider": "openai",
                    "model_name": "gpt-4o",
                    "params": {"temperature": 0.7},
                },
                "doubao-seed": {
                    "provider": "doubao",
                    "model_name": "doubao-seed-1.6-250615",
                    "params": {"temperature": 0.5},
                },
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o", "fallback": "doubao-seed"},
                "sub_query": {"model": "doubao-seed"},
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        config = load_config(temp_path)
        os.unlink(temp_path)
        return config

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_get_client(self, mock_client_class, sample_config):
        """Test getting a client by model name."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        client = registry.get_client("gpt-4o")

        assert client is mock_client
        mock_client_class.assert_called_once()
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["model_name"] == "gpt-4o"
        assert call_kwargs["base_url"] == "https://api.openai.com/v1"
        assert call_kwargs["api_key"] == "test-key"

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_client_caching(self, mock_client_class, sample_config):
        """Test that clients are cached."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)

        client1 = registry.get_client("gpt-4o")
        client2 = registry.get_client("gpt-4o")

        assert client1 is client2
        assert mock_client_class.call_count == 1

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_get_client_for_role(self, mock_client_class, sample_config):
        """Test getting a client by role."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        client = registry.get_client_for_role("completion_turn")

        assert client is mock_client
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["model_name"] == "gpt-4o"

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_get_fallback_client(self, mock_client_class, sample_config):
        """Test getting a fallback client for a role."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        client = registry.get_fallback_client("completion_turn")

        assert client is mock_client
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["model_name"] == "doubao-seed-1.6-250615"

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_get_fallback_client_none(self, mock_client_class, sample_config):
        """Test getting a fallback client when none is configured."""
        registry = ModelRegistry(sample_config)
        client = registry.get_fallback_client("sub_query")

        assert client is None

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_register_all(self, mock_client_class, sample_config):
        """Test pre-creating all clients."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        registry.register_all()

        assert mock_client_class.call_count == 2

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_get_all_clients(self, mock_client_class, sample_config):
        """Test getting all created clients."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        registry.get_client("gpt-4o")
        registry.get_client("doubao-seed")

        clients = registry.get_all_clients()
        assert len(clients) == 2
        assert "gpt-4o" in clients
        assert "doubao-seed" in clients

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_clear(self, mock_client_class, sample_config):
        """Test clearing cached clients."""
        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        registry.get_client("gpt-4o")

        assert len(registry.get_all_clients()) == 1

        registry.clear()
        assert len(registry.get_all_clients()) == 0

    def test_model_not_found(self, sample_config):
        """Test that ValueError is raised for unknown model."""
        registry = ModelRegistry(sample_config)
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            registry.get_client("nonexistent")

    @patch("rlm.clients.openai_compatible.OpenAICompatibleClient")
    def test_extra_body_passed(self, mock_client_class, sample_config):
        """Test that extra_body is passed to client."""
        sample_config.models["gpt-4o"].extra_body = {"custom_param": True}

        mock_client = MagicMock(spec=BaseLM)
        mock_client_class.return_value = mock_client

        registry = ModelRegistry(sample_config)
        registry.get_client("gpt-4o")

        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["extra_body"] == {"custom_param": True}
