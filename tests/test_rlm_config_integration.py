"""Integration tests for RLM config-driven initialization."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from rlm import RLM, create_rlm, load_config


class TestRLMConfigIntegration:
    """Integration tests for config-driven RLM initialization."""

    @pytest.fixture
    def sample_config_path(self):
        """Create a sample config file for testing."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "test-openai-key",
                },
                "doubao": {
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    "api_key": "test-doubao-key",
                },
            },
            "models": {
                "gpt-4o": {
                    "provider": "openai",
                    "model_name": "gpt-4o",
                    "params": {"temperature": 0.7, "max_tokens": 4096},
                },
                "gpt-4o-mini": {
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                    "params": {"temperature": 0.5},
                },
                "doubao-seed": {
                    "provider": "doubao",
                    "model_name": "doubao-seed-1.6-250615",
                    "params": {"temperature": 0.3},
                },
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o", "fallback": "doubao-seed"},
                "sub_query": {"model": "gpt-4o-mini"},
                "default_answer": {"model": "gpt-4o"},
                "fallback_answer": {"model": "doubao-seed"},
            },
            "defaults": {
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "iteration_diversity": {
                "enabled": False,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_create_rlm_from_config_path(self, sample_config_path):
        """Test creating RLM from config path."""
        with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
            rlm = create_rlm(config_path=sample_config_path, environment="local")

            assert rlm is not None
            assert rlm._config is not None
            assert rlm._registry is not None
            assert rlm._selector is not None

    def test_create_rlm_from_config_object(self, sample_config_path):
        """Test creating RLM from config object."""
        config = load_config(sample_config_path)

        with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
            rlm = RLM(config=config, environment="local")

            assert rlm is not None
            assert rlm._config is config

    def test_rlm_init_with_config(self, sample_config_path):
        """Test RLM __init__ with config parameter."""
        config = load_config(sample_config_path)

        with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
            rlm = RLM(config=config, environment="local")

            assert rlm._config is config
            assert rlm._config_mode is True

    def test_rlm_init_with_config_path(self, sample_config_path):
        """Test RLM __init__ with config_path parameter."""
        with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
            rlm = RLM(config_path=sample_config_path, environment="local")

            assert rlm._config is not None
            assert rlm._config_mode is True

    def test_rlm_legacy_init(self):
        """Test RLM legacy initialization (without config)."""
        with patch("rlm.core.rlm.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.model_name = "gpt-4o"
            mock_get_client.return_value = mock_client

            rlm = RLM(
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o", "api_key": "test"},
                environment="local",
            )

            assert rlm._config_mode is False


class TestRLMConfigRoles:
    """Tests for role-based model selection."""

    @pytest.fixture
    def config_with_roles(self):
        """Create a config with different models per role."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "test-key",
                },
            },
            "models": {
                "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
                "gpt-4o-mini": {"provider": "openai", "model_name": "gpt-4o-mini"},
                "gpt-3.5-turbo": {"provider": "openai", "model_name": "gpt-3.5-turbo"},
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o"},
                "sub_query": {"model": "gpt-4o-mini"},
                "default_answer": {"model": "gpt-4o"},
                "fallback_answer": {"model": "gpt-3.5-turbo"},
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

    def test_get_model_for_role(self, config_with_roles):
        """Test getting model for different roles."""
        assert config_with_roles.get_model_for_role("completion_turn") == "gpt-4o"
        assert config_with_roles.get_model_for_role("sub_query") == "gpt-4o-mini"
        assert config_with_roles.get_model_for_role("default_answer") == "gpt-4o"
        assert config_with_roles.get_model_for_role("fallback_answer") == "gpt-3.5-turbo"


class TestRLMIterationDiversity:
    """Tests for iteration diversity in RLM."""

    @pytest.fixture
    def config_with_diversity(self):
        """Create a config with iteration diversity enabled."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "test-key",
                },
            },
            "models": {
                "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
                "gpt-4o-mini": {"provider": "openai", "model_name": "gpt-4o-mini"},
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o"},
                "sub_query": {"model": "gpt-4o-mini"},
                "default_answer": {"model": "gpt-4o"},
                "fallback_answer": {"model": "gpt-4o"},
            },
            "iteration_diversity": {
                "enabled": True,
                "strategy": "round_robin",
                "models": ["gpt-4o", "gpt-4o-mini"],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_rlm_with_diversity(self, config_with_diversity):
        """Test RLM initialization with diversity enabled."""
        with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
            rlm = RLM(config_path=config_with_diversity, environment="local")

            assert rlm._selector is not None
            assert rlm._selector.diversity_enabled is True
            assert rlm._selector.strategy == "round_robin"

    def test_get_model_for_iteration(self, config_with_diversity):
        """Test model selection per iteration with diversity."""
        with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
            rlm = RLM(config_path=config_with_diversity, environment="local")

            model0 = rlm._get_model_for_iteration(0)
            model1 = rlm._get_model_for_iteration(1)
            model2 = rlm._get_model_for_iteration(2)

            assert model0 == "gpt-4o"
            assert model1 == "gpt-4o-mini"
            assert model2 == "gpt-4o"  # Wraps around


class TestRLMBackwardCompatibility:
    """Tests for backward compatibility with legacy initialization."""

    def test_legacy_params_still_work(self):
        """Test that legacy parameters still work."""
        with patch("rlm.core.rlm.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.model_name = "gpt-4o"
            mock_get_client.return_value = mock_client

            rlm = RLM(
                backend="openai",
                backend_kwargs={
                    "model_name": "gpt-4o",
                    "api_key": "test-key",
                },
                environment="local",
                max_depth=2,
                max_iterations=20,
                verbose=False,
            )

            assert rlm.max_depth == 2
            assert rlm.max_iterations == 20
            # verbose is a VerbosePrinter object, check enabled attribute
            assert rlm.verbose.enabled is False

    def test_config_overrides_legacy(self):
        """Test that config mode takes precedence when both specified."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "config-key",
                },
            },
            "models": {
                "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o"},
                "sub_query": {"model": "gpt-4o"},
                "default_answer": {"model": "gpt-4o"},
                "fallback_answer": {"model": "gpt-4o"},
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
                # Both config and legacy params specified
                rlm = RLM(
                    config_path=temp_path,
                    backend="anthropic",  # Should be ignored
                    backend_kwargs={"model_name": "claude-3"},  # Should be ignored
                    environment="local",
                )

                assert rlm._config_mode is True
        finally:
            os.unlink(temp_path)


class TestCreateRLMFunction:
    """Tests for create_rlm convenience function."""

    def test_create_rlm_without_config_uses_legacy(self):
        """Test that create_rlm without config uses legacy mode."""
        with patch("rlm.core.rlm.get_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.model_name = "gpt-4o"
            mock_get_client.return_value = mock_client

            rlm = create_rlm(
                environment="local",
                backend="openai",
                backend_kwargs={"model_name": "gpt-4o"},
            )

            assert rlm._config_mode is False

    def test_create_rlm_with_environment_kwargs(self):
        """Test create_rlm passes environment_kwargs."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "test-key",
                },
            },
            "models": {
                "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o"},
                "sub_query": {"model": "gpt-4o"},
                "default_answer": {"model": "gpt-4o"},
                "fallback_answer": {"model": "gpt-4o"},
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with patch("rlm.clients.openai_compatible.OpenAICompatibleClient"):
                rlm = create_rlm(
                    config_path=temp_path,
                    environment="local",
                    max_iterations=50,
                    verbose=True,
                )

                assert rlm.max_iterations == 50
                assert rlm.verbose.enabled is True
        finally:
            os.unlink(temp_path)
