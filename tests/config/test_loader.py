"""Tests for ConfigLoader."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from rlm.config.loader import ConfigLoader, load_config
from rlm.config.types import RLMConfig


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_load_default_config(self):
        """Test loading the default config file."""
        config = load_config()
        assert isinstance(config, RLMConfig)
        assert len(config.providers) > 0
        assert len(config.models) > 0
        assert len(config.roles) > 0

    def test_load_custom_config(self):
        """Test loading a custom config file."""
        config_data = {
            "providers": {
                "test_provider": {
                    "base_url": "https://api.test.com/v1",
                    "api_key_env": "TEST_API_KEY",
                }
            },
            "models": {
                "test_model": {
                    "provider": "test_provider",
                    "model_name": "test-model-v1",
                    "params": {"temperature": 0.5},
                }
            },
            "roles": {
                "completion_turn": {"model": "test_model"},
                "sub_query": "test_model",
            },
            "defaults": {"temperature": 0.7, "max_tokens": 2048},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert "test_provider" in config.providers
            assert "test_model" in config.models
            assert config.models["test_model"].provider == "test_provider"
            assert config.models["test_model"].params["temperature"] == 0.5
            assert config.roles["completion_turn"].model == "test_model"
            assert config.roles["sub_query"].model == "test_model"
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_empty_config_error(self):
        """Test that ValueError is raised for empty config."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")  # Empty file
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Empty configuration"):
                load_config(temp_path)
        finally:
            os.unlink(temp_path)


class TestEnvVarResolution:
    """Tests for environment variable resolution."""

    def test_resolve_env_var(self):
        """Test resolving environment variables in config."""
        os.environ["TEST_RLM_API_KEY"] = "test-key-12345"

        config_data = {
            "providers": {
                "test": {
                    "base_url": "https://api.test.com/v1",
                    "api_key": "${TEST_RLM_API_KEY}",
                }
            },
            "models": {},
            "roles": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.providers["test"].api_key == "test-key-12345"
        finally:
            os.unlink(temp_path)
            del os.environ["TEST_RLM_API_KEY"]

    def test_resolve_env_var_with_default(self):
        """Test resolving env var with default value."""
        if "NONEXISTENT_VAR_FOR_TEST" in os.environ:
            del os.environ["NONEXISTENT_VAR_FOR_TEST"]

        config_data = {
            "providers": {
                "test": {
                    "base_url": "${NONEXISTENT_VAR_FOR_TEST:https://default.api.com/v1}",
                }
            },
            "models": {},
            "roles": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.providers["test"].base_url == "https://default.api.com/v1"
        finally:
            os.unlink(temp_path)

    def test_unresolved_env_var_kept(self):
        """Test that unresolved env vars without defaults are kept as-is."""
        if "TOTALLY_NONEXISTENT_VAR" in os.environ:
            del os.environ["TOTALLY_NONEXISTENT_VAR"]

        config_data = {
            "providers": {
                "test": {
                    "base_url": "https://api.test.com/v1",
                    "api_key": "${TOTALLY_NONEXISTENT_VAR}",
                }
            },
            "models": {},
            "roles": {},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config.providers["test"].api_key == "${TOTALLY_NONEXISTENT_VAR}"
        finally:
            os.unlink(temp_path)


class TestRLMConfigMethods:
    """Tests for RLMConfig methods."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config_data = {
            "providers": {
                "openai": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key_env": "OPENAI_API_KEY",
                },
                "doubao": {
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    "api_key_env": "ARK_API_KEY",
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
                "default_answer": "gpt-4o",
            },
            "iteration_diversity": {
                "enabled": True,
                "strategy": "round_robin",
                "models": ["gpt-4o", "doubao-seed"],
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

    def test_get_model(self, sample_config):
        """Test get_model method."""
        model = sample_config.get_model("gpt-4o")
        assert model.name == "gpt-4o"
        assert model.provider == "openai"
        assert model.model_name == "gpt-4o"

    def test_get_model_not_found(self, sample_config):
        """Test get_model with nonexistent model."""
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            sample_config.get_model("nonexistent")

    def test_get_provider(self, sample_config):
        """Test get_provider method."""
        provider = sample_config.get_provider("openai")
        assert provider.name == "openai"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_get_provider_not_found(self, sample_config):
        """Test get_provider with nonexistent provider."""
        with pytest.raises(ValueError, match="Provider 'nonexistent' not found"):
            sample_config.get_provider("nonexistent")

    def test_get_provider_for_model(self, sample_config):
        """Test get_provider_for_model method."""
        provider = sample_config.get_provider_for_model("doubao-seed")
        assert provider.name == "doubao"
        assert "ark.cn-beijing" in provider.base_url

    def test_get_model_for_role(self, sample_config):
        """Test get_model_for_role method."""
        assert sample_config.get_model_for_role("completion_turn") == "gpt-4o"
        assert sample_config.get_model_for_role("sub_query") == "doubao-seed"

    def test_get_fallback_for_role(self, sample_config):
        """Test get_fallback_for_role method."""
        assert sample_config.get_fallback_for_role("completion_turn") == "doubao-seed"
        assert sample_config.get_fallback_for_role("sub_query") is None


class TestConfigLoaderClassMethod:
    """Tests for ConfigLoader class method."""

    def test_load_from_path(self):
        """Test load_from_path class method."""
        config = ConfigLoader.load_from_path()
        assert isinstance(config, RLMConfig)
