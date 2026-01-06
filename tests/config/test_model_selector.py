"""Tests for ModelSelector."""

import os
import random
import tempfile

import pytest
import yaml

from rlm.config.loader import load_config
from rlm.config.model_selector import ModelSelector


class TestModelSelector:
    """Tests for ModelSelector class."""

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
                "gpt-3.5-turbo": {"provider": "openai", "model_name": "gpt-3.5-turbo"},
            },
            "roles": {
                "completion_turn": {"model": "gpt-4o"},
            },
            "iteration_diversity": {
                "enabled": True,
                "strategy": "round_robin",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
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

    @pytest.fixture
    def config_without_diversity(self):
        """Create a config with iteration diversity disabled."""
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

        config = load_config(temp_path)
        os.unlink(temp_path)
        return config

    def test_diversity_enabled_property(self, config_with_diversity):
        """Test diversity_enabled property."""
        selector = ModelSelector(config_with_diversity)
        assert selector.diversity_enabled is True

    def test_diversity_disabled_property(self, config_without_diversity):
        """Test diversity_enabled property when disabled."""
        selector = ModelSelector(config_without_diversity)
        assert selector.diversity_enabled is False

    def test_strategy_property(self, config_with_diversity):
        """Test strategy property."""
        selector = ModelSelector(config_with_diversity)
        assert selector.strategy == "round_robin"

    def test_models_property(self, config_with_diversity):
        """Test models property."""
        selector = ModelSelector(config_with_diversity)
        assert selector.models == ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]


class TestRoundRobinStrategy:
    """Tests for round-robin selection strategy."""

    @pytest.fixture
    def selector(self):
        """Create a selector with round-robin strategy."""
        config_data = {
            "providers": {
                "openai": {"base_url": "https://api.openai.com/v1", "api_key": "test"},
            },
            "models": {
                "model-a": {"provider": "openai", "model_name": "model-a"},
                "model-b": {"provider": "openai", "model_name": "model-b"},
                "model-c": {"provider": "openai", "model_name": "model-c"},
            },
            "roles": {"completion_turn": {"model": "model-a"}},
            "iteration_diversity": {
                "enabled": True,
                "strategy": "round_robin",
                "models": ["model-a", "model-b", "model-c"],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        config = load_config(temp_path)
        os.unlink(temp_path)
        return ModelSelector(config)

    def test_round_robin_iteration_0(self, selector):
        """Test round-robin at iteration 0."""
        model = selector.select_for_iteration(0)
        assert model == "model-a"

    def test_round_robin_iteration_1(self, selector):
        """Test round-robin at iteration 1."""
        model = selector.select_for_iteration(1)
        assert model == "model-b"

    def test_round_robin_iteration_2(self, selector):
        """Test round-robin at iteration 2."""
        model = selector.select_for_iteration(2)
        assert model == "model-c"

    def test_round_robin_wraps_around(self, selector):
        """Test round-robin wraps around after all models used."""
        model = selector.select_for_iteration(3)
        assert model == "model-a"

    def test_round_robin_large_iteration(self, selector):
        """Test round-robin with large iteration number."""
        model = selector.select_for_iteration(100)
        assert model == "model-b"  # 100 % 3 = 1


class TestRandomStrategy:
    """Tests for random selection strategy."""

    @pytest.fixture
    def selector(self):
        """Create a selector with random strategy."""
        config_data = {
            "providers": {
                "openai": {"base_url": "https://api.openai.com/v1", "api_key": "test"},
            },
            "models": {
                "model-a": {"provider": "openai", "model_name": "model-a"},
                "model-b": {"provider": "openai", "model_name": "model-b"},
            },
            "roles": {"completion_turn": {"model": "model-a"}},
            "iteration_diversity": {
                "enabled": True,
                "strategy": "random",
                "models": ["model-a", "model-b"],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        config = load_config(temp_path)
        os.unlink(temp_path)
        return ModelSelector(config)

    def test_random_returns_valid_model(self, selector):
        """Test random strategy returns a valid model."""
        model = selector.select_for_iteration(0)
        assert model in ["model-a", "model-b"]

    def test_random_with_seed(self, selector):
        """Test random strategy with seed for reproducibility."""
        random.seed(42)
        results = [selector.select_for_iteration(i) for i in range(10)]

        random.seed(42)
        results2 = [selector.select_for_iteration(i) for i in range(10)]

        assert results == results2


class TestWeightedStrategy:
    """Tests for weighted selection strategy."""

    @pytest.fixture
    def selector(self):
        """Create a selector with weighted strategy."""
        config_data = {
            "providers": {
                "openai": {"base_url": "https://api.openai.com/v1", "api_key": "test"},
            },
            "models": {
                "model-a": {"provider": "openai", "model_name": "model-a"},
                "model-b": {"provider": "openai", "model_name": "model-b"},
            },
            "roles": {"completion_turn": {"model": "model-a"}},
            "iteration_diversity": {
                "enabled": True,
                "strategy": "weighted",
                "models": ["model-a", "model-b"],
                "weights": [0.8, 0.2],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        config = load_config(temp_path)
        os.unlink(temp_path)
        return ModelSelector(config)

    def test_weighted_returns_valid_model(self, selector):
        """Test weighted strategy returns a valid model."""
        model = selector.select_for_iteration(0)
        assert model in ["model-a", "model-b"]

    def test_weighted_distribution(self, selector):
        """Test weighted strategy favors higher weight models."""
        random.seed(42)

        results = {"model-a": 0, "model-b": 0}
        for i in range(1000):
            model = selector.select_for_iteration(i)
            results[model] += 1

        ratio = results["model-a"] / 1000
        assert 0.7 < ratio < 0.9


class TestDiversityDisabled:
    """Tests for when diversity is disabled."""

    @pytest.fixture
    def selector(self):
        """Create a selector with diversity disabled."""
        config_data = {
            "providers": {
                "openai": {"base_url": "https://api.openai.com/v1", "api_key": "test"},
            },
            "models": {
                "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
            },
            "roles": {"completion_turn": {"model": "gpt-4o"}},
            "iteration_diversity": {
                "enabled": False,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        config = load_config(temp_path)
        os.unlink(temp_path)
        return ModelSelector(config)

    def test_uses_role_default(self, selector):
        """Test that role default is used when diversity disabled."""
        model = selector.select_for_iteration(0)
        assert model == "gpt-4o"

        model = selector.select_for_iteration(100)
        assert model == "gpt-4o"


class TestReset:
    """Tests for reset functionality."""

    def test_reset(self):
        """Test reset method."""
        config_data = {
            "providers": {
                "openai": {"base_url": "https://api.openai.com/v1", "api_key": "test"},
            },
            "models": {
                "gpt-4o": {"provider": "openai", "model_name": "gpt-4o"},
            },
            "roles": {"completion_turn": {"model": "gpt-4o"}},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        config = load_config(temp_path)
        os.unlink(temp_path)

        selector = ModelSelector(config)
        selector._iteration_count = 10
        selector.reset()
        assert selector._iteration_count == 0
