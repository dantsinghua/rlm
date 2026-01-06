"""
Model Selector for RLM.

Implements iteration diversity strategies for model rotation.
"""

import random
from typing import Literal

from rlm.config.types import RLMConfig


class ModelSelector:
    """
    Selects models based on iteration diversity strategy.

    Supports round_robin, random, and weighted strategies.
    """

    def __init__(self, config: RLMConfig):
        """
        Initialize the selector with configuration.

        Args:
            config: RLM configuration object
        """
        self.config = config
        self._iteration_count = 0

    @property
    def diversity_enabled(self) -> bool:
        """Check if iteration diversity is enabled."""
        return self.config.iteration_diversity.enabled

    @property
    def strategy(self) -> Literal["round_robin", "random", "weighted"]:
        """Get the diversity strategy."""
        return self.config.iteration_diversity.strategy

    @property
    def models(self) -> list[str]:
        """Get the list of models for diversity."""
        return self.config.iteration_diversity.models

    @property
    def weights(self) -> list[float]:
        """Get weights for weighted strategy."""
        return self.config.iteration_diversity.weights

    def select_for_iteration(self, iteration: int, role: str = "completion_turn") -> str:
        """
        Select a model for the given iteration.

        If diversity is disabled, returns the default model for the role.

        Args:
            iteration: Current iteration number (0-indexed)
            role: Role name (default: completion_turn)

        Returns:
            Model name to use
        """
        if not self.diversity_enabled or not self.models:
            # Use role's default model
            return self.config.get_model_for_role(role)

        if self.strategy == "round_robin":
            return self._round_robin(iteration)
        elif self.strategy == "random":
            return self._random()
        elif self.strategy == "weighted":
            return self._weighted()
        else:
            # Fallback to role's default
            return self.config.get_model_for_role(role)

    def _round_robin(self, iteration: int) -> str:
        """
        Round-robin model selection.

        Args:
            iteration: Current iteration number

        Returns:
            Model name
        """
        index = iteration % len(self.models)
        return self.models[index]

    def _random(self) -> str:
        """
        Random model selection.

        Returns:
            Model name
        """
        return random.choice(self.models)

    def _weighted(self) -> str:
        """
        Weighted random model selection.

        Returns:
            Model name
        """
        if not self.weights or len(self.weights) != len(self.models):
            # Fallback to uniform distribution
            return random.choice(self.models)

        # Normalize weights
        total = sum(self.weights)
        normalized = [w / total for w in self.weights]

        return random.choices(self.models, weights=normalized, k=1)[0]

    def reset(self) -> None:
        """Reset iteration counter."""
        self._iteration_count = 0
