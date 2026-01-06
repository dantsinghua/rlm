"""
OpenAI Compatible Client for RLM.

A unified client that works with any OpenAI API compatible endpoint,
including OpenAI, Doubao (Volcengine Ark), Azure OpenAI, vLLM, etc.
"""

import os
from collections import defaultdict
from typing import Any

import openai
from dotenv import load_dotenv

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary

load_dotenv()


class OpenAICompatibleClient(BaseLM):
    """
    Unified LM Client for OpenAI API compatible endpoints.

    Supports:
    - OpenAI (api.openai.com)
    - Doubao/Volcengine Ark (ark.cn-beijing.volces.com)
    - Azure OpenAI
    - vLLM
    - Any OpenAI-compatible API
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        params: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        api_version: str | None = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI compatible client.

        Args:
            model_name: Name of the model to use
            base_url: API base URL (default: OpenAI)
            api_key: API key (or from env)
            params: Default parameters (temperature, max_tokens, etc.)
            extra_body: Provider-specific parameters (e.g., Doubao thinking)
            api_version: API version (for Azure)
            **kwargs: Additional arguments
        """
        super().__init__(model_name=model_name, **kwargs)

        # Set defaults
        if base_url is None:
            base_url = "https://api.openai.com/v1"

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.base_url = base_url
        self.params = params or {}
        self.extra_body = extra_body or {}
        self.api_version = api_version

        # Create clients
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)

        # Last call tracking
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    def completion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """
        Execute a completion request.

        Args:
            prompt: User prompt or message list
            model: Model override (optional)

        Returns:
            Completion response text
        """
        messages = self._prepare_messages(prompt)
        model = model or self.model_name

        if not model:
            raise ValueError("Model name is required")

        # Build request kwargs
        kwargs = {
            "model": model,
            "messages": messages,
        }

        # Add configured params
        if "temperature" in self.params:
            kwargs["temperature"] = self.params["temperature"]
        if "max_tokens" in self.params:
            kwargs["max_tokens"] = self.params["max_tokens"]
        if "top_p" in self.params:
            kwargs["top_p"] = self.params["top_p"]

        # Add extra_body for provider-specific params
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = self.client.chat.completions.create(**kwargs)
        self._track_usage(response, model)

        return response.choices[0].message.content

    async def acompletion(
        self, prompt: str | list[dict[str, Any]], model: str | None = None
    ) -> str:
        """
        Execute an async completion request.

        Args:
            prompt: User prompt or message list
            model: Model override (optional)

        Returns:
            Completion response text
        """
        messages = self._prepare_messages(prompt)
        model = model or self.model_name

        if not model:
            raise ValueError("Model name is required")

        # Build request kwargs
        kwargs = {
            "model": model,
            "messages": messages,
        }

        # Add configured params
        if "temperature" in self.params:
            kwargs["temperature"] = self.params["temperature"]
        if "max_tokens" in self.params:
            kwargs["max_tokens"] = self.params["max_tokens"]
        if "top_p" in self.params:
            kwargs["top_p"] = self.params["top_p"]

        # Add extra_body for provider-specific params
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        response = await self.async_client.chat.completions.create(**kwargs)
        self._track_usage(response, model)

        return response.choices[0].message.content

    def _prepare_messages(
        self, prompt: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Prepare messages for the API call.

        Args:
            prompt: User prompt or message list

        Returns:
            List of message dictionaries
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _track_usage(self, response: Any, model: str) -> None:
        """
        Track token usage from response.

        Args:
            response: API response
            model: Model name
        """
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)

            self.model_input_tokens[model] += input_tokens
            self.model_output_tokens[model] += output_tokens
            self.model_total_tokens[model] += total_tokens

            self.last_prompt_tokens = input_tokens
            self.last_completion_tokens = output_tokens
        else:
            # No usage data available
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0

    def get_usage_summary(self) -> UsageSummary:
        """Get cumulative usage summary for all models."""
        model_summaries = {}
        for model in self.model_call_counts:
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> UsageSummary:
        """Get usage summary for the last call."""
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=1,
                    total_input_tokens=self.last_prompt_tokens,
                    total_output_tokens=self.last_completion_tokens,
                )
            }
        )
