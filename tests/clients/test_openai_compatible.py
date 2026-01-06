"""Tests for OpenAICompatibleClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rlm.clients.openai_compatible import OpenAICompatibleClient
from rlm.core.types import UsageSummary


class TestOpenAICompatibleClient:
    """Tests for OpenAICompatibleClient class."""

    @pytest.fixture
    def mock_openai(self):
        """Create mocked OpenAI client."""
        with patch("rlm.clients.openai_compatible.openai") as mock:
            mock_sync_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30
            mock_sync_client.chat.completions.create.return_value = mock_response
            mock.OpenAI.return_value = mock_sync_client

            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock.AsyncOpenAI.return_value = mock_async_client

            yield mock

    def test_init_defaults(self, mock_openai):
        """Test initialization with defaults."""
        client = OpenAICompatibleClient(model_name="gpt-4o")

        assert client.model_name == "gpt-4o"
        assert client.base_url == "https://api.openai.com/v1"
        assert client.params == {}
        assert client.extra_body == {}

    def test_init_custom_base_url(self, mock_openai):
        """Test initialization with custom base URL."""
        client = OpenAICompatibleClient(
            model_name="doubao-seed-1.6",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key="test-key",
        )

        assert client.base_url == "https://ark.cn-beijing.volces.com/api/v3"
        mock_openai.OpenAI.assert_called_with(
            api_key="test-key",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

    def test_init_with_params(self, mock_openai):
        """Test initialization with custom params."""
        client = OpenAICompatibleClient(
            model_name="gpt-4o",
            params={"temperature": 0.5, "max_tokens": 2048},
        )

        assert client.params["temperature"] == 0.5
        assert client.params["max_tokens"] == 2048

    def test_completion_string_prompt(self, mock_openai):
        """Test completion with string prompt."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        response = client.completion("Hello, world!")

        assert response == "Test response"
        mock_openai.OpenAI.return_value.chat.completions.create.assert_called_once()

        call_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello, world!"}]

    def test_completion_message_list(self, mock_openai):
        """Test completion with message list."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        response = client.completion(messages)

        assert response == "Test response"
        call_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["messages"] == messages

    def test_completion_with_model_override(self, mock_openai):
        """Test completion with model override."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        client.completion("Hello", model="gpt-4o-mini")

        call_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_completion_with_params(self, mock_openai):
        """Test completion uses configured params."""
        client = OpenAICompatibleClient(
            model_name="gpt-4o",
            api_key="test-key",
            params={"temperature": 0.3, "max_tokens": 1000, "top_p": 0.9},
        )
        client.completion("Hello")

        call_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["top_p"] == 0.9

    def test_completion_with_extra_body(self, mock_openai):
        """Test completion with extra_body."""
        client = OpenAICompatibleClient(
            model_name="doubao-seed-1.6",
            api_key="test-key",
            extra_body={"thinking": {"type": "enabled"}},
        )
        client.completion("Hello")

        call_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["extra_body"] == {"thinking": {"type": "enabled"}}

    def test_completion_no_model_error(self, mock_openai):
        """Test completion raises error when no model specified."""
        client = OpenAICompatibleClient(model_name="", api_key="test-key")
        with pytest.raises(ValueError, match="Model name is required"):
            client.completion("Hello")

    def test_invalid_prompt_type(self, mock_openai):
        """Test completion raises error for invalid prompt type."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        with pytest.raises(ValueError, match="Invalid prompt type"):
            client.completion(12345)


class TestAsyncCompletion:
    """Tests for async completion."""

    @pytest.fixture
    def mock_openai(self):
        """Create mocked OpenAI client."""
        with patch("rlm.clients.openai_compatible.openai") as mock:
            mock_sync_client = MagicMock()
            mock.OpenAI.return_value = mock_sync_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Async response"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 15
            mock_response.usage.completion_tokens = 25
            mock_response.usage.total_tokens = 40

            mock_async_client = MagicMock()
            mock_async_client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )
            mock.AsyncOpenAI.return_value = mock_async_client

            yield mock

    @pytest.mark.asyncio
    async def test_acompletion(self, mock_openai):
        """Test async completion."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        response = await client.acompletion("Hello async!")

        assert response == "Async response"
        mock_openai.AsyncOpenAI.return_value.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_acompletion_with_model_override(self, mock_openai):
        """Test async completion with model override."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        await client.acompletion("Hello", model="gpt-4o-mini")

        call_kwargs = (
            mock_openai.AsyncOpenAI.return_value.chat.completions.create.call_args.kwargs
        )
        assert call_kwargs["model"] == "gpt-4o-mini"


class TestUsageTracking:
    """Tests for usage tracking."""

    @pytest.fixture
    def mock_openai(self):
        """Create mocked OpenAI client."""
        with patch("rlm.clients.openai_compatible.openai") as mock:
            mock_sync_client = MagicMock()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30

            mock_sync_client.chat.completions.create.return_value = mock_response
            mock.OpenAI.return_value = mock_sync_client

            mock_async_client = MagicMock()
            mock.AsyncOpenAI.return_value = mock_async_client

            yield mock

    def test_usage_tracking_single_call(self, mock_openai):
        """Test usage tracking for a single call."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        client.completion("Hello")

        summary = client.get_usage_summary()
        assert isinstance(summary, UsageSummary)
        assert "gpt-4o" in summary.model_usage_summaries
        assert summary.model_usage_summaries["gpt-4o"].total_calls == 1
        assert summary.model_usage_summaries["gpt-4o"].total_input_tokens == 10
        assert summary.model_usage_summaries["gpt-4o"].total_output_tokens == 20

    def test_usage_tracking_multiple_calls(self, mock_openai):
        """Test usage tracking accumulates across calls."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        client.completion("Hello 1")
        client.completion("Hello 2")
        client.completion("Hello 3")

        summary = client.get_usage_summary()
        assert summary.model_usage_summaries["gpt-4o"].total_calls == 3
        assert summary.model_usage_summaries["gpt-4o"].total_input_tokens == 30
        assert summary.model_usage_summaries["gpt-4o"].total_output_tokens == 60

    def test_last_usage(self, mock_openai):
        """Test get_last_usage returns only last call's usage."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        client.completion("Hello 1")
        client.completion("Hello 2")

        last_usage = client.get_last_usage()
        assert last_usage.model_usage_summaries["gpt-4o"].total_calls == 1
        assert last_usage.model_usage_summaries["gpt-4o"].total_input_tokens == 10
        assert last_usage.model_usage_summaries["gpt-4o"].total_output_tokens == 20

    def test_usage_tracking_no_usage_data(self, mock_openai):
        """Test handling of responses without usage data."""
        mock_openai.OpenAI.return_value.chat.completions.create.return_value.usage = (
            None
        )

        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")
        client.completion("Hello")

        assert client.last_prompt_tokens == 0
        assert client.last_completion_tokens == 0


class TestMultiModelTracking:
    """Tests for tracking multiple models."""

    @pytest.fixture
    def mock_openai(self):
        """Create mocked OpenAI client."""
        with patch("rlm.clients.openai_compatible.openai") as mock:
            mock_sync_client = MagicMock()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Response"
            mock_response.usage = MagicMock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 20
            mock_response.usage.total_tokens = 30

            mock_sync_client.chat.completions.create.return_value = mock_response
            mock.OpenAI.return_value = mock_sync_client

            mock_async_client = MagicMock()
            mock.AsyncOpenAI.return_value = mock_async_client

            yield mock

    def test_per_model_tracking(self, mock_openai):
        """Test that different models are tracked separately."""
        client = OpenAICompatibleClient(model_name="gpt-4o", api_key="test-key")

        client.completion("Hello", model="gpt-4o")
        client.completion("Hello", model="gpt-4o-mini")
        client.completion("Hello", model="gpt-4o")

        summary = client.get_usage_summary()
        assert summary.model_usage_summaries["gpt-4o"].total_calls == 2
        assert summary.model_usage_summaries["gpt-4o-mini"].total_calls == 1
