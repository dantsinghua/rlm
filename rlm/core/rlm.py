"""
RLM - Recursive Language Model

Main entry point for RLM functionality. Supports both legacy initialization
and config-driven initialization.
"""

import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.clients import BaseLM, get_client
from rlm.core.lm_handler import LMHandler
from rlm.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
)
from rlm.environments import BaseEnv, get_environment
from rlm.logger import RLMLogger, VerbosePrinter
from rlm.utils.parsing import (
    find_code_blocks,
    find_final_answer,
    format_iteration,
)
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from rlm.utils.rlm_utils import filter_sensitive_keys

if TYPE_CHECKING:
    from rlm.config import ModelRegistry, ModelSelector, RLMConfig


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.

    Each completion() call spawns its own environment and LM handler, which are
    cleaned up when the call completes.

    Supports two initialization modes:
    1. Config-driven: Pass config or config_path to load from YAML
    2. Legacy: Pass backend, backend_kwargs directly (backward compatible)
    """

    def __init__(
        self,
        # Config-driven mode
        config: "RLMConfig | None" = None,
        config_path: str | Path | None = None,
        # Legacy mode (backward compatible)
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
    ):
        """
        Initialize RLM instance.

        Config-driven mode (new):
            config: RLMConfig object loaded from YAML
            config_path: Path to config YAML file

        Legacy mode (backward compatible):
            backend: The backend to use for the RLM.
            backend_kwargs: The kwargs to pass to the backend.
            environment: The environment to use for the RLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RLM (0-indexed).
            max_depth: The maximum depth of the RLM.
            max_iterations: The maximum number of iterations of the RLM.
            custom_system_prompt: The custom system prompt to use for the RLM.
            other_backends: A list of other client backends for sub-calls.
            other_backend_kwargs: The kwargs to pass to the other backends.
            logger: The logger to use for the RLM.
            verbose: Whether to print verbose output.
        """
        # Common attributes
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)
        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT

        # Config-driven mode
        self._config: "RLMConfig | None" = None
        self._registry: "ModelRegistry | None" = None
        self._selector: "ModelSelector | None" = None
        self._config_mode = False

        if config is not None or config_path is not None:
            self._init_from_config(config, config_path, environment, environment_kwargs, verbose)
        else:
            self._init_legacy(
                backend, backend_kwargs, environment, environment_kwargs,
                other_backends, other_backend_kwargs, verbose
            )

    def _init_from_config(
        self,
        config: "RLMConfig | None",
        config_path: str | Path | None,
        environment: EnvironmentType,
        environment_kwargs: dict[str, Any] | None,
        verbose: bool,
    ) -> None:
        """Initialize from configuration file."""
        from rlm.config import ConfigLoader, ModelRegistry, ModelSelector

        # Load config if not provided
        if config is None:
            loader = ConfigLoader(config_path)
            config = loader.load()

        self._config = config
        self._registry = ModelRegistry(config)
        self._selector = ModelSelector(config)
        self._config_mode = True

        # Get primary model info for legacy compatibility
        primary_role = "completion_turn"
        if primary_role in config.roles:
            primary_model_name = config.get_model_for_role(primary_role)
            primary_model = config.get_model(primary_model_name)
            self.backend = "openai_compatible"
            self.backend_kwargs = {
                "model_name": primary_model.model_name,
                "params": primary_model.params,
            }
        else:
            # Fallback to first model in config
            if config.models:
                first_model = next(iter(config.models.values()))
                self.backend = "openai_compatible"
                self.backend_kwargs = {
                    "model_name": first_model.model_name,
                    "params": first_model.params,
                }
            else:
                self.backend = "openai"
                self.backend_kwargs = {}

        # Environment settings
        self.environment_type = environment
        self.environment_kwargs = environment_kwargs.copy() if environment_kwargs else {}

        # Not used in config mode but kept for compatibility
        self.other_backends = None
        self.other_backend_kwargs = None

        # Log metadata
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=self.backend_kwargs.get("model_name", "config-driven"),
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                backend="config-driven",
                backend_kwargs={"config_mode": True},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs) if environment_kwargs else {},
                other_backends=list(self._config.models.keys()) if self._config else None,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    def _init_legacy(
        self,
        backend: ClientBackend,
        backend_kwargs: dict[str, Any] | None,
        environment: EnvironmentType,
        environment_kwargs: dict[str, Any] | None,
        other_backends: list[ClientBackend] | None,
        other_backend_kwargs: list[dict[str, Any]] | None,
        verbose: bool,
    ) -> None:
        """Initialize with legacy parameters (backward compatible)."""
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = environment_kwargs.copy() if environment_kwargs else {}
        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        # Log metadata
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown") if backend_kwargs else "unknown",
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs) if environment_kwargs else {},
                other_backends=other_backends,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]):
        """
        Spawn an LM handler and environment for a single completion call.
        Cleans up both when the context exits.
        """
        if self._config_mode and self._registry:
            # Config-driven mode: use registry to get clients
            lm_handler = self._create_handler_from_config()
        else:
            # Legacy mode
            lm_handler = self._create_handler_legacy()

        lm_handler.start()

        # Pass handler address to environment so it can make llm_query() calls
        env_kwargs = self.environment_kwargs.copy()
        env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
        env_kwargs["context_payload"] = prompt

        # Initialize the environment
        environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

        try:
            yield lm_handler, environment
        finally:
            # Cleanup
            lm_handler.stop()
            if hasattr(environment, "cleanup"):
                environment.cleanup()

    def _create_handler_from_config(self) -> LMHandler:
        """Create LMHandler from config-driven registry."""
        # Get primary client for completion_turn role
        primary_client = self._registry.get_client_for_role("completion_turn")
        lm_handler = LMHandler(primary_client)

        # Register all configured models
        for model_name in self._config.models:
            client = self._registry.get_client(model_name)
            lm_handler.register_client(model_name, client)
            # Also register by actual model name
            model_config = self._config.get_model(model_name)
            if model_config.model_name != model_name:
                lm_handler.register_client(model_config.model_name, client)

        return lm_handler

    def _create_handler_legacy(self) -> LMHandler:
        """Create LMHandler using legacy parameters."""
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        lm_handler = LMHandler(client)

        # Register other clients
        if self.other_backends and self.other_backend_kwargs:
            for backend, kwargs in zip(self.other_backends, self.other_backend_kwargs, strict=True):
                other_client: BaseLM = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        return lm_handler

    def _setup_prompt(self, prompt: str | dict[str, Any]) -> list[dict[str, Any]]:
        """
        Setup the system prompt for the RLM. Also include metadata about the prompt and build
        up the initial message history.
        """
        metadata = QueryMetadata(prompt)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt, query_metadata=metadata
        )

        return message_history

    def _get_model_for_iteration(self, iteration: int, role: str = "completion_turn") -> str | None:
        """
        Get the model to use for a given iteration.

        Returns model name if in config mode with diversity enabled, None otherwise.
        """
        if self._config_mode and self._selector:
            return self._selector.select_for_iteration(iteration, role)
        return None

    def _get_root_model_name(self) -> str:
        """Get the root model name for logging/tracking."""
        if self._config_mode and self._config:
            try:
                model_name = self._config.get_model_for_role("completion_turn")
                return self._config.get_model(model_name).model_name
            except (ValueError, KeyError):
                pass
        if self.backend_kwargs:
            return self.backend_kwargs.get("model_name", "unknown")
        return "unknown"

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """
        Recursive Language Model completion call. This is the main entry point for querying an RLM, and
        can replace a regular LM completion call.

        Spawns its own environment and LM handler for the duration of this call.

        Args:
            prompt: A single string or dictionary of messages to pass as context to the model.
            root_prompt: We allow the RLM's root LM to see a (small) prompt that the user specifies. A common example of this
            is if the user is asking the RLM to answer a question, we can pass the question as the root prompt.
        Returns:
            A final answer as a string.
        """
        time_start = time.perf_counter()

        # If we're at max depth, the RLM is an LM, so we fallback to the regular LM.
        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt)

            for i in range(self.max_iterations):
                # Current prompt = message history + additional prompt suffix
                current_prompt = message_history + [build_user_prompt(root_prompt, i)]

                # Get model for this iteration (supports diversity)
                model = self._get_model_for_iteration(i, "completion_turn")

                iteration: RLMIteration = self._completion_turn(
                    prompt=current_prompt,
                    lm_handler=lm_handler,
                    environment=environment,
                    model=model,
                )

                # Check if RLM is done and has a final answer.
                final_answer = find_final_answer(iteration.response, environment=environment)
                iteration.final_answer = final_answer

                # If logger is used, log the iteration.
                if self.logger:
                    self.logger.log(iteration)

                # Verbose output for this iteration
                self.verbose.print_iteration(iteration, i + 1)

                if final_answer is not None:
                    time_end = time.perf_counter()
                    usage = lm_handler.get_usage_summary()
                    self.verbose.print_final_answer(final_answer)
                    self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())
                    return RLMChatCompletion(
                        root_model=self._get_root_model_name(),
                        prompt=prompt,
                        response=final_answer,
                        usage_summary=usage,
                        execution_time=time_end - time_start,
                    )

                # Format the iteration for the next prompt.
                new_messages = format_iteration(iteration)

                # Update message history with the new messages.
                message_history.extend(new_messages)

            # Default behavior: we run out of iterations, provide one final answer
            time_end = time.perf_counter()

            # Get model for default answer
            default_model = self._get_model_for_role("default_answer")
            final_answer = self._default_answer(message_history, lm_handler, model=default_model)

            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())
            return RLMChatCompletion(
                root_model=self._get_root_model_name(),
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
            )

    def _get_model_for_role(self, role: str) -> str | None:
        """Get model name for a specific role."""
        if self._config_mode and self._config:
            try:
                return self._config.get_model_for_role(role)
            except (ValueError, KeyError):
                pass
        return None

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
        model: str | None = None,
    ) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.

        Args:
            prompt: The prompt to send to the model
            lm_handler: The LM handler to use
            environment: The execution environment
            model: Optional model name to use (for iteration diversity)
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt, model=model)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(
        self,
        message_history: list[dict[str, Any]],
        lm_handler: LMHandler,
        model: str | None = None,
    ) -> str:
        """
        Default behavior if the RLM runs out of iterations and does not find a final answer.
        It will take the message history, and try to generate a final answer from it.

        Args:
            message_history: The accumulated message history
            lm_handler: The LM handler to use
            model: Optional model name to use
        """
        current_prompt = message_history + [
            {
                "role": "assistant",
                "content": "Please provide a final answer to the user's question based on the information provided.",
            }
        ]
        response = lm_handler.completion(current_prompt, model=model)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=response,
                    code_blocks=[],
                )
            )

        return response

    def _fallback_answer(self, message: str | dict[str, Any]) -> RLMChatCompletion:
        """
        Fallback behavior if the RLM is actually at max depth, and should be treated as an LM.
        """
        time_start = time.perf_counter()

        if self._config_mode and self._registry:
            # Use fallback_answer role model
            model_name = self._get_model_for_role("fallback_answer")
            if model_name:
                client = self._registry.get_client(model_name)
            else:
                client = self._registry.get_client_for_role("completion_turn")
        else:
            client = get_client(self.backend, self.backend_kwargs)

        response = client.completion(message)
        time_end = time.perf_counter()

        return RLMChatCompletion(
            root_model=self._get_root_model_name(),
            prompt=message,
            response=response,
            usage_summary=client.get_usage_summary(),
            execution_time=time_end - time_start,
        )


def create_rlm(
    config_path: str | Path | None = None,
    **kwargs,
) -> RLM:
    """
    Convenience function to create an RLM instance.

    If config_path is provided, loads configuration from file.
    Otherwise, uses default configuration or passed kwargs.

    Args:
        config_path: Path to configuration YAML file
        **kwargs: Additional arguments passed to RLM constructor

    Returns:
        RLM instance
    """
    if config_path is not None:
        return RLM(config_path=config_path, **kwargs)
    return RLM(**kwargs)
