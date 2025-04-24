# type: ignore


import concurrent.futures
import functools
import random
import re
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List

from llms.generation_config import GenerationConfig
from llms.types import Message
from utils.func_utils import MaxRetriesExceeded, RetryHandler, retry_with_exponential_backoff
from utils.logger_utils import logger
from utils.timing_utils import timeit

# Global persistent thread pool to manage API call timeout.
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


class BaseGenerator:
    def __init__(
        self,
        prompter: Any,
        client_manager: Any | None = None,
        api_call_timeout: int = 3 * 60,
        retry_config: dict[str, Any] = {
            "base_delay": 1.0,
            "max_delay": 60.0,
            "exp_base": 2.0,
            "jitter": True,
            "max_retries": 2,
        },
        api_errors: tuple[type[Exception], ...] = (),
        custom_errors: tuple[type[Exception], ...] = (FutureTimeoutError,),
        provider_name: str = "baseclass",
    ):
        """
        Initialize the API provider with the provided client_manager and prompter.
        """
        # Instance cache for generation arguments and processed messages.
        self._cached: dict[str, Any] = {"provider_msgs": None, "provider_gen_config": None}
        self.client_manager = client_manager
        self.prompter = prompter
        self.provider_name: str = provider_name
        self.api_errors: tuple = api_errors
        self.custom_errors: tuple = custom_errors
        self.retry_config = retry_config
        self.api_call_timeout = api_call_timeout

    # ==========================================================================
    # Convert prompt to the provider's format
    # ==========================================================================

    def reset_provider_msgs(self, provider_msgs: Any):
        return provider_msgs

    def convert_messages(self, messages: List[Message]):
        """
        Processes the input messages by converting the prompt using the
        GooglePrompter. Also handles prompt reset or image uploading in case
        of payload issues.
        """
        provider_msgs = self._cached.get("provider_msgs")
        if not provider_msgs:
            provider_msgs = self.prompter.convert_prompt(messages)
            self._cached["provider_msgs"] = provider_msgs

        provider_msgs = self.reset_provider_msgs(provider_msgs)

        return provider_msgs

    # ==========================================================================
    # Convert generation configuration to the provider's format
    # ==========================================================================

    def convert_gen_config(self, provider_msgs: Any, gen_config: GenerationConfig) -> Any:
        """
        Constructs the generation configuration for the API call.
        """
        raise NotImplementedError("Subclasses must implement this method")

    # ==========================================================================
    # Convert API response from provider's format to universal format
    # ==========================================================================
    # TODO

    @staticmethod
    def convert_single_output(candidate: Any) -> List[Dict[str, Any]]:
        """
        Converts a single candidate output to a uniform list of output dicts.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def convert_output(cls, response: Any) -> List[List[Dict[str, Any]]]:
        """
        Converts the API response to a list of outputs.
        """
        all_outputs = []
        for candidate in response.candidates:
            all_outputs.append(cls.convert_single_output(candidate))
        return all_outputs

    # ==========================================================================
    # Provider-specific error handlers
    # ==========================================================================

    def handle_custom_errors(self, e: Exception, num_retries: int) -> tuple[bool, bool, Exception]:
        """
        Default handler for custom errors (e.g. timeout errors).
        Subclasses can override if needed.
        """
        should_retry, apply_delay, e = False, False, e
        if isinstance(e, FutureTimeoutError):
            logger.info(f"Timeout occurred (attempt {num_retries + 1}): {e}. Retrying without delay...")
            should_retry, apply_delay = True, False
        return should_retry, apply_delay, e

    def handle_api_errors(self, e: Exception, num_retries: int) -> tuple[bool, bool, Exception]:
        """
        Provider-specific error handler for API errors.
        Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the API error handler.")

    def handle_max_retries(self, e: Exception, num_retries: int) -> tuple[bool, bool, Exception]:
        """
        Error handler when max retries has been exceeded.
        Subclasses can override if needed.

        Returns:
            A tuple (should_retry, apply_delay, e) where:
            - should_retry: Whether to retry the API call.
            - apply_delay: Whether to apply a delay before retrying.
            - e: An exception to raise if any.
        """
        # Default: do not retry.
        return False, False, e

    def handle_uncaught_errors(self, e: Exception, num_retries: int) -> tuple[bool, bool, Exception]:
        """
        A generic error handler for errors not categorized above.
        """
        # Default: do not retry.
        return False, False, e

    def handle_finally(self):
        return

    @staticmethod
    def _retry_with_exponential_backoff(method):
        """
        A decorator that applies the retry logic using the instance's configuration.
        """

        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            def func():
                return method(self, *args, **kwargs)

            return retry_with_exponential_backoff(
                func,
                base_delay=self.retry_config["base_delay"],
                max_delay=self.retry_config["max_delay"],
                exp_base=self.retry_config["exp_base"],
                jitter=self.retry_config["jitter"],
                max_retries=self.retry_config["max_retries"],
                api_errors=self.api_errors,
                custom_errors=self.custom_errors,
                max_wait_time=self.api_call_timeout,
                handle_api_errors=self.handle_api_errors,
                handle_custom_errors=self.handle_custom_errors,
                handle_max_retries=self.handle_max_retries,
                handle_uncaught_errors=self.handle_uncaught_errors,
                handle_finally=self.handle_finally,
            )()

        return wrapper

    # ==========================================================================
    # Provider-specific synchronous generation
    # ==========================================================================
    def generate_from_chat_completion(self, messages: List[Message], gen_config: GenerationConfig):
        raise NotImplementedError("Subclasses must implement this method")

    async def generate_from_chat_completion_async(self, messages: List[Message], gen_config: GenerationConfig):
        raise NotImplementedError("Subclasses must implement this method")

    # ==========================================================================
    # Provider-specific asynchronous generation
    # ==========================================================================
