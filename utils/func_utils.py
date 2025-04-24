import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

RetryHandler = Callable[[Exception, int, Any], Tuple[bool, bool, Exception]]


class MaxRetriesExceeded(Exception):
    """Exception raised when the maximum number of retries has been exceeded for a function call."""

    def __init__(self, attempts: int, last_exception: Optional[Exception] = None):
        if last_exception is None:
            append_text = ""
        else:
            append_text = f" Last error: {last_exception}"

        message = f"Maximum number of retries exceeded after {attempts} attempts.{append_text}"
        super().__init__(message)


def dont_retry_handler(e: Exception, num_retries: int, *args: Any, **kwargs: Any) -> Tuple[bool, bool, Exception]:
    should_retry = False
    apply_delay = True
    error = e
    return should_retry, apply_delay, error


def retry_with_exponential_backoff(
    func: Callable,
    base_delay: float = 1.0,
    max_delay: float = 60,
    exp_base: float = 2,
    jitter: bool = True,
    max_retries: int = 2,
    api_errors: Tuple = (),
    custom_errors: Tuple = (),
    handle_api_errors: Optional[RetryHandler] = None,
    handle_custom_errors: Optional[RetryHandler] = None,
    handle_uncaught_errors: RetryHandler = dont_retry_handler,
    handle_max_retries: RetryHandler = dont_retry_handler,
    handle_finally: Optional[Callable] = None,
    logger=None,
    max_wait_time: float = -1,
    executor: Optional[ThreadPoolExecutor] = None,
):
    """
    Retry a function with exponential backoff.

    If max_wait_time == -1, the function is executed directly without using the executor
    since no timeout is needed. Otherwise, the function is run within the executor to enforce
    the max_wait_time.
    """

    # Use provided executor or fall back to the module-level executor.
    delete_executor = False
    if max_wait_time > 0 and executor is None:
        executor = ThreadPoolExecutor(max_workers=1)
        delete_executor = True

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = base_delay

        while True:
            try:
                if max_wait_time > 0:
                    # Execute within executor to enforce timeout.
                    future = executor.submit(func, *args, **kwargs)  # type: ignore
                    result = future.result(timeout=max_wait_time)
                else:
                    result = func(*args, **kwargs)

                return result

            except Exception as e:
                # Determine the appropriate error handler.
                if handle_custom_errors and isinstance(e, custom_errors):
                    should_retry, apply_delay, error = handle_custom_errors(e, num_retries)
                elif handle_api_errors and isinstance(e, api_errors):
                    should_retry, apply_delay, error = handle_api_errors(e, num_retries)
                else:
                    should_retry, apply_delay, error = handle_uncaught_errors(e, num_retries)

                # If we've exceeded max_retries, let handle_max_retries decide.
                if num_retries >= max_retries:
                    should_retry, apply_delay, error = handle_max_retries(MaxRetriesExceeded(num_retries), num_retries)

                if not should_retry:
                    raise error

                if apply_delay:
                    sleep_time = min(delay, max_delay)
                    if logger:
                        logger.info(f"Retrying after {sleep_time:.2f} seconds (attempt {num_retries + 1})...")
                    else:
                        print(f"Retrying after {sleep_time:.2f} seconds (attempt {num_retries + 1})...")
                    time.sleep(sleep_time)
                    delay = (delay + jitter * random.random()) * exp_base

                num_retries += 1

            finally:
                if delete_executor and executor is not None:
                    executor.shutdown(wait=False)
                if handle_finally:
                    handle_finally()

    return wrapper
