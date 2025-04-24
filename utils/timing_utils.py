import json
import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from utils.logger_utils import logger  # Use the central logger

# A global dictionary to accumulate timings
timings = {}
JSON_TIMINGS_FILENAME = "timings.json"
global_id = None


def set_timings_global_id(id: str | None):
    """Set an identifier to append to all timings; use None to clear."""
    global global_id
    global_id = id


def get_timing_key(custom_name: str, func_name: str) -> str:
    """Construct the timing key, prefixing with task id (if available)."""
    base = custom_name or func_name
    if global_id:
        return f"{global_id}:{base}"
    return base


def timeit(_func: Callable | None = None, *, custom_name: str = "", verbose: bool = False) -> Callable:
    def decorator_timeit(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            key = get_timing_key(custom_name, func.__name__)
            timings.setdefault(key, []).append(elapsed)
            if verbose:
                logger.debug(f"{key} took {elapsed:.4f} seconds")
            return result

        return wrapper

    if _func is None:
        # The decorator was applied with arguments.
        return decorator_timeit
    else:
        # The decorator was applied without arguments.
        return decorator_timeit(_func)


def dump_timings(save_dir: str):
    file_path = os.path.join(save_dir, JSON_TIMINGS_FILENAME)
    # Load previous timings if they exist
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                previous_timings = json.load(f)
            except json.JSONDecodeError:
                previous_timings = {}
    else:
        previous_timings = {}

    # Merge current timings with previous timings
    for key, times_list in timings.items():
        previous_timings.setdefault(key, []).extend(times_list)

    # Dump merged timings to the file
    with open(file_path, "w") as f:
        json.dump(previous_timings, f)

    # Clear the timings dictionary after dumping to free up memory
    timings.clear()


# A context manager to time little blocks of code
@contextmanager
def time_block(custom_name="block"):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        key = get_timing_key(custom_name, "")
        timings.setdefault(key, []).append(elapsed)
        logger.debug(f"{key} took {elapsed:.4f} seconds")


# Internal dictionary to hold active timer start times.
_active_timers = {}


def start(custom_name: str = "block") -> None:
    """
    Start timing a custom code block.

    Usage:
        timing.start("my_label")
        # ... code to be timed ...
        timing.end("my_label")
    """
    key = get_timing_key(custom_name, "")
    # Use a list so that multiple start calls with the same key are handled in a stack-like fashion.
    if key not in _active_timers:
        _active_timers[key] = []
    _active_timers[key].append(time.perf_counter())
    # logger.debug(f"Started timer for {key}")


def end(custom_name: str = "block", verbose: bool = False) -> None:
    """
    End timing a custom code block and record the elapsed time.

    See also:
        start(custom_name)
    """
    key = get_timing_key(custom_name, "")
    if key not in _active_timers or not _active_timers[key]:
        logger.warning(f"No active timer found for {key}.")
        return
    start_time = _active_timers[key].pop()
    elapsed = time.perf_counter() - start_time
    timings.setdefault(key, []).append(elapsed)
    if verbose:
        logger.debug(f"Ended timer for {key}, took {elapsed:.4f} seconds")
        print(f"Ended timer for {key}, took {elapsed:.4f} seconds")
