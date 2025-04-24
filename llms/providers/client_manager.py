import json
import os
import random
from typing import Any, Optional

from filelock import FileLock, Timeout

from llms.constants import API_KEY_ENV_VARS
from llms.constants.constants import API_KEYS_PATH
from llms.setup_utils import restore_api_keys_to_file, safe_add_key_to_file, safe_remove_key_from_file
from utils.concurrency_utils import get_file_lock
from utils.logger_utils import logger
from utils.signal_utils import signal_manager
from utils.timing_utils import timeit

# Default value
MAX_API_KEY_RETRY = 2
MAX_KEY_PROCESS_COUNT = 1
signal_manager.add_cleanup_function(restore_api_keys_to_file)
LOCK_TIMEOUT = 60


class NoAPIKeyException(Exception):
    """
    Custom exception raised when no API key is available.
    """

    def __init__(self, message: str = "No API keys available.") -> None:
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class ClientManager:
    def __init__(
        self,
        provider: str,
        max_api_key_retry: int | float = MAX_API_KEY_RETRY,
        max_key_process_count: int | float = MAX_KEY_PROCESS_COUNT,
        api_key_env_var: str | None = None,
    ) -> None:
        self.api_key: Optional[str] = None
        self.client: Optional[Any] = None
        self.aclient: Optional[Any] = None
        self.max_api_key_retry = max_api_key_retry  # Maximum number of times to retry an API key
        self.api_keys_retry_count: dict[int, int] = {}  # Track the number of times an API key has been used
        self.provider = provider
        self.api_key_env_var = api_key_env_var or API_KEY_ENV_VARS[provider]
        self.api_process_count: dict[int, int] = {}
        self.max_key_process_count = max_key_process_count

    def fetch_api_key(self) -> None:
        try:
            if os.environ.get(self.api_key_env_var):
                # Fetch API key from environment variables if available
                self.api_key = os.environ[self.api_key_env_var]
                self.api_keys_retry_count[hash(self.api_key)] = 0
                self.api_process_count[hash(self.api_key)] = 1
                # If API key in env variables, remove it from repo to prevent other processes from using it
                try:
                    if self.api_process_count[hash(self.api_key)] >= self.max_key_process_count:
                        safe_remove_key_from_file(self.api_key, provider=self.provider, logger=logger)
                except FileNotFoundError:
                    pass
            else:
                # Fetch API key from local file if available
                self.api_key = self._get_add_remove_key_from_file()
                if self.api_key:
                    os.environ[self.api_key_env_var] = self.api_key
                else:
                    raise NoAPIKeyException(f"No API keys available during initialization for {self.provider} client")
        except Exception as e:
            raise Exception(f"Error while fetching {self.provider} API key: {e}")

    def _get_add_remove_key_from_file(self, previous_api_key: str | None = None) -> str | None:
        """Safely retrieve and remove an API key from the shared file."""
        try:
            lock = get_file_lock(API_KEYS_PATH, timeout=LOCK_TIMEOUT)
            with lock, open(API_KEYS_PATH, "r+") as file:
                # Read all API keys
                api_keys = json.load(file)
                provider_keys = api_keys.get(self.provider, [])

                # If previous API key provided (changing keys), decrease num process using it
                if previous_api_key:
                    self.api_process_count[hash(previous_api_key)] -= 1

                # If previous API key provided hasn't exceeded retry limit add it back to the end of the list
                if (
                    previous_api_key
                    and self.api_keys_retry_count.get(hash(previous_api_key), 0) < self.max_api_key_retry
                ):
                    if provider_keys and previous_api_key not in provider_keys:
                        provider_keys.append(previous_api_key)
                    else:
                        provider_keys = [previous_api_key]

                # If no API keys available, return None
                if not provider_keys:
                    return None

                # Try to get an API key that hasn't exceeded the process count limit
                api_key = None
                while len(provider_keys) > 0:
                    # Take a random API key from the list
                    idx = random.randint(0, len(provider_keys) - 1)
                    api_key = provider_keys.pop(idx)
                    hash_api_key = hash(api_key)

                    # Initialize retry count if not exists
                    if hash_api_key not in self.api_keys_retry_count:
                        self.api_keys_retry_count[hash_api_key] = 0

                    # Initialize process count if not exists
                    if hash_api_key not in self.api_process_count:
                        self.api_process_count[hash_api_key] = 0

                    proposed_process_count = self.api_process_count[hash_api_key] + 1
                    if proposed_process_count > self.max_key_process_count:
                        api_key = None
                        continue
                    else:
                        # Update process count
                        self.api_process_count[hash_api_key] = proposed_process_count

                        # If not reaching the limit, add it back for potential reuse.
                        if proposed_process_count < self.max_key_process_count:
                            provider_keys.append(api_key)
                        break

                # Reset file position and write updated keys
                api_keys[self.provider] = provider_keys
                file.seek(0)
                json.dump(api_keys, file, indent=2)
                file.truncate()

                return api_key  # type: ignore
        except Timeout:
            logger.info(f"Failed to acquire lock on API keys file after {LOCK_TIMEOUT} seconds")
            return None
        except Exception as e:
            logger.info(f"Error while fetching {self.provider} API key: {e}")
            return None

    @timeit(custom_name="LLM:reset_api_key")
    def reset_api_key(self) -> None:
        """
        Reset the API key by retrieving a new one and update the {provider} client.
        """
        while True:
            new_api_key = self._get_add_remove_key_from_file(previous_api_key=self.api_key)
            if not new_api_key:
                raise Exception("Resources exhausted and no other API keys available.")
            self.api_key = new_api_key
            break

        os.environ[self.api_key_env_var] = self.api_key
        self.set_client()
        logger.info(f"API key and {self.provider} client were redefined.")

    def set_aclient(self) -> None:
        """Set the async client using the API key from client manager."""
        raise NotImplementedError("Subclasses must implement this method")

    def set_client(self) -> None:
        """Set the sync client using the API key from client manager."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_client(self) -> Any:
        """Return the provider client instance."""
        if not self.client:
            try:
                self.set_client()
            except Exception as e:
                raise Exception(f"Error getting {self.provider} client: {e}")
        return self.client

    def get_aclient(self) -> Any:
        """Return the provider async client instance."""
        if not self.aclient:
            try:
                self.set_aclient()
            except Exception as e:
                raise Exception(f"Error getting {self.provider} async client: {e}")
        return self.aclient

    def close_client(self) -> None:
        # Add key back to the file
        if not self.api_key:
            return
        safe_add_key_to_file(self.api_key, provider=self.provider, logger=None)
        self.api_key = None
        self.client = None
        self.aclient = None
