from typing import Any, Dict, List

from google import genai
from google.genai import types as genai_types

from llms.providers.client_manager import ClientManager
from llms.providers.google.constants import (
    API_VERSION,
    DEFAULT_REQUEST_TIMEOUT,
    MAX_API_KEY_RETRY,
    MAX_KEY_PROCESS_COUNT,
)


class GoogleClientManager(ClientManager):
    def __init__(self) -> None:
        super().__init__(
            provider="google",
            max_api_key_retry=MAX_API_KEY_RETRY,
            max_key_process_count=MAX_KEY_PROCESS_COUNT,
        )

    def set_client(self) -> None:
        """Set the client using the API key from client manager."""
        try:
            # If no API key, fetch it
            if not self.api_key:
                self.fetch_api_key()

            # Set Google client
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=genai_types.HttpOptions(api_version=API_VERSION, timeout=DEFAULT_REQUEST_TIMEOUT),
            )
        except Exception as e:
            raise Exception(f"Error setting {self.provider} client: {e}")

    def set_aclient(self) -> None:
        # Google sync and async clients are the same
        self.set_client()

    def get_client(self) -> genai.Client:
        """Get the client using the API key from client manager.
        If client is not set, automatically try to set it by fetching the API keys."""
        return self.client if self.client else super().get_client()  # type: ignore

    def get_aclient(self) -> genai.Client:
        # Google sync and async clients are the same
        return self.get_client()


# Create a module-level instance (singleton)
_global_client_manager: GoogleClientManager = GoogleClientManager()


def get_client_manager() -> GoogleClientManager:
    global _global_client_manager
    return _global_client_manager


def get_google_models() -> List[Dict[str, Any]]:
    page = get_client_manager().get_client().models.list()
    all_models = []
    for model in page:
        all_models.append({"model_path": model.name, "data": model.model_dump()})
    return all_models
