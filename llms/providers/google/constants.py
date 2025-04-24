# Client Manager
MAX_API_KEY_RETRY = 2
MAX_KEY_PROCESS_COUNT = 1
API_VERSION = "v1alpha"
DEFAULT_REQUEST_TIMEOUT: int | None = None  # 10 * 60 * 1000


# Prompter
ROLE_MAPPINGS = {
    "assistant": "model",
    "user": "user",
    "system": "system",
}

UPLOAD_IMAGES = False
