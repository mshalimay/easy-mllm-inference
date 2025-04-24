import os

model_repo_relative_path = "./config/model_repo.yaml"
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_REPO_PATH = os.path.join(os.path.dirname(current_dir), model_repo_relative_path)
API_KEYS_PATH = "api_keys.json"
API_KEYS_REPO = "api_keys_backup.json"


API_KEY_ENV_VARS = {
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HF_TOKEN",
    "openrouter": "OPENROUTER_API_KEY",
}

ROLE_MAPPINGS = {
    "google": {
        # Assistant roles
        "assistant": "model",
        "model": "model",
        # User role
        "user": "user",
        # System role
        "system": "system",
    },
    "openai": {
        # User role
        "user": "user",
        # Assistant role
        "assistant": "assistant",
        # System roles
        "system": "system",
        "developer": "system",  # developer is the new key, but `system` is backwards compatible
    },
}


GENERATION_PREFIX_PATTERN = "-" * 10 + " GENERATION {} " + "-" * 10
