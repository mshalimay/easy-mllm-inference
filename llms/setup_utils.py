import json
import os
import re
from typing import Any

import yaml
from filelock import Timeout

from utils.concurrency_utils import get_file_lock

from .constants import API_KEYS_PATH, MODEL_REPO_PATH

LOCK_TIMEOUT = 60


def infer_provider(model: str) -> str:
    if re.search(r"gemini", model, flags=re.IGNORECASE):
        return "google"
    # match cases: contains 'gpt' or starts with 'o' followed by digits
    elif re.search(r"gpt|^o\d+", model, flags=re.IGNORECASE):
        return "openai"
    elif re.search(r"claude", model, flags=re.IGNORECASE):
        return "anthropic"
    else:
        return "huggingface"


def get_model_attribute(model_name: str, attribute: str) -> str | None:
    try:
        with open(MODEL_REPO_PATH, "r") as file:
            model_config = yaml.safe_load(file)  # type: ignore
            if model_name not in model_config["models"]:
                return None
            # Get attribute from model config
            if attribute in model_config["models"][model_name]:
                return model_config["models"][model_name][attribute]  # type: ignore
            else:
                return None
    except Exception as e:
        raise ValueError(f"Error getting model attribute {attribute} for model {model_name}: {e}")


def get_provider(model: str) -> str:
    with open(MODEL_REPO_PATH, "r") as file:
        model_repo = yaml.safe_load(file)["models"]  # type: ignore
        return model_repo[model]["provider"]  # type: ignore


def add_model_to_repo(model: str, provider: str = "") -> None:
    if not provider:
        provider = infer_provider(model)

    with open(MODEL_REPO_PATH, "r") as file:
        model_repo = yaml.safe_load(file)["models"]  # type: ignore
        model_repo[model] = {"provider": provider}  # type: ignore
        with open(MODEL_REPO_PATH, "w") as file:
            yaml.dump({"models": model_repo}, file)  # type: ignore


def safe_remove_key_from_file(api_key: str, provider: str, logger: Any | None = None) -> None:
    if not os.path.exists(API_KEYS_PATH):
        return

    try:
        lock = get_file_lock(API_KEYS_PATH, timeout=LOCK_TIMEOUT)
        with lock, open(API_KEYS_PATH, "r+") as file:
            try:
                api_keys = json.load(file)

                if provider in api_keys and api_key in api_keys[provider]:
                    api_keys[provider].remove(api_key)

                    file.seek(0)
                    json.dump(api_keys, file, indent=2)
                    file.truncate()
            except Exception as e:
                if logger:
                    logger.info(f"Error removing key from file: {e}")
                else:
                    print(f"Error removing key from file: {e}")
    except Timeout:
        if logger:
            logger.info("Timeout removing key from file")
        else:
            print("Timeout removing key from file")
    except Exception as e:
        if logger:
            logger.info(f"Error removing key from file: {e}")
        else:
            print(f"Error removing key from file: {e}")


def safe_add_key_to_file(api_key: str, provider: str, logger: Any | None = None) -> None:
    if not os.path.exists(API_KEYS_PATH) or not api_key:
        return
    try:
        lock = get_file_lock(API_KEYS_PATH, timeout=LOCK_TIMEOUT)
        with lock, open(API_KEYS_PATH, "r+") as file:
            api_keys = json.load(file)

            if provider not in api_keys:
                api_keys[provider] = []

            if api_key not in api_keys[provider]:
                api_keys[provider].append(api_key)

            file.seek(0)
            json.dump(api_keys, file, indent=2)
            file.truncate()

    except Timeout:
        if logger:
            logger.info("Timeout adding key to file")
        else:
            print("Timeout adding key to file")

    except Exception as e:
        if logger:
            logger.info(f"Error adding key to file: {e}")
        else:
            print(f"Error adding key to file: {e}")


def restore_api_keys_to_file(logger: Any | None = None) -> None:
    if logger:
        logger.info("Restoring API keys to file")
    if os.getenv("GOOGLE_API_KEY"):
        safe_add_key_to_file(os.getenv("GOOGLE_API_KEY", ""), provider="google", logger=logger)
    if os.getenv("OPENAI_API_KEY"):
        safe_add_key_to_file(os.getenv("OPENAI_API_KEY", ""), provider="openai", logger=logger)
    if logger:
        logger.info("Finished restoring API keys to file")
