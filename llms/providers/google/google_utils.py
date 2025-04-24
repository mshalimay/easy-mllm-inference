import functools
import os
import re
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable, List

from google.genai import types as genai_types
from google.genai.errors import APIError
from google.genai.types import SafetySetting

from llms.generation_config import GenerationConfig
from llms.providers.google.dummy_api_error import TestAPIError
from llms.providers.google.google_client_manager import get_client_manager
from llms.providers.google.prompter import GooglePrompter
from llms.retry_utils import retry_with_exponential_backoff
from llms.types import Cache, ContentItem, Message
from utils.image_utils import any_to_pil
from utils.logger_utils import logger
from utils.timing_utils import timeit

# ===============================================================================
# Globals
# ===============================================================================
# NOTE: It has been more convenient to have a file of functions and globals than
# an object-oriented solution due to multiple differences between providers that
# makes little of the code re-usable between them.
# The general structure is similar among providers though:
# 1) Convert prompt messages and generation config from uniform to provider-specific format
# 2) Call the API 'num_generations' times
# 3) Convert the API response back to list of messages in uniform format
# +: Provider-specific logic for error handling and retry with exponential backoff

# --- State control flow ---
RESET_PROMPT = False  # Controls whether to reset the prompt messages. Important if uploading files.
PAYLOAD_TOO_LARGE = False  # Controls if should upload parts of the prompt to the cloud.

# Global cache storing the provider-specific prompt messages, gen configs, api responses.
# This reduces overhead of prompt conversions and also helps control flow during multiple generations.
cache = Cache()

# --- Handling retries with exponential backoff ---

MAX_API_WAIT_TIME = 10 * 60  # Maximum wait time for overall API call before flagging as failed
MAX_WAIT_PER_GEN = 5 * 60  # Maximum wait time for each generation
MAX_RETRIES = 2  # Max retries before switching to a new API key
MAX_DELAY = 60  # Maximum delay between retries

# --- Provider configs ---
# Max size of generation batch. # TODO: implement throttled generation (see openai_utils.py)
MAX_GENERATION_PER_BATCH = 8

# Safety settings
safety_settings = [
    SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),  # type: ignore
    SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),  # type: ignore
    SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),  # type: ignore
    SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),  # type: ignore
]

# ==============================================================================
# LINK: Provider-specific Error handling and retry logic
# ==============================================================================
# Google API error documentation: https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py

# By default, always retry, apply exponential backoff, increment retries
# The handlers below can override this behavior depending on error


def handle_custom_errors(
    e: Exception,
) -> tuple[Exception, bool, bool, bool]:
    """Handle errors that are not due to the API call.

    Args:
        e (Exception): Error to handle

    Returns:
        tuple[Exception, bool, bool, bool]:
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
        `increment_retries`: Whether to increment the number of retries
    """

    if isinstance(e, TimeoutError):
        # If API just took too long to respond, and num_retries < max_retries, retry without delay
        logger.info(f"Google API didn't respond after {MAX_API_WAIT_TIME} seconds. Retrying...")
        should_retry, apply_delay, increment_retries = True, False, True
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_api_errors(
    e: APIError | TestAPIError,
) -> tuple[Exception, bool, bool, bool]:
    """Handle errors from the provider API.

    Args:
        e (APIError): Error due to the API call

    Returns:
        tuple[Exception, bool, bool, bool]:
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
        `increment_retries`: Whether to increment the number of retries
    """

    global PAYLOAD_TOO_LARGE

    # If error due to payload too large, retry without exponential backoff.
    if e.message and re.search("payload", e.message, re.IGNORECASE):
        logger.error(f"API error in Google call: {e}. Uploading prompt to cloud and retrying...")
        PAYLOAD_TOO_LARGE = True
        should_retry, apply_delay, increment_retries = True, False, False

    # If other invalid argument error, do not retry.
    elif e.status and re.search("invalid", e.status, re.IGNORECASE):
        logger.error(f"API error in Google call: {e}. Stopping generation.")
        should_retry, apply_delay, increment_retries = False, False, True
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        logger.error(f"API error in Google call: {e}. Retrying with exponential backoff...")
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_max_retries(
    e: Exception,
) -> tuple[Exception, bool, bool, bool]:
    """Specific logic in case number of exp backoff retries is hit.

    Args:
        e (Exception): Error to handle

    Returns:
        tuple[Exception, bool, bool]: (`e`, `should_retry`, `apply_delay`)
        `e`: Error to raise in case of no retry
        `should_retry`: Whether to retry the API call
        `apply_delay`: Whether to apply exp backoff delay before retrying
    """

    global RESET_PROMPT
    try:
        logger.info(f"Max retries reached for API key. Last error: {e}.\nResetting client and retrying...")

        # Update retry count for the current API key
        client_manager = get_client_manager()
        client_manager.api_keys_retry_count[hash(client_manager.api_key)] += 1
        client_manager.reset_api_key()

        RESET_PROMPT = True

        should_retry, apply_delay, increment_retries = True, False, True

        return e, should_retry, apply_delay, increment_retries
    # If no API keys left or other errors, do not retry
    except Exception as e:
        logger.error(f"{e}")
        return e, False, False, True


retry_exp_backoff = functools.partial(
    retry_with_exponential_backoff,
    base_delay=1.0,
    max_delay=MAX_DELAY,
    exp_base=2,
    jitter=True,
    max_retries=MAX_RETRIES,
    api_errors=(APIError, TestAPIError),
    custom_errors=(FutureTimeoutError,),
    handle_custom_errors=handle_custom_errors,
    handle_api_errors=handle_api_errors,
    handle_max_retries=handle_max_retries,
    max_workers=MAX_GENERATION_PER_BATCH,
)


# If API call doesnt return in min(MAX_WAIT_PER_GEN * num_generations, MAX_API_WAIT_TIME) seconds, retry
# This should be passed to the `retry_exp_backoff` decorator (see `sync_api_call`)
def timeout_getter(args: Any, kwargs: Any, key: str = "provider_gen_config") -> float:
    provider_gen_config: genai_types.GenerateContentConfig = kwargs.get(key)
    n = provider_gen_config.candidate_count if provider_gen_config.candidate_count else MAX_API_WAIT_TIME
    return min(MAX_WAIT_PER_GEN * n, MAX_API_WAIT_TIME)


# ==============================================================================
# LINK: Output conversion: provider-specific -> uniform format
# ==============================================================================
def convert_single_part(part: genai_types.Part) -> ContentItem:
    """Convert a single part to a list of content items."""
    if part.text is not None:
        return ContentItem(type="text", data=part.text, raw_model_output=part)

    elif part.inline_data is not None and part.inline_data.data is not None:
        try:
            img = any_to_pil(part.inline_data.data)
            return ContentItem(type="image", data=img, raw_model_output=part)
        except Exception as e:
            logger.error(f"Error converting inline_data to PIL Image: {e}")
            raise e

    elif part.function_call is not None:
        return ContentItem(type="function_call", data=part.to_json_dict(), raw_model_output=part)

    elif part.thought is not None:
        return ContentItem(type="reasoning", data=part.to_json_dict(), raw_model_output=part)

    else:
        raise NotImplementedError(f"Part type not implemented: {part}")


def convert_single_generation(
    candidate: genai_types.Candidate,
) -> Message | None:
    """
    Convert a single candidate to a list of content items.
    """
    if candidate.content is None:
        return None
    if candidate.content.parts is None:
        return None

    all_parsed_parts = []
    # Convert all outputs of a single generation
    for part in candidate.content.parts:
        all_parsed_parts.append(convert_single_part(part))
    if all_parsed_parts:
        return Message(role="assistant", name="", contents=all_parsed_parts)
    else:
        return None


def convert_generations(response: genai_types.GenerateContentResponse) -> list[Message]:
    all_generations = []
    if response.candidates:
        for candidate in response.candidates:
            msg = convert_single_generation(candidate)
            all_generations.append(msg) if msg else None
    return all_generations


# ==============================================================================
# LINK: Prompt messages conversion: uniform -> provider-specific format
# ==============================================================================


def get_provider_msgs(messages: List[Message]) -> List[genai_types.Content]:
    """
    Processes the input messages:
    - Converts the prompt using GooglePrompter
    - Resets the prompt if needed
    - Uploads images if payload is too large
    """
    global RESET_PROMPT, PAYLOAD_TOO_LARGE, cache

    # If no preprocessed messages, create them
    provider_msgs = cache.messages_to_provider
    if not provider_msgs:
        provider_msgs = GooglePrompter.convert_prompt(messages)
        cache.messages_to_provider = provider_msgs

    # Re-create prompt only on specific flags
    if RESET_PROMPT:
        logger.info("Resetting prompt...")
        provider_msgs = GooglePrompter.reset_prompt(provider_msgs)
        RESET_PROMPT = False
        cache.messages_to_provider = provider_msgs

    elif PAYLOAD_TOO_LARGE:
        logger.info("Payload too large. Uploading images...")
        provider_msgs = GooglePrompter.upload_all_images(provider_msgs)
        PAYLOAD_TOO_LARGE = False
        cache.messages_to_provider = provider_msgs

    return provider_msgs


# ===============================================================================
# LINK: Generation config conversion: uniform -> provider-specific format
# ===============================================================================


def gen_config_to_provider(gen_config: GenerationConfig) -> genai_types.GenerateContentConfig:
    """
    Convert the uniform generation configuration to Google API format.
    """
    # Generation arguments
    gen_args = {
        "candidate_count": gen_config.num_generations,
        "max_output_tokens": gen_config.max_tokens,
        "top_p": gen_config.top_p,
        "temperature": gen_config.temperature,
        "stop_sequences": gen_config.stop_sequences,
        "top_k": gen_config.top_k,
        "seed": gen_config.seed,
        "presence_penalty": gen_config.presence_penalty,
        "frequency_penalty": gen_config.frequency_penalty,
        "safety_settings": safety_settings,
        "response_modalities": gen_config.modalities,
    }
    provider_gen_config = genai_types.GenerateContentConfig(**gen_args)  # type: ignore
    return provider_gen_config


def regularize_provider_gen_config_for_model(
    model: str,
    provider_gen_config: genai_types.GenerateContentConfig,
) -> genai_types.GenerateContentConfig:
    """Regularize the provider generation configuration for model-specific settings.

    Args:
        model (str): Model name
        provider_gen_config (genai_types.GenerateContentConfig): Provider-specific generation configuration

    Returns:
        genai_types.GenerateContentConfig: Regularized generation configuration
    """

    # Model specific regularization
    if model == "gemini-2.0-flash-exp-image-generation":
        provider_gen_config.system_instruction = None
        provider_gen_config.candidate_count = 1
        # logger.warning(
        #     f"Warning: model {model} arguments regularized: {provider_gen_config.system_instruction} -> {None} and {provider_gen_config.candidate_count} -> {1}"
        # )

    else:
        provider_gen_config.response_modalities = ["Text"]

    return provider_gen_config


def get_provider_gen_config(
    gen_config: GenerationConfig,
    provider_msgs: List[genai_types.Content],
) -> genai_types.GenerateContentConfig:
    """
    Constructs the generation configuration to be used in the API call.
    """
    global cache
    if not cache.gen_config:
        provider_gen_config = gen_config_to_provider(gen_config)
        cache.gen_config = provider_gen_config
    else:
        provider_gen_config = cache.gen_config

    return provider_gen_config


# ==============================================================================
# Synchronous generation
# ==============================================================================


def generate_from_google_chat_completion(
    messages: List[Message],
    gen_config: GenerationConfig,
) -> tuple[list[dict[str, Any]], list[Message]]:
    """Synchronous generation from Google API.

    This function:
     - Converts prompt messages and genconfig from uniform to provider-specific format.
     - Applies model-specific regularizations to genconfigs and messages.
     - Handles multiple generations of different modalities.
     - Converts model outputs back to uniform format.

    Args:
        messages (List[Message]): List of Message objects in uniform format to send to the model.
        gen_config (GenerationConfig): Generation configuration.

    Returns:
        tuple[list[Dict[str, Any]], list[Message]]: List of API responses and list of generated messages in uniform format
    """
    global MAX_GENERATION_PER_BATCH, cache
    cache.reset()

    # Number of generations remaining to be generated
    remaining_generation_count = gen_config.num_generations

    # Build provider messages and generation config on first call
    provider_gen_config = get_provider_gen_config(gen_config, get_provider_msgs(messages))
    # (obs.: needs to rebuild prompt on each retry to cover API resets; see `sync_api_call`)

    # Generate outputs
    logger.info(f"CALLING MODEL: `{gen_config.model}`: generating {gen_config.num_generations} outputs...")

    while remaining_generation_count > 0:
        provider_gen_config.candidate_count = min(MAX_GENERATION_PER_BATCH, remaining_generation_count)

        # Regularizing here handles where model supports only `num_generations=1` by calling the API `n` times
        provider_gen_config = regularize_provider_gen_config_for_model(gen_config.model, provider_gen_config)

        # Call the API
        response: genai_types.GenerateContentResponse
        response = sync_api_call(
            model=gen_config.model,
            messages=messages,
            provider_gen_config=provider_gen_config,
        )
        model_messages = convert_generations(response)

        # Update cache and decrement remaining generation count
        if model_messages:
            cache.api_responses.append(response.model_dump(mode="json"))
            cache.model_messages.extend(model_messages)
            remaining_generation_count -= len(model_messages)

    return cache.api_responses, cache.model_messages


# NOTE: Keeping the final API call separate from main generation function
# for more isolated application of `retry_with_exponential_backoff`.
@timeit(custom_name=f"LLM:sync_{os.path.basename(__file__)}_api_call")
@retry_exp_backoff(timeout_getter=timeout_getter)
async def async_api_call(
    model: str,
    messages: List[Message],
    provider_gen_config: genai_types.GenerateContentConfig,
) -> genai_types.GenerateContentResponse:
    """Asynchronous API call to Google API."""

    # Get global client
    client = get_client_manager().get_aclient()

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages)

    # Obs.: this redundancy in system_instruction and regularization handle cases
    # Where API is redefined which must reset system_prompt message
    # (not currently the case, but possible if contain files or context caching)

    # For Google, system prompt goes into the generation config
    if provider_msgs and provider_msgs[0].parts:
        provider_gen_config.system_instruction = provider_msgs[0].parts[0]

    # Regularize provider generation config for model-specific settings.
    # (some models don't support a system_instruction)
    provider_gen_config = regularize_provider_gen_config_for_model(model, provider_gen_config)

    # Call the API
    # raise FutureTimeoutError("test")

    response = client.models.generate_content(
        model=model,
        contents=provider_msgs[1:],  # type: ignore # all msgs except sys_prompt
        config=provider_gen_config,
    )
    return response


# NOTE: Keeping the final API call separate from main generation function
# for more isolated application of `retry_with_exponential_backoff`.
@timeit(custom_name=f"LLM:sync_{os.path.basename(__file__)}_api_call")
@retry_exp_backoff(timeout_getter=timeout_getter)
def sync_api_call(
    model: str,
    messages: List[Message],
    provider_gen_config: genai_types.GenerateContentConfig,
) -> genai_types.GenerateContentResponse:
    """Synchronous API call to Google API."""

    # Get global client
    client = get_client_manager().get_client()

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages)

    # Obs.: this redundancy in system_instruction and regularization handle cases
    # Where API is redefined which must reset system_prompt message
    # (not currently the case, but possible if contain files or context caching)

    # For Google, system prompt goes into the generation config
    if provider_msgs and provider_msgs[0].parts:
        provider_gen_config.system_instruction = provider_msgs[0].parts[0]

    # Regularize provider generation config for model-specific settings.
    # (some models don't support a system_instruction)
    provider_gen_config = regularize_provider_gen_config_for_model(model, provider_gen_config)

    # Call the API
    # raise FutureTimeoutError("test")

    response = client.models.generate_content(
        model=model,
        contents=provider_msgs[1:],  # type: ignore # all msgs except sys_prompt
        config=provider_gen_config,
    )
    return response


# TODO async mode
def google_count_tokens(model: str, lm_input: str) -> int:
    client = get_client_manager().get_client()
    token_count = client.models.count_tokens(model=model, contents=lm_input)
    if token_count is None or token_count.total_tokens is None:
        logger.warning(f"Token count is None for model {model} and input {lm_input}")
        return -1
    return token_count.total_tokens
