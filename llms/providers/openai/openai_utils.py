import asyncio
import functools
import logging
import os
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Dict, List

import aiolimiter
import openai
from openai import AsyncOpenAI, BadRequestError, InternalServerError, NotFoundError, OpenAIError, RateLimitError
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.responses.response import Response
from openai.types.responses.response_output_item import ResponseOutputItem
from openai.types.responses.response_output_message import ResponseOutputMessage

from llms.generation_config import GenerationConfig
from llms.providers.openai.constants import DEFAULT_OPENAI_MODE
from llms.providers.openai.dummy_api_error import TestAPIError
from llms.providers.openai.openai_client_manager import get_client_manager
from llms.providers.openai.prompter import OpenAIPrompter
from llms.retry_utils import retry_with_exponential_backoff
from llms.types import Cache, ContentItem, Message
from utils.image_utils import is_image
from utils.logger_utils import logger
from utils.timing_utils import timeit
from utils.types import ImageInput

# ===============================================================================
# Globals
# ===============================================================================
# --- State control flow ---
# OBS: Not implemented for OpenAI, but may be useful in the future.
# RESET_PROMPT = False  # Whether to reset the prompt messages. Important if changing API keys and uploading files.
# PAYLOAD_TOO_LARGE = False  # Whether to upload parts of the prompt to the cloud.

# Global cache storing the provider-specific prompt messages, gen configs, api responses.
# This reduces overhead of prompt conversions and also helps control flow during multiple generations.
cache = Cache()

# --- Handling retries with exponential backoff ---

MAX_API_WAIT_TIME = 5 * 60  # Maximum wait time for overall API call before flagging as failed
MAX_WAIT_PER_GEN = 2 * 60  # Maximum wait time for each generation
MAX_RETRIES = 10  # Max retries before declaring failure for an API key
MAX_DELAY = 60 * 2  # Maximum delay between retries

# --- Provider configs ---
# Max size of generation batch. # TODO: use throttled generation instead
MAX_GENERATION_PER_BATCH = 8


# Global persistent thread pool to manage API call timeout. Create it here to avoid overhead of creating it on each call.


# ===============================================================================
# LINK Provider-specific Error handling and retry logic
# ===============================================================================


def handle_custom_errors(e: Exception) -> tuple[Exception, bool, bool, bool]:
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
    # By default, retry, apply exponential backoff, increment retries
    should_retry, apply_delay, increment_retries = True, True, True

    if isinstance(e, FutureTimeoutError):
        # If API just took too long to respond, and num_retries < max_retries, retry without delay
        logger.info(f"OpenAI API didn't respond after {MAX_API_WAIT_TIME} seconds. Retrying...")

        # Re-set the client (this cover cases where IP of machine changes).
        get_client_manager().set_client()
        should_retry, apply_delay, increment_retries = True, False, True
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        should_retry, apply_delay, increment_retries = True, True, True

    return e, should_retry, apply_delay, increment_retries


def handle_api_errors(e: OpenAIError | TestAPIError) -> tuple[Exception, bool, bool, bool]:
    """
    Handle errors raised by the OpenAI API call.
    """
    # By default, retry, apply exponential backoff, increment retries
    should_retry, apply_delay, increment_retries = True, True, True

    if isinstance(e, BadRequestError):
        logger.error(f"BadRequestError during OpenAI API call: {e}.")
        # Do not retry on bad requests.
        should_retry, apply_delay, increment_retries = False, False, False

    elif isinstance(e, NotFoundError):
        logger.error(f"NotFoundError during OpenAI API call: {e}. Stopping generation.")
        # Do not retry on not found errors.
        should_retry, apply_delay, increment_retries = False, False, False
    else:
        # All other errors: retry with exponential backoff and increment the number of retries
        logger.error(f"OpenAI API error: {e}. Retrying...")
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

    # global RESET_PROMPT
    try:
        # Update retry count for the current API key
        client_manager = get_client_manager()
        client_manager.api_keys_retry_count[hash(client_manager.api_key)] += 1
        client_manager.reset_api_key()
        # RESET_PROMPT = True

        # If manages to redefine API key, retry without delay
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
    api_errors=(OpenAIError, TestAPIError),
    custom_errors=(FutureTimeoutError,),
    handle_custom_errors=handle_custom_errors,
    handle_api_errors=handle_api_errors,
    handle_max_retries=handle_max_retries,
    max_workers=MAX_GENERATION_PER_BATCH,
)


# If API call doesnt return in min(MAX_WAIT_PER_GEN * num_generations, MAX_API_WAIT_TIME) seconds, retry
# This should be passed to the `retry_exp_backoff` decorator (see `sync_api_call`)
def timeout_getter(args: Any, kwargs: Any, key: str = "provider_gen_config") -> float:
    provider_gen_config: dict[str, Any] = kwargs.get(key)
    n: float = provider_gen_config.get("n", MAX_API_WAIT_TIME)
    return min(MAX_WAIT_PER_GEN * n, MAX_API_WAIT_TIME)


# ==============================================================================
# LINK: Output conversion: provider-specific -> uniform format
# ==============================================================================


def _convert_chat_completion_generation_dict(choice: dict[str, Any]) -> Message | None:
    """
    Convert a single API completion to a uniform Message.
    """
    all_contents: List[ContentItem] = []
    if "message" not in choice:
        return None

    if "content" not in choice["message"] or not choice["message"]["content"]:
        return None

    if not isinstance(choice["message"]["content"], list):
        choice["message"]["content"] = [choice["message"]["content"]]  # type: ignore

    for c in choice["message"]["content"]:
        annotations = {}
        if choice["message"].get("annotations", None):
            # https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat
            annotations = {"annotations": [ann.to_dict() for ann in choice["message"]["annotations"]]}

        # NOTE: Commented out as not supported for openai chat completion yet
        # if img := is_image(c, return_image=True):
        #     all_contents.append(ContentItem(type="image", data=img, meta_data=annotations))

        if isinstance(c, str):
            all_contents.append(ContentItem(type="text", data=c, meta_data=annotations))

        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if choice["message"].get("tool_calls", None):
        # https://platform.openai.com/docs/guides/function-calling?api-mode=chat#handling-function-calls
        for tool_call in choice["message"]["tool_calls"]:
            all_contents.append(ContentItem(type="function_call", data=tool_call.to_dict()))

    if choice["message"].get("reasoning", None):
        all_contents.append(ContentItem(type="reasoning", data=choice["message"]["reasoning"].to_dict()))

    if choice["message"].get("role", None):
        role = choice["message"]["role"]
    else:
        role = "assistant"

    return Message(role=role, contents=all_contents, name="")


def _convert_chat_completion_generation(choice: Choice) -> Message | None:
    """
    Convert a single API completion to a uniform Message.
    """
    all_contents: List[ContentItem] = []
    if choice.message.content is None:
        return None

    if not isinstance(choice.message.content, list):
        choice.message.content = [choice.message.content]  # type: ignore

    for c in choice.message.content:
        annotations = {}
        if choice.message.annotations:
            # https://platform.openai.com/docs/guides/tools-web-search?api-mode=chat
            annotations = {"annotations": [ann.to_dict() for ann in choice.message.annotations]}

        # NOTE: Commented out as not supported for openai chat completion yet
        # if img := is_image(c, return_image=True):
        #     all_contents.append(ContentItem(type="image", data=img, meta_data=annotations))

        if isinstance(c, str):
            all_contents.append(ContentItem(type="text", data=c, meta_data=annotations))

        else:
            raise ValueError(f"Unsupported content type: {type(c)}")

    if choice.message.tool_calls is not None:
        # https://platform.openai.com/docs/guides/function-calling?api-mode=chat#handling-function-calls
        for tool_call in choice.message.tool_calls:
            all_contents.append(ContentItem(type="function_call", data=tool_call.to_dict()))

    if hasattr(choice.message, "role"):
        role = choice.message.role

    else:
        role = "assistant"

    return Message(role=role, contents=all_contents, name="")


def _handle_single_response_item(response_item: ResponseOutputItem) -> List[ContentItem]:
    all_content_items = []
    if response_item.type == "message":
        for c in response_item.content:
            if c.type == "output_text":
                c_dict = c.to_dict()
                meta_data = {k: v for k, v in c_dict.items() if k != "text" and k != "type"}
                all_content_items.append(ContentItem(type="text", data=c.text, meta_data=meta_data, raw_model_output=c))

    elif response_item.type == "file_search_call":
        all_content_items.append(
            ContentItem(type="file_search", data=response_item.to_dict(), raw_model_output=response_item)
        )

    elif response_item.type == "function_call":
        all_content_items.append(
            ContentItem(
                type="function_call", data=response_item.to_dict(), meta_data={}, raw_model_output=response_item
            )
        )

    elif response_item.type == "web_search_call":
        # The relevant parts for this are handled by `message` in the cases with `annotations`
        pass

    elif response_item.type == "computer_call":
        all_content_items.append(
            ContentItem(type="computer_call", data=response_item.to_dict(), raw_model_output=response_item)
        )

    elif response_item.type == "reasoning":
        all_content_items.append(
            ContentItem(type="reasoning", data=response_item.to_dict(), raw_model_output=response_item)
        )
    return all_content_items


def _convert_response_generation(response: Response) -> Message | None:
    """
    Convert a 'response' format output from OpenAI into a uniform Message.
    This function handles multiple types of outputs:
    """
    # https://platform.openai.com/docs/api-reference/responses/object#responses/object-output

    # Aggregate all content items into a single message.
    all_content_items = []
    for out_item in response.output:
        all_content_items.extend(_handle_single_response_item(out_item))

    if all_content_items:
        return Message(role="assistant", contents=all_content_items, name="")
    else:
        return None


def convert_single_generation(output: Choice | Response | dict[str, Any]) -> Message | None:
    if isinstance(output, Choice):
        return _convert_chat_completion_generation(output)
    elif isinstance(output, Response):
        return _convert_response_generation(output)
    elif isinstance(output, dict):
        return _convert_chat_completion_generation_dict(output)
    else:
        raise ValueError(f"Unsupported API output type: {type(output)}")


def convert_generations(api_response: ChatCompletion | Response | dict[str, Any]) -> List[Message]:
    """
    Convert the API response into a list of Message objects.
    """
    all_generations = []

    if isinstance(api_response, ChatCompletion) and hasattr(api_response, "choices"):
        for choice in api_response.choices:
            msg = convert_single_generation(choice)
            all_generations.append(msg) if msg else None

    elif isinstance(api_response, dict) and "choices" in api_response:
        for choice in api_response["choices"]:
            if choice is None:
                continue
            msg = convert_single_generation(choice)
            all_generations.append(msg) if msg else None

    elif isinstance(api_response, Response) and hasattr(api_response, "output"):
        # NOTE: the Response API does not accept a `n` parameter, so we only return the first generation (Mar-2025)
        msg = convert_single_generation(api_response)
        all_generations.append(msg) if msg else None

    return all_generations


# ==============================================================================
# LINK: Prompt messages conversion: uniform -> provider-specific format
# ==============================================================================


def get_provider_msgs(
    messages: List[Message], gen_config: GenerationConfig, use_cache: bool = True
) -> List[Dict[str, Any]]:
    """
    Process the input messages:
      - Use OpenAIPrompter to convert the unified List[Message] to provider-specific format.
      - Reset the prompt or trigger image upload if needed.
    """
    global cache

    provider_msgs = []
    if use_cache:
        provider_msgs = cache.messages_to_provider

    if not provider_msgs:
        provider_msgs = OpenAIPrompter.convert_prompt(messages, gen_config.mode)
        cache.messages_to_provider = provider_msgs

    # global RESET_PROMPT # Not implemented for OpenAI
    # if RESET_PROMPT:
    #     logger.info("Resetting prompt...")
    #     provider_msgs = OpenAIPrompter.reset_prompt(provider_msgs)
    #     RESET_PROMPT = False
    #     cache.messages_to_provider = provider_msgs

    return provider_msgs


# ===============================================================================
# LINK: Generation config conversion: uniform -> provider-specific format
# ===============================================================================
def regularize_provider_gen_config(provider_gen_config: dict[str, Any], gen_config: GenerationConfig) -> dict[str, Any]:
    provider_gen_config = regularize_provider_gen_config_for_model(provider_gen_config)
    if gen_config.mode == "response":
        if "n" in provider_gen_config:
            del provider_gen_config["n"]

    if "modalities" in provider_gen_config:
        provider_gen_config["modalities"] = [m.lower() for m in provider_gen_config["modalities"]]

    return provider_gen_config


def regularize_provider_gen_config_for_model(provider_gen_config: dict[str, Any]) -> dict[str, Any]:
    """
    Regularize generation arguments for different models.
    """
    model = provider_gen_config["model"]
    if "o1" in model:
        unsupported_params = [
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "modalities",
        ]
        if "max_tokens" in provider_gen_config:
            provider_gen_config["max_completion_tokens"] = provider_gen_config["max_tokens"]
            del provider_gen_config["max_tokens"]

    elif "4o" in model:
        unsupported_params = ["reasoning_effort", "reasoning"]
        provider_gen_config["modalities"] = ["text"]

    elif "computer-use" in model:
        unsupported_params = []  # TODO
        if "truncation" in provider_gen_config:
            provider_gen_config["truncation"] = "auto"

        if "reasoning" in provider_gen_config and "generate_summary" in provider_gen_config["reasoning"]:
            provider_gen_config["reasoning"]["generate_summary"] = "concise"
        return provider_gen_config
    else:
        return provider_gen_config

    for param in unsupported_params:
        if param in provider_gen_config:
            del provider_gen_config[param]

    return provider_gen_config


def _build_base_config(gen_config: GenerationConfig) -> dict[str, Any]:
    """
    Builds the portion of the config shared by both 'chat_completion'
    and 'response' modes.
    """
    base_kwargs: dict[str, Any] = {
        "model": gen_config.model,
        "temperature": gen_config.temperature,
        "top_p": gen_config.top_p,
    }

    if gen_config.tools is not None:
        base_kwargs["tools"] = gen_config.tools

    if gen_config.tool_choice is not None:
        base_kwargs["tool_choice"] = gen_config.tool_choice

    # Optional fields (only add them if they are not None)
    if gen_config.web_search_options is not None:
        base_kwargs["web_search_options"] = gen_config.web_search_options

    if gen_config.stop_sequences is not None:
        base_kwargs["stop"] = gen_config.stop_sequences

    if gen_config.seed is not None:
        base_kwargs["seed"] = gen_config.seed

    if gen_config.response_format is not None:
        base_kwargs["response_format"] = gen_config.response_format

    if gen_config.metadata is not None:
        base_kwargs["metadata"] = gen_config.metadata

    return base_kwargs


def _build_chat_completion_config(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Builds the config needed specifically for 'chat_completion' mode.
    """
    # Start with the base config
    chat_kwargs = _build_base_config(gen_config)

    # Chat-specific fields
    chat_kwargs["n"] = gen_config.num_generations
    chat_kwargs["modalities"] = [m.lower() for m in gen_config.modalities]
    chat_kwargs["reasoning_effort"] = gen_config.reasoning_effort
    chat_kwargs["max_completion_tokens"] = gen_config.max_tokens

    chat_kwargs["frequency_penalty"] = gen_config.frequency_penalty
    chat_kwargs["presence_penalty"] = gen_config.presence_penalty
    chat_kwargs["logprobs"] = gen_config.logprobs
    return chat_kwargs


def _build_response_config(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Builds the config needed specifically for 'response' mode.
    """
    # Start with the base config
    resp_kwargs = _build_base_config(gen_config)

    # Response-specific handling of reasoning params
    resp_kwargs["reasoning"] = {"generate_summary": "detailed"}
    if gen_config.reasoning_effort is not None:
        resp_kwargs["reasoning"] = {"effort": gen_config.reasoning_effort}

    # The response endpoint uses max_output_tokens instead of max_completion_tokens
    resp_kwargs["max_output_tokens"] = gen_config.max_tokens

    # Additional fields only used by response endpoints
    if gen_config.include is not None:
        resp_kwargs["include"] = gen_config.include

    if gen_config.previous_response_id is not None:
        resp_kwargs["previous_response_id"] = gen_config.previous_response_id

    if gen_config.text is not None:
        resp_kwargs["text"] = gen_config.text

    if gen_config.truncation is not None:
        resp_kwargs["truncation"] = gen_config.truncation

    return resp_kwargs


def gen_config_to_provider(gen_config: GenerationConfig) -> Dict[str, Any]:
    """
    Build OpenAI-specific generation arguments from a unified GenerationConfig,
    dispatching to the appropriate builder based on `mode`.
    """
    if not gen_config.mode:
        # By default, use chat completion mode
        gen_config.mode = DEFAULT_OPENAI_MODE

    if gen_config.mode == "chat_completion":
        return _build_chat_completion_config(gen_config)

    elif gen_config.mode == "response":
        return _build_response_config(gen_config)

    else:
        raise ValueError(f"Unsupported mode: {gen_config.mode}")


def get_provider_gen_config(
    gen_config: GenerationConfig,
) -> Dict[str, Any]:
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


# ===============================================================================
# LINK Synchronous Generation
# ===============================================================================
def generate_from_openai(
    messages: List[Message],
    gen_config: GenerationConfig,
) -> tuple[List[dict[str, Any]], List[Message]]:
    """
    Synchronous generation from OpenAI's API for both 'chat_completion' and 'response' modes.

    Both modes use a loop to repeatedly call the API until the requested number of generations are obtained.

    Returns:
        A tuple containing:
         - A list of raw API responses
         - A list of uniform Message objects generated from the responses.
    """
    global MAX_GENERATION_PER_BATCH, cache
    cache.reset()

    # Number of generations remaining
    remaining_generation_count = gen_config.num_generations

    # Build provider-specific configuration and messages.
    provider_gen_config = get_provider_gen_config(gen_config)
    _ = get_provider_msgs(messages, gen_config)

    logger.info(
        f"[{__file__}] CALLING MODEL: `{gen_config.model}` in mode `{gen_config.mode}`: generating {gen_config.num_generations} output(s)..."
    )

    while remaining_generation_count > 0:
        # Set current batch size in the provider configuration.
        provider_gen_config["n"] = min(MAX_GENERATION_PER_BATCH, remaining_generation_count)
        # Regularize the provider configuration based on model and mode.
        provider_gen_config = regularize_provider_gen_config(provider_gen_config, gen_config)

        response = sync_call(
            messages=messages,
            provider_gen_config=provider_gen_config,
            gen_config=gen_config,
        )
        model_messages = convert_generations(response)

        if model_messages:
            cache.api_responses.append(response.to_dict())
            cache.model_messages.extend(model_messages)
            remaining_generation_count -= len(model_messages)
        else:
            logger.warning("No generations returned from API call; breaking out of loop.")
            break

    return cache.api_responses, cache.model_messages


@timeit(custom_name=f"LLM:sync_{os.path.basename(__file__)}_api_call")
@retry_exp_backoff(timeout_getter=timeout_getter)  # type: ignore
def sync_call(
    messages: List[Message],
    provider_gen_config: Dict[str, Any],
    gen_config: GenerationConfig,
) -> ChatCompletion | Response:
    # Get global client
    client = get_client_manager().get_client()

    # Get provider messages. Obs.: This caches, re-upload, reset prompts if needed.
    provider_msgs = get_provider_msgs(messages, gen_config)

    if gen_config.mode == "chat_completion":
        return client.chat.completions.create(messages=provider_msgs, **provider_gen_config)  # type:ignore

    elif gen_config.mode == "response":
        return client.responses.create(input=provider_msgs, **provider_gen_config)  # type: ignore

    else:
        raise ValueError(f"Unsupported mode: {gen_config.mode}")


# ===============================================================================
# LINK Asynchronous Generation
# ===============================================================================


async def _throttled_openai_agenerate(
    aclient: AsyncOpenAI, limiter: aiolimiter.AsyncLimiter, **kwargs: Any
) -> ChatCompletion | dict[str, Any]:
    """Call OpenAI asynchronously with built-in retry logic and rate-limiting."""
    async with limiter:
        for _ in range(MAX_RETRIES):
            try:
                return await aclient.chat.completions.create(**kwargs)  # type: ignore
            except openai.RateLimitError:
                logging.warning("OpenAI API rate limit exceeded. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except asyncio.exceptions.TimeoutError:
                logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(10)
            except openai.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                break
        return {"choices": [{"message": {"content": ""}}]}


def batch_generate_from_openai(
    messages_list: list[list[Message]],
    gen_config: GenerationConfig,
    requests_per_minute: int = 100,
) -> tuple[List[dict[str, Any]], List[List[Message]]]:
    """
    Asynchronous generation from OpenAI's Chat Completion API.

    Args:
        prompt_batches: A list where each element is a list of messages to be sent to the API.
        gen_config: Generation configuration.
        requests_per_minute: Rate-limit for async requests.

    Returns:
        A tuple of:
          - List of raw JSON response objects.
          - List of generated text contents.
    """

    aclient = get_client_manager().get_aclient()
    provider_gen_config = get_provider_gen_config(gen_config)
    provider_gen_config = regularize_provider_gen_config(provider_gen_config, gen_config)

    if provider_gen_config.get("n", 1) > 1:
        provider_gen_config["n"] = 1
        logger.warning("Setting num_generations to 1 for batch async generation.")

    provider_msgs_list = [get_provider_msgs(messages, gen_config, use_cache=False) for messages in messages_list]
    # Create a rate-limiter
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)

    # Convert each prompt using OpenAIPrompter (without caching for async calls)
    tasks = [
        _throttled_openai_agenerate(
            aclient=aclient, limiter=limiter, **{"messages": provider_msgs, **provider_gen_config}
        )
        for provider_msgs in provider_msgs_list
    ]

    async def _async_generate() -> tuple[list[dict[str, Any]], list[List[Message]]]:
        results: list[ChatCompletion | dict[str, Any]] = await asyncio.gather(*tasks)
        all_api_responses = []
        all_model_messages = []
        for response in results:
            if isinstance(response, dict):
                all_api_responses.append(response)
            else:
                all_api_responses.append(response.to_dict())
            all_model_messages.append(convert_generations(response))
        return all_api_responses, all_model_messages

    # Gather all responses asynchronously.
    all_api_responses, all_model_messages = asyncio.run(_async_generate())

    return all_api_responses, all_model_messages
