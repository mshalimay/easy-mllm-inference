import datetime
import gc
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Sequence

import numpy as np
import pandas as pd  # type: ignore
import yaml  # type: ignore
from PIL import Image

from llms.providers.anthropic.anthropic_utils import generate_from_anthropic
from utils.file_utils import flatten_dict
from utils.timing_utils import timeit
from utils.types import ImageInput

from .constants import MODEL_REPO_PATH
from .generation_config import GenerationConfig, get_fields, make_generation_config
from .prompt_utils import (
    ValidInputs,
    build_message,
    conversation_to_html,
    conversation_to_txt,
    flatten_generations,
    get_messages,
)
from .prompt_utils import visualize_prompt as visualize_prompt_utils
from .providers.google.google_utils import generate_from_google_chat_completion
from .providers.hugging_face.hf_utils import batch_generate_from_huggingface, generate_from_huggingface
from .providers.openai.openai_utils import batch_generate_from_openai, generate_from_openai
from .setup_utils import infer_provider
from .types import Message

# ==============================================================================
# LINK High level functions
# ==============================================================================


# Main user facing function for LLM calling
def call_llm(
    gen_kwargs: dict[str, Any],
    prompt: ValidInputs | List[Message],
    meta_data: dict[str, Any] = {},
    conversation_dir: str = "",
    usage_dir: str = "",
    call_id: str = "",
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
) -> tuple[list[dict[str, Any]], List[Message]]:
    """
    Call an LLM.

    Args:
        gen_kwargs (dict[str, Any]): The generation arguments.
        prompt (List[ImageInput | Dict[str, Any] | Message | List[Message]]): The prompt to send to the LLM.
        meta_data (dict[str, Any], optional): Optional arguments for additional behavior.
        conversation_dir (str, optional): Directory path to store conversation logs.
            e.g. "results/experiment_1"
        usage_dir (str, optional): Directory path to store usage logs.
            e.g. "results/experiment_1"
        call_id (str, optional): Unique identifier for the LLM call; used to name the files stored in `conversation_dir` and `usage_dir`.
            e.g. "task_0" => `results/experiment_1/task_0.html`, `results/experiment_1/task_0.txt`, `results/experiment_1/task_0.csv`.
        dump_html (bool, optional): If true, creates an HTML file to store the conversation.
        dump_txt (bool, optional): If true, creates a TXT file to store the conversation.
        verbose (bool, optional): If true, prints verbose output.

    Returns:
        tuple[list[Any], list[Message]]: A tuple of the API response and the model generations.
    """
    if call_id:
        call_id = str(call_id)
    elif conversation_dir or usage_dir:
        call_id = get_call_id(gen_kwargs)

    if not isinstance(prompt, list):
        prompt = [prompt]  # type: ignore

    prompt_uniform_format = get_messages(inputs=prompt)

    # Regularize/add arguments for the LLM call
    reg_gen_kwargs = _regularize_gen_kwargs(gen_kwargs)

    # If manual input signal given, get input from user
    if meta_data.get("manual_input", False):
        model_generation = get_manual_input(
            prompt_uniform_format, meta_data, conversation_dir, reg_gen_kwargs, verbose, dump_html, dump_txt
        )
        if not model_generation[0].contents[0].data == "llm":
            return [], model_generation

    # Get generation config
    gen_config = make_generation_config(reg_gen_kwargs)

    # Generate from LLM
    api_responses, model_generations = _generate_from_llm(
        messages=prompt_uniform_format, gen_config=gen_config, provider=reg_gen_kwargs["provider"]
    )
    if len(api_responses) == 0 or len(model_generations) == 0:
        print(f"No API responses or model generations returned from LLM call. Returning empty lists.")
        return [], []

    if conversation_dir:
        dump_llm_output(
            prompt_uniform_format,
            model_generations,
            conversation_dir,
            call_id,
            reg_gen_kwargs,
            verbose,
            dump_html,
            dump_txt,
        )

    if usage_dir:
        dump_usage(reg_gen_kwargs["provider"], api_responses, usage_dir, call_id, reg_gen_kwargs)

    return api_responses, model_generations


def batch_call_llm(
    gen_kwargs: dict[str, Any],
    prompts: list[ValidInputs] | list[list[Message]],
    conversation_dirs: list[str] | None = None,
    usage_dirs: list[str] | None = None,
    call_ids: list[str] | None = None,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    max_batch_size: int = 10,
    num_workers: int = 2,
    return_outputs: bool = True,
    multiprocess_mode: bool = False,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    """
    Batch call an LLM.

    Args:
        gen_kwargs (dict[str, Any]): The generation arguments.
        prompts (list[ValidInputs | List[Message]]): The prompts to send to the LLM.
        conversation_dirs (list[str] | None, optional): The directories to save the conversation logs.
        usage_dirs (list[str] | None, optional): The directories to save the usage logs.
        call_ids (list[str] | None, optional): The unique identifiers for each call.
        max_batch_size (int, optional): Maximum batch size per worker at each iteration.
        num_workers (int, optional): Number of workers for parallel processing.
        return_outputs (bool, optional): If true, return the API responses and model generations. Use false if to save memory if only dumping outputs.
        verbose (bool, optional): If true, print verbose output.
        dump_html (bool, optional): If true, dump the conversation to an HTML file.
        dump_txt (bool, optional): If true, dump the conversation to a TXT file.

    Returns:
        tuple[list[Any], list[Message]]: A tuple of the API responses and the model generations.
    """

    # Argument validation
    if call_ids:
        if len(prompts) != len(call_ids):
            raise ValueError("prompts and call_ids must be the same length")
        if conversation_dirs or usage_dirs:
            if len(call_ids) != len(conversation_dirs) or len(call_ids) != len(usage_dirs):
                raise ValueError("prompts, conversation_dirs, and usage_dirs must be the same length")

        call_ids = [str(call_id) for call_id in call_ids]

    elif conversation_dirs or usage_dirs:
        if len(prompts) != len(conversation_dirs) or len(prompts) != len(usage_dirs):
            raise ValueError("prompts, conversation_dirs, and usage_dirs must be the same length")
        call_ids = [get_call_id(gen_kwargs) for _ in prompts]

    if n := gen_kwargs.get("num_generations", None) is not None:
        if int(n) > 1:
            raise NotImplementedError("Batch generation with num_generations > 1 is not implemented yet.")

    # Regularize arguments for the LLM call and get generation config
    reg_gen_kwargs = _regularize_gen_kwargs(gen_kwargs)
    gen_config = make_generation_config(reg_gen_kwargs)

    if not multiprocess_mode:
        reg_prompts = _batch_regularize_prompts(prompts)
    else:
        reg_prompts = prompts

    if multiprocess_mode:
        all_api_responses, all_model_generations = _batch_generate_multiprocess(
            prompts=reg_prompts,
            conversation_dirs=conversation_dirs,
            usage_dirs=usage_dirs,
            call_ids=call_ids,
            gen_config=gen_config,
            num_workers=num_workers,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            return_outputs=return_outputs,
            max_batch_size=max_batch_size,
        )
    else:
        all_api_responses, all_model_generations = _batch_generate_from_provider(
            prompts=reg_prompts,  # type: ignore
            conversation_dirs=conversation_dirs,
            usage_dirs=usage_dirs,
            call_ids=call_ids,
            gen_config=gen_config,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            max_batch_size=max_batch_size,
        )
    return all_api_responses, all_model_generations


def get_gen_config_fields() -> list[str]:
    gen_fields = get_fields()
    return gen_fields


# TODO: deprecate this function
def get_avail_models(model_name: str) -> list[str]:
    # For each model in the repo, print base name that matches the model_name
    model_matches: list[str] = []
    with open(MODEL_REPO_PATH, "r") as file:
        # For each model in the repo, print base name that matches the model_name
        for model in yaml.safe_load(file)["models"]:  # type:ignore
            if model_name in model:
                model_matches.append(model)
    return model_matches


def get_google_models() -> list[dict[str, Any]]:
    from .providers.google.google_client_manager import get_google_models as _get_google_models

    return _get_google_models()


def get_hf_model_object(model_path: str = "") -> Any:
    from .providers.hugging_face.hugging_face_client_manager import get_client_managers

    client_managers = get_client_managers()

    if client_managers:
        if model_path and model_path in client_managers:
            return client_managers[model_path].get_model(gen_config={}, engine="automodel")
        else:
            key = list(client_managers.keys())[0]
            return client_managers[key].get_model(gen_config={}, engine="automodel")
    else:
        print("No models available")
        return None


# ==============================================================================
# LINK LLM call - low level functions
# ==============================================================================


def _generate_from_llm(
    messages: List[Message],
    gen_config: GenerationConfig,
    provider: str,
    add_args: dict[str, Any] = {},
) -> tuple[List[Any], List[Message]]:
    if provider == "openai":
        api_responses, model_generations = generate_from_openai(
            messages=messages,
            gen_config=gen_config,
        )
    elif provider == "google":
        api_responses, model_generations = generate_from_google_chat_completion(
            messages=messages,
            gen_config=gen_config,
        )
    elif provider == "huggingface":
        api_responses, model_generations = generate_from_huggingface(
            messages=messages,
            gen_config=gen_config,
        )
    elif provider == "anthropic":
        api_responses, model_generations = generate_from_anthropic(
            messages=messages,
            gen_config=gen_config,
        )
    else:
        raise NotImplementedError(f"Provider {provider} not implemented")
    return api_responses, model_generations


def get_manual_input(
    prompt: List[Message],
    meta_data: dict[str, Any],
    conversation_dir: str,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
) -> List[Message]:
    # Generate a call id to save logs
    call_id = f"manual_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if conversation_dir:
        pass
        # Save a firtst version of the conversation to help in the manual input
        # dump_llm_output(prompt, "", conversation_dir, call_id, gen_kwargs, verbose, dump_html, dump_txt)

    # Save an image obs if available, to help in the manual input
    if "trajectory" in meta_data:
        # Save the last observation as an image
        img = Image.fromarray(meta_data["trajectory"].states[-1]["observation"]["image"])
        img.save("observation.png")

    # Get the manual input from the user
    print("Enter text followed by CTRL+D:", end="", flush=True)
    utterance = sys.stdin.read()
    model_generations = build_message(contents=[utterance], role="assistant", name="manual_input")

    # If the user inputs "llm", then the next utterance will come from an LLM call
    # If not "llm", then dump conversation and return utterance
    if not utterance == "llm" and conversation_dir:
        dump_llm_output(
            prompt, [model_generations], conversation_dir, call_id, gen_kwargs, verbose, dump_html, dump_txt
        )
    return [model_generations]


# ==============================================================================
# LINK Batch generation
# ==============================================================================


def _batch_regularize_prompts(
    prompts: list[ValidInputs] | list[list[Message]],
) -> list[list[Message]]:
    """
    Regularize the prompts.
    """
    regularized_prompts = []
    for prompt in prompts:
        if not isinstance(prompt, list):
            prompt = [prompt]
        regularized_prompts.append(get_messages(prompt))  # type: ignore
    return regularized_prompts


def _process_batch(
    prompts: list[ValidInputs] | list[List[Message]],
    gen_config: GenerationConfig,
    conversation_dirs: list[str] | None = None,
    usage_dirs: list[str] | None = None,
    call_ids: list[str] | None = None,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    return_outputs: bool = True,
) -> tuple[list[Any], list[Message]]:
    import builtins
    import functools

    # fmt: off
    builtins.print = functools.partial(builtins.print, flush=True)
    batch_api_responses = []
    batch_model_generations = []

    print(f"Batch LLM call [{os.getpid()}]: Processing batch of size {len(prompts)}")

    # For each conversation in the batch, generate from LLM and dump outputs.
    for i, prompt in enumerate(prompts):
        if not isinstance(prompt, list):
            prompt = [prompt]
        prompt_uniform_format = get_messages(inputs=prompt)  # type: ignore

        api_response, model_generations = _generate_from_llm(
            messages=prompt_uniform_format,
            gen_config=gen_config,
            provider=gen_config.provider,
        )
        
        if conversation_dirs:
            dump_llm_output(prompt_uniform_format, model_generations, conversation_dirs[i], call_ids[i], gen_config.to_dict(), verbose, dump_html, dump_txt)  # type: ignore
        if usage_dirs:
            dump_usage(gen_config.provider, api_response, usage_dirs[i], call_ids[i], gen_config.to_dict(), verbose)  # type: ignore

        # If no need to return outputs, skip accummulation to save memory.
        if return_outputs:
            batch_api_responses.append(api_response)
            batch_model_generations.append(model_generations)
    # fmt: on
    return batch_api_responses, batch_model_generations


def compute_batches(
    prompts: list[ValidInputs] | list[List[Message]],
    max_batch_size: int = 10,
    conversation_dirs: list[str] | None = None,
    usage_dirs: list[str] | None = None,
    call_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Returns a list of dictionaries, each containing the data for a batch of prompts to send to the LLM.
    Example: output:
    output[0] = {
        "prompts": [prompts[0], prompts[1]],
        "conversation_dirs": [conversation_dirs[0], conversation_dirs[1]],
        "usage_dirs": [usage_dirs[0], usage_dirs[1]],
        "call_ids": [call_ids[0], call_ids[1]],
    }
    """
    # Calculate the number of batches; each batch will have at most max_batch_size items.
    total_tasks = len(prompts)
    num_batches = (total_tasks + max_batch_size - 1) // max_batch_size

    # Create batches: each sublist is at most max_batch_size long.
    batch_idxs = np.array_split(range(total_tasks), num_batches)

    all_job_data = []
    for batch_idx in batch_idxs:
        job_data = {}
        job_data["prompts"] = [prompts[i] for i in batch_idx.tolist()]

        job_data["conversation_dirs"] = None
        if conversation_dirs:
            job_data["conversation_dirs"] = [conversation_dirs[i] for i in batch_idx.tolist()]

        job_data["usage_dirs"] = None
        if usage_dirs:
            job_data["usage_dirs"] = [usage_dirs[i] for i in batch_idx.tolist()]

        job_data["call_ids"] = None
        if call_ids:
            job_data["call_ids"] = [call_ids[i] for i in batch_idx.tolist()]
        all_job_data.append(job_data)

    return all_job_data


def _batch_generate_multiprocess(
    prompts: list[ValidInputs] | list[list[Message]],
    gen_config: GenerationConfig,
    conversation_dirs: list[str] | None = None,
    usage_dirs: list[str] | None = None,
    call_ids: list[str] | None = None,
    num_workers: int = 2,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    return_outputs: bool = True,
    max_batch_size: int = 10,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    """
    Process a batch call for Google LLM in parallel.

    This function splits the messages_list into batches of size at most max_batch_size,
    then uses Joblib to process those batches using num_workers in parallel.

    Args:
        messages_list (list[List[Message]]): A list where each element is a conversation (a list of Message).
        gen_config (GenerationConfig): The generation configuration.
        max_batch_size (int, optional): Maximum number of messages to process per iteration.
        num_workers (int, optional): Number of workers for parallel processing.
        return_outputs (bool, optional): If true, return the API responses and model generations. Use false if to save memory if only dumping outputs.
    Returns:
        tuple[list[list[Any]], list[list[Message]]]: Tuple of all API responses and all model generations.
    """
    from joblib import Parallel, delayed

    all_jobs_data = compute_batches(
        prompts=prompts,
        conversation_dirs=conversation_dirs,
        usage_dirs=usage_dirs,
        call_ids=call_ids,
        max_batch_size=max_batch_size,
    )

    if len(all_jobs_data) < num_workers:
        num_workers = len(all_jobs_data)

    # Why joblib: needs isolated environments for some functionalities in providers code (e.g.: isolated clients, worker pools for timeout)
    results = Parallel(n_jobs=num_workers)(
        delayed(_process_batch)(
            **job_data,
            gen_config=gen_config,
            verbose=verbose,
            dump_html=dump_html,
            dump_txt=dump_txt,
            return_outputs=return_outputs,
        )
        for job_data in all_jobs_data
    )
    all_api_responses, all_model_generations = [], []
    # Flatten the results from all batches.
    if results is not None and return_outputs:
        for responses, generations in results:  # type: ignore
            all_api_responses.extend(responses)
            all_model_generations.extend(generations)
    else:
        # Try to release memory
        del results
        gc.collect()
    return all_api_responses, all_model_generations


def _batch_generate_from_provider(
    prompts: list[list[Message]],
    gen_config: GenerationConfig,
    conversation_dirs: list[str] | None = None,
    usage_dirs: list[str] | None = None,
    call_ids: list[str] | None = None,
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
    max_batch_size: int = 10,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    """
    Batch generate from provider.
    """
    # fmt: off
    all_api_responses, all_model_generations = [], []
    gen_config_dict = gen_config.to_dict()


    if gen_config.provider == "openai":
        
        all_api_responses, all_model_generations = batch_generate_from_openai(
            messages_list=prompts,
            gen_config=gen_config,
        )

        if conversation_dirs and call_ids:
            for prompt, conversation_dir, api_response, model_generations, call_id in zip(prompts, conversation_dirs, all_api_responses, all_model_generations, call_ids):
                dump_llm_output(prompt, model_generations, conversation_dir, call_id, gen_config_dict, verbose, dump_html, dump_txt)

        if usage_dirs and call_ids:
            for prompt, usage_dir, api_response, model_generations, call_id in zip(prompts, usage_dirs, all_api_responses, all_model_generations, call_ids):
                dump_usage(gen_config.provider, [api_response], usage_dir, call_id, gen_config_dict, verbose)

        return all_api_responses, all_model_generations


    elif gen_config.provider == "huggingface" and gen_config.engine == "openai":
        return batch_generate_from_huggingface(
            messages_list=prompts,
            gen_config=gen_config,
        )


    else:
        request_data_in_batches = compute_batches(
            prompts=prompts,
            conversation_dirs=conversation_dirs,
            usage_dirs=usage_dirs,
            call_ids=call_ids,
            max_batch_size=max_batch_size,
        )
        for batch_data in request_data_in_batches:
            batch_api_responses, batch_model_generations = _route_batch_generation(
                messages=batch_data["prompts"],
                gen_config=gen_config,
                provider=gen_config.provider,
            )
            all_api_responses.extend(batch_api_responses)
            all_model_generations.extend(batch_model_generations)

        # TODO: parallelize this or do while provider is generating
        if "conversation_dirs" in batch_data:
            for prompt, model_generations, conversation_dir, call_id in zip(batch_data["prompts"], batch_model_generations, batch_data["conversation_dirs"], batch_data["call_ids"]):
                dump_llm_output(prompt, model_generations, conversation_dir, call_id, gen_config_dict, verbose, dump_html, dump_txt)
        if "usage_dirs" in batch_data:
            for api_response, usage_dir, call_id in zip(batch_api_responses, batch_data["usage_dirs"], batch_data["call_ids"]):
                dump_usage(gen_config.provider, [api_response], usage_dir, call_id, gen_config_dict, verbose)
        # fmt: on
        return all_api_responses, all_model_generations


def _route_batch_generation(
    messages: list[list[Message]],
    gen_config: GenerationConfig,
    provider: str,
) -> tuple[list[dict[str, Any]], list[list[Message]]]:
    if provider == "huggingface":
        return batch_generate_from_huggingface(messages, gen_config)
    elif provider == "openai":
        raise NotImplementedError("OpenAI batch generation not implemented")
    elif provider == "google":
        raise NotImplementedError("Google batch generation not implemented")
    else:
        raise NotImplementedError(f"Provider {provider} not implemented")


# ==============================================================================
# LINK Get data per provider or model
# ==============================================================================


def _regularize_gen_kwargs(gen_kwargs: dict[str, Any], inplace: bool = False) -> dict[str, Any]:
    if not inplace:
        gen_kwargs = gen_kwargs.copy()

    if "model" not in gen_kwargs:
        raise ValueError("Please specify a model in gen_kwargs. Run get_avail_models() to see available models.")

    model = gen_kwargs["model"]

    with open(MODEL_REPO_PATH, "r") as file:
        model_config = yaml.safe_load(file)  # type: ignore
        if model not in model_config["models"]:
            model_config = {}
        else:
            model_config = model_config["models"][model]

    # If provider not specified, infer from model
    if "provider" not in gen_kwargs:
        provider = model_config.get("provider", None) or infer_provider(gen_kwargs["model"])
        if provider is None:
            raise ValueError(
                "Unable to infer provider from model name. Please specify a provider in gen_kwargs or add a provider to the model repository file."
            )
        gen_kwargs["provider"] = provider

    # If `mode` not specified, optionally infer from model (`mode` not needed for all providers)
    mode = gen_kwargs.get("mode", None) or model_config.get("mode", None)
    gen_kwargs["mode"] = mode

    model_path = gen_kwargs.get("model_path", None) or model_config.get("model_path", None) or model
    gen_kwargs["model_path"] = model_path

    # Reg parameters dependent on  `engine`
    if engine := gen_kwargs.get("engine", None):
        if re.search(r"server", engine, flags=re.IGNORECASE):
            endpoint = gen_kwargs.get("endpoint", None)
            if not endpoint:
                raise ValueError("Please specify an endpoint in gen_kwargs when hosting a model on a server.")
            gen_kwargs["endpoint"] = endpoint
            # Extract host and port from endpoint
            match = re.search(r"^(?:https?://)?([^:/]+):(\d+)", endpoint)
            if match and len(match.groups()) == 2:
                host = match.group(1)
                port = match.group(2)
                gen_kwargs["endpoint"] = f"http://{host}:{port}/generate"
            else:
                raise ValueError("Invalid endpoint format. Please use the format 'host:port'.")
    return gen_kwargs


def _get_usage(provider: str, api_response: dict[str, Any]) -> dict[str, Any]:
    try:
        if provider == "openai":
            usage = flatten_dict(api_response["usage"])

            reasoning_tokens = usage["completion_tokens_details:reasoning_tokens"]
            completion_tokens = usage["completion_tokens"] - reasoning_tokens
            usage_normalized = {
                "_input_tokens": usage["prompt_tokens"],
                "_completion_tokens": completion_tokens,
                "_reasoning_tokens": reasoning_tokens,
            }
            usage.update(usage_normalized)

        elif provider == "google":
            usage = flatten_dict(api_response["usage_metadata"])
            usage_normalized = {
                "_input_tokens": usage["prompt_token_count"],
                "_completion_tokens": usage["candidates_token_count"],
            }
            usage.update(usage_normalized)
        elif provider == "huggingface":
            if "usage" in api_response:
                usage = flatten_dict(api_response["usage"])
            elif "usage:input_tokens" in api_response:
                usage = {k: v for k, v in api_response.items() if "usage" in k}
            else:
                usage = {}

        elif provider == "anthropic":
            usage = api_response["usage"]
            usage_normalized = {
                "_input_tokens": usage["input_tokens"],
                "_completion_tokens": usage["output_tokens"],
                # "reasoning_tokens": usage["total_tokens"],
            }
            usage.update(usage_normalized)
        else:
            raise NotImplementedError(f"Provider {provider} not implemented")

    except Exception as e:
        print(f"Error getting usage for provider {provider}: {e}")
        return {}

    return usage


# ==============================================================================
# LINK Dumping output helpers
# ==============================================================================


def get_call_id(gen_kwargs: dict[str, Any]) -> str:
    model = gen_kwargs["model"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model}_{timestamp}"


def dump_llm_output(
    messages: list[Message],
    model_generations: list[Message],
    conversation_dir: str | Path,
    call_id: str,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
    dump_html: bool = True,
    dump_txt: bool = True,
) -> None:
    if not isinstance(model_generations, list):
        model_generations = [model_generations]

    if dump_html:
        out_path = Path(conversation_dir) / f"{call_id}.html"
        dump_to_html(messages, model_generations, out_path, gen_kwargs, verbose=verbose)
    if dump_txt:
        out_path = Path(conversation_dir) / f"{call_id}.txt"
        dump_to_txt(messages, model_generations, out_path, gen_kwargs, verbose=verbose)


@timeit(custom_name="CALL_LLM:DUMP_TO_TXT")
def dump_to_txt(
    messages: list[Message],
    model_generations: List[Message],
    output_path: str | Path,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
) -> None:
    conversation_to_txt(
        prompt_messages=messages,
        model_messages=model_generations,
        output_path=output_path,
        verbose=verbose,
        gen_kwargs=gen_kwargs,
    )


@timeit(custom_name="CALL_LLM:DUMP_TO_HTML")
def dump_to_html(
    messages: list[Message],
    model_generations: List[Message],
    output_path: str | Path,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
) -> None:
    flat_generations = flatten_generations(model_generations)
    conversation_to_html(
        messages=messages + [flat_generations], output_path=output_path, verbose=verbose, gen_kwargs=gen_kwargs
    )


@timeit(custom_name="CALL_LLM:DUMP_USAGE")
def dump_usage(
    provider: str,
    api_responses: list[Any],
    usage_dir: str | Path,
    call_id: str,
    gen_kwargs: dict[str, Any],
    verbose: bool = False,
) -> None:
    os.makedirs(usage_dir, exist_ok=True)
    call_id = re.sub(r"/", "_", call_id)
    csv_path = Path(usage_dir) / f"{call_id}.csv"

    # For each API response, get its usage information
    usage_list = [_get_usage(provider, response) for response in api_responses]

    # Convert the list of usage dicts to DataFrame
    df_new = pd.DataFrame(usage_list)
    df_new["gen_config"] = None
    df_new.loc[0, "gen_config"] = str(gen_kwargs)
    # Column with number of generations
    df_new["num_generations"] = len(api_responses)

    if csv_path.exists():
        # Read existing CSV
        df_existing = pd.read_csv(csv_path)

        # Combine existing columns with new columns
        all_columns = list(set(df_existing.columns) | set(df_new.columns))

        # Update existing DataFrame with new columns
        for col in all_columns:
            if col not in df_existing.columns:
                df_existing[col] = None
            if col not in df_new.columns:
                df_new[col] = None

        # Exclude empty or all-NA columns for compatiblity with newer versions of pandas
        df_existing = df_existing.dropna(axis=1, how="all")
        df_new = df_new.dropna(axis=1, how="all")

        # Append new data
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_new = df_new.dropna(axis=1, how="all")
        df_updated = df_new

    # Save to CSV
    df_updated.to_csv(csv_path, index=False)
    if verbose:
        print(f"API usage saved to {csv_path}")


# ==============================================================================
# LINK Prompt visualization
# ==============================================================================


def visualize_prompt(messages: ValidInputs | List[Message], output_path: str | Path = "") -> None:
    prompt_uniform_format = get_messages(inputs=messages)
    visualize_prompt_utils(prompt_uniform_format, output_path)
