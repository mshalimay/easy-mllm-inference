import argparse
import json
import os
import random
import select
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import requests
import torch
import yaml

from constants.constants import LM_CONFIGS
from utils.file_utils import get_attribute_from_dict


# ===============================================================================
# TGI deployment aux functions #TODO move or delete this
# ===============================================================================
def deploy_tgi(model_path, quantize, num_shard):
    print(f"\n\nDeploying TGI locally for model {model_path}\n")
    process = subprocess.Popen(
        [
            "text-generation-launcher",
            "--model-id",
            model_path,
            "--quantize",
            quantize,
            "--num-shard",
            str(num_shard),
            "--port8080--master-port8080--master-addrlocalhost",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print("TGI has been started and is running in the background.")
    print(f"Process ID: {process.pid}")
    return process


def wait_tgi(tgi_model_endpoint, max_mins=10, timeout_secs=60, process=None, try_again_sec=30):
    from text_generation import Client  # Assuming this is correctly imported

    client = Client(tgi_model_endpoint, timeout=timeout_secs)
    running = False
    start_time = time.time()

    # Convert file objects to file descriptors
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    download_flag = False

    while not running and (time.time() - start_time) / 60 < max_mins:
        try:
            print(client.generate(prompt="Hello world").generated_text)
            running = True
        except:
            print(f"TGI didn't responde after {timeout_secs}. Trying again in {try_again_sec} seconds")
            time.sleep(try_again_sec)

        # Print tgi process output without blocking
        rlist, _, _ = select.select([stdout_fd, stderr_fd], [], [], 0)
        for fd in rlist:
            if fd == stdout_fd:
                output = os.read(stdout_fd, 1024)  # Read up to 1024 bytes
                if output:
                    out_str = output.decode("utf-8", errors="ignore").strip()
                    print(out_str)

                    if "Starting download process" in out_str:
                        download_flag = True
                    if "Skipping download." in out_str:
                        download_flag = False
                else:
                    if download_flag:
                        print("TGI is downloading the model. Sleeping for 5 minutes")
                        max_mins += 5
                        time.sleep(60 * 5)

            elif fd == stderr_fd:
                error_output = os.read(stderr_fd, 1024)  # Read up to 1024 bytes
                if error_output:
                    print(error_output.decode("utf-8", errors="ignore").strip(), file=sys.stderr)

    if not running:
        # raise TimeoutError(f"TGI Model {tgi_model_endpoint} not running after {max_mins} minutes")
        print(f"TGI Model {tgi_model_endpoint} not answring after {max_mins} minutes. Exiting...")
        terminate_process(process)
        sys.exit(0)
    else:
        print(f"\n\nTGI Model {tgi_model_endpoint} running\n\n")


def terminate_process(process):
    if process and process.poll() is None:
        print(f"Terminating TGI process...{process.pid}")
        os.kill(process.pid, signal.SIGTERM)
        process.wait()
        print("Process terminated")


def signal_handler(process, sig, frame):
    print(f"SIGINT received, terminating the process...{process.pid}")
    if process and process.poll() is None:  # Check if the process is still running
        os.kill(process.pid, signal.SIGTERM)
        process.wait()
        print("Process terminated gracefully.")
    sys.exit(0)


# ===============================================================================
# Agent and Model config helpers
# ===============================================================================


def load_agent_config(agent_config_file: str) -> dict:
    with open(agent_config_file, "r") as file:
        agents_configs = yaml.safe_load(file)
    return agents_configs


def get_lm_config(model, gen_config_alias, lm_config_file: str = LM_CONFIGS) -> dict:
    with open(lm_config_file, "r") as file:
        all_lm_configs = yaml.safe_load(file)
    return all_lm_configs[model][gen_config_alias]


def resolve_inheritance(agents_configs: dict):
    # For each agent
    for agent_str, agent_config in agents_configs.items():
        # For ach attribute
        for key, subconfig in agent_config.items():
            if not isinstance(subconfig, dict):
                continue

            if "_inherit_from" in subconfig:
                inherit_from = subconfig["_inherit_from"]
                agents_configs[agent_str][key] = agents_configs[inherit_from][key].copy()
    return agents_configs


def get_agent_attribute(agent_config_path: str, attribute: str):
    agents_configs = load_agent_config(agent_config_path)
    return get_attribute_from_dict(attribute, agents_configs)


def get_agent_config(agent_config_path: str, active_modules_str: str = ""):
    # Load general agent configurations
    agents_configs = load_agent_config(agent_config_path)

    active_modules = []
    if active_modules_str:
        active_modules = active_modules_str.lower().split(":")

    # Set model and LM Config for each agent
    for module, config in agents_configs.items():
        if active_modules and not any(m in module.lower() for m in active_modules):
            continue

        # If not LLM-based agent, skip
        if "lm_config" not in config:
            continue

        # If LM configs inherited from other modules, inherit
        if "_inherit_from" in config["lm_config"]:
            inherit_from = config["lm_config"]["_inherit_from"]
            config["lm_config"] = agents_configs[inherit_from]["lm_config"]

        # Fill LM config with actual values
        config["lm_config"].update(
            get_lm_config(
                model=config["lm_config"]["model"],
                gen_config_alias=config["lm_config"]["gen_config_alias"],
            )
        )
    return agents_configs


def check_env_status(
    urls_to_check: list[str] | str,
    check_only: list[str] = [],
    max_attempts: int = 30,
    wait_time: int = 5,
) -> bool:
    """
    Check whether the specified environments are up by parsing the output of the check_websites.sh script.

    Parameters:
      domains_to_check: List of site names to check. (For example: ["shopping", "reddit", "homepagewa", ...])
                        The check is skipped for "homepagewa" and "homepagevwa" since they are handled differently.
      max_attempts: Maximum number of attempts (default: 30).
      wait_time: Time (in seconds) to wait between attempts (default: 5).

    Returns:
      True if all the specified environments (except "homepagewa" and "homepagevwa") are up;
      False otherwise.
    """

    if isinstance(urls_to_check, str) and urls_to_check.endswith(".txt"):
        urls = []
        with open(urls_to_check, "r") as f:
            for line in f:
                # Strip out any extra quotes and whitespace.
                cleaned_line = line.strip().strip("'").strip('"')
                if not cleaned_line:
                    continue
                parts = cleaned_line.split()
                if len(parts) < 2:
                    continue
                domain_name = parts[0]
                url = parts[1]
                if check_only and not any(domain_name.lower() in d.lower() for d in check_only):
                    continue
                urls.append(url)

    elif isinstance(urls_to_check, str):
        urls = [urls_to_check]

    elif isinstance(urls_to_check, list):
        urls = urls_to_check

    attempt = 0
    while attempt < max_attempts:
        # Send a request to each URL and check if it is up
        all_sites_up = True
        for url in urls:
            if not is_website_up(url):
                all_sites_up = False
                break

        if all_sites_up:
            return True

        print(f"Waiting for {wait_time} seconds before next attempt...")
        time.sleep(wait_time)
        attempt += 1

    print(f"Warning: Some services failed to start after {max_attempts} attempts")
    print("\nFinal websites status:")
    final_result = subprocess.run(["./scripts/environments/check_websites.sh"], capture_output=True, text=True)
    print(final_result.stdout)
    return False


def is_website_up(url: str) -> bool:
    try:
        response = requests.get(url)
        return True if response.status_code else False
    except Exception as e:
        return False


def dump_config(args: argparse.Namespace, logger) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


# ===============================================================================
# Helpers for logging results
# ===============================================================================
def log_error(e: Exception, config_file: str, result_dir: str) -> None:
    with open(Path(result_dir) / "error.txt", "a") as f:
        f.write(f"[Config file]: {config_file}\n")
        f.write(f"[Unhandled Error] {repr(e)}\n")
        f.write(traceback.format_exc())  # write stack trace to file


# ===============================================================================
# Other helpers
# ===============================================================================


def set_seed(seed: int | None = None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    check_only = ["shopping", "reddit"]
    check_env_status("./config/webarena_urls/urls.txt", check_only=check_only)
