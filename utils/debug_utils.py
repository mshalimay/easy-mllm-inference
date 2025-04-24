import os
import subprocess
from typing import Any


def set_api_keys(bash_script: str = "scripts/set_api_keys.sh") -> None:
    command = f"source {bash_script} >/dev/null 2>&1 && env"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash")
    stdout, _ = process.communicate()

    # Parse the environment variables from the output
    env_vars = {}
    for line in stdout.decode().splitlines():
        key, _, value = line.partition("=")
        # Filter out any invalid or empty keys/values
        if key and value:
            env_vars[key] = value.strip()  # Stripping any leading/trailing whitespace

    # Update the current environment with the captured environment variables
    try:
        os.environ.update(env_vars)
    except OSError as e:
        print(f"Error updating environment variables: {e}")
        # Optionally, log the specific variable causing the issue
        for key, value in env_vars.items():
            try:
                os.putenv(key, value)
            except OSError as err:
                print(f"Failed to set {key}: {err}")


def set_env_variables(
    bash_script: str = "scripts/environments/set_env_variables.sh",
    arg1: str = "local_vwebarena",
    arg2: str = "localhost",
) -> dict[str, str]:
    command = f"source {bash_script} {arg1} {arg2} >/dev/null 2>&1 && env"
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash")
    stdout, _ = process.communicate()

    # Parse the environment variables from the output
    env_vars = {}
    for line in stdout.decode().splitlines():
        key, _, value = line.partition("=")
        # Filter out any invalid or empty keys/values
        if key and value and "\0" not in value:  # Check for null bytes
            env_vars[key] = value.strip()  # Strip whitespace

    # Update the environment with better error handling
    for key, value in env_vars.items():
        try:
            os.environ[key] = value
        except OSError as e:
            print(f"Failed to set environment variable {key}: {e}")
            print(f"Value: {repr(value)}")

    return env_vars
