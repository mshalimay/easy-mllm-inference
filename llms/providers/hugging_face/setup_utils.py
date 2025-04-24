"""Helpers for setting up Hugging Face models."""

import re
from typing import Any, Dict, Union

import requests
import torch
from transformers import AutoConfig

from utils.signal_utils import signal_manager


def is_flash_attn_available() -> bool:
    try:
        import flash_attn  # type: ignore

        return True
    except ImportError:
        return False


def is_bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # type: ignore

        return True
    except ImportError:
        return False


def get_attr_from_hf(model_id: str, attr: str) -> Any:
    # Try to fetch from auto config
    autoconfig = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    attr_val = getattr(autoconfig, "quantization_config", None)
    if attr_val is not None:
        return attr_val

    # Try to fetch from config.json hosted on huggingface
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    try:
        response = requests.get(url)  # type: ignore
        config = response.json()
    except Exception as e:
        print(f"Failed to fetch {attr} from {url}: {e}")
        return ""

    attr_val = config.get(attr)
    if attr_val is not None:
        return attr_val

    return ""


def is_quantized_model(model_id: str) -> tuple[bool, str]:
    quant_method = ""
    is_quantized = False

    quant_config = get_attr_from_hf(model_id, "quantization_config")

    if quant_config:
        is_quantized = True
        quant_method = quant_config.get("quant_method", "")

    # Redundant check
    if "awq" in model_id.lower():
        is_quantized = True
        quant_method = "awq"
    elif "gptq" in model_id.lower():
        is_quantized = True
        quant_method = "gptq"
    elif "int8" in model_id.lower():
        is_quantized = True
    elif "int4" in model_id.lower():
        is_quantized = True

    return is_quantized, quant_method


def get_dtype(device: str = "", model_id: str = "") -> str:
    if device.lower().startswith("cpu"):
        return "auto"

    if not model_id or not torch.cuda.is_available():
        return "auto"

    base_dtype = get_attr_from_hf(model_id, "torch_dtype")
    base_dtype = re.sub("torch.", "", str(base_dtype))
    if base_dtype:
        return base_dtype  # type: ignore

    return "auto"


def get_device_map(device: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given a string or dictionary for the device, return a dictionary to be passed as the `device_map`.

    - If a dictionary (e.g., {"layer1": "cuda:0", "layer2": "cuda:1"}) is provided, it is returned as is.
    - If a string is provided:
      - If it starts with "cuda" (e.g., "cuda:0", "cuda:1"), the entire model is mapped onto that device.
      - If it is "cpu", the device map is set as "cpu".
      - Otherwise, the device map is set as "auto" for automatic placement.
    """
    # Case 1: If device is already a dictionary, use it directly.
    device_map: str | Dict[str, Any]
    if isinstance(device, dict):
        device_map = device
    # Case 2: Handle when device is provided as a string.
    elif isinstance(device, str):
        if device.lower().startswith("cuda"):
            device_map = {"": device}
        elif device.lower() == "cpu":
            device_map = "cpu"
        else:
            device_map = "auto"
    else:
        raise ValueError("The device parameter must be a string or a dictionary.")

    return {"device_map": device_map}
