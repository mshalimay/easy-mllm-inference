# Prompter
ROLE_MAPPINGS = {
    "assistant": "assistant",
    "user": "user",
    "system": "system",  # Obs.: new role is `developer`, but system is backward and forward compatible
    "developer": "system",
}
DEFAULT_HF_MODE = "chat_completion"


# VLLM defaults
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_MAX_MODEL_LEN: int | None = None
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_QUANTIZE = False
VLLM_ENFORCE_EAGER = False

# VLLM OpenAI Compatile server
VLLM_OPENAI_KEY = "EMPTY"
ENDPOINT_TEMPLATE = "http://{host}:{port}/v1"
VLLM_USE_V1 = 1
VLLM_DEFAULT_PARAMS_PER_MODEL = {
    "Qwen2.5-VL": {
        "repetition_penalty": 1.05,
    },
}


# Flash attention
USE_FLASH_ATTN = True
