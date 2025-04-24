# General generation config for LLMs.
# All provider generation utils utilizes this cofing as a base, and adjust generation parameters for provider-specifics

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

from llms.types import Message


@dataclass
class GenerationConfig:
    model: str
    model_path: str = field(default_factory=lambda: "")
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None  # Consider removing if unused.
    max_tokens: Optional[int] = None
    num_generations: int = 1
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    modalities: List[str] = field(default_factory=lambda: ["text"])
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    reasoning_effort: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    web_search_options: Optional[Dict[str, Any]] = None
    include: Optional[List[str]] = None
    text: Optional[str] = None
    truncation: Optional[str] = None
    parallel_tool_calls: Optional[bool] = None
    mode: str = field(default_factory=lambda: "")

    # OpenAI response API-specific
    previous_response_id: Optional[str] = None

    # OpenAI Response specific - Reasoning
    reasoning: Optional[Dict[str, Any]] = field(default_factory=lambda: {"generate_summary": "detailed"})
    # {reasoning_effort: str, generate_summary: "concise" | "detailed"}

    # OpenAI computer use
    current_url: Optional[str] = None
    acknowledged_safety_checks: Optional[List[str]] = None

    # Hugging Face specific
    engine: str = field(default_factory=lambda: "automodel")  # "automodel" | "vllm" | "server" | "tgi"
    torch_dtype: str = field(default_factory=lambda: "bfloat16")
    flash_attn: Optional[bool] = None
    device: str = field(default_factory=lambda: "")
    quant_bits: str = field(default_factory=lambda: "")
    do_sample: bool = True
    endpoint: str = field(default_factory=lambda: "")
    gpu_memory_utilization: Optional[float] = None

    # VLLM specific
    max_model_len: Optional[int] = None

    provider: str = field(default_factory=lambda: "")

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


valid_fields = {f.name for f in fields(GenerationConfig)}


def get_fields() -> list[str]:
    # Return list to preserve order
    return [f.name for f in fields(GenerationConfig)]


def make_generation_config(
    gen_kwargs: dict[str, Any],
    safe: bool = True,
) -> GenerationConfig:
    try:
        gen_config = GenerationConfig(**gen_kwargs)
    except TypeError as _:
        filtered_combined = {k: v for k, v in gen_kwargs.items() if k in valid_fields}
        gen_config = GenerationConfig(**filtered_combined)
    except Exception as e:
        raise ValueError(f"Error creating GenerationConfig: {e}")

    return gen_config
