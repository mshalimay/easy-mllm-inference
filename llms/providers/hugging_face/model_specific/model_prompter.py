import re
from typing import Any, Callable, Dict

from llms.types import ContentItem
from utils.image_utils import any_to_b64, any_to_pil


class ModelPrompter:
    """
    Encapsulates methods to convert model inputs to the correct format for Hugging Face models.
    `openai` mode: assumes it is using a third-party provider that uses the OpenAI client and chat completion format,
    NOTE: hugging face uses the openai chat completion, but with different keys and dict format.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def to_model(
        cls,
        model_id: str,
        input: ContentItem,
        engine: str,
        provider: str = "",
    ) -> Dict[str, Any]:
        """Route to the correct model prompter based on the model ID."""
        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_id, flags=re.IGNORECASE):
            return cls.to_qwen_vl_2_5(input, engine)

        elif re.search(r"Kimi-VL", model_id, flags=re.IGNORECASE):
            return cls.to_kimi_vl(input, engine)

        else:
            return cls.to_open_ai_chat_completion(input, engine, provider)

    @classmethod
    def to_open_ai_chat_completion(
        cls,
        input: ContentItem,
        engine: str,
        provider: str = "",
    ) -> Dict[str, Any]:
        # Textual input
        if input.type == "text":
            return {"type": "text", "text": input.data}

        # Image input
        elif input.type == "image":
            img_b64 = any_to_b64(input.data, add_header=True)
            if engine == "openai":
                # Vanilla OpenAI format
                return {"type": "image_url", "image_url": {"url": img_b64}}
            else:
                # Hugging Face OpenAI format
                # https://huggingface.co/docs/transformers/main/en/chat_templating_multimodal#image-inputs
                return {"type": "image", "base64": img_b64}

        # Video input
        elif input.type == "video":
            return {"type": "video", "path": input.data}

        else:
            raise NotImplementedError(
                f"{__file__}: Type: {input.type} not implemented for OpenAI chat completion with Hugging Face"
            )

    @classmethod
    def to_kimi_vl(cls, input: ContentItem, engine: str) -> Dict[str, Any]:
        """Kimivl specific prompt format."""

        if engine == "openai":
            return cls.to_open_ai_chat_completion(input, engine)

        if input.type == "text":
            return {"type": "text", "text": input.data}

        elif input.type == "image":
            if engine == "server" or engine == "vllm" or engine == "tgi":
                img = any_to_b64(input.data, add_header=True)
            else:
                img = any_to_pil(input.data)
            return {"type": "image", "image": img}
        else:
            raise NotImplementedError(f"{__file__}: Type: {input.type} not implemented for KimiVL with Hugging Face")

    @classmethod
    def to_qwen_vl_2_5(cls, input: ContentItem, engine: str) -> Dict[str, Any]:
        """Qwen2.5 VL specific prompt format."""

        if engine == "openai":
            return cls.to_open_ai_chat_completion(input, engine)

        if input.type == "text":
            return {"type": "text", "text": input.data}

        elif input.type == "image":
            add_args = {}
            if min_pixels := input.meta_data.get("min_pixels", None):
                add_args["min_pixels"] = min_pixels
            if max_pixels := input.meta_data.get("max_pixels", None):
                add_args["max_pixels"] = max_pixels

            if engine == "server" or engine == "vllm" or engine == "tgi" or engine == "openai":
                img = any_to_b64(input.data, add_header=True)
            else:
                img = any_to_pil(input.data)
            return {"type": "image", "image": img, **add_args}
        else:
            raise NotImplementedError(
                f"{__file__}: Type: {input.type} not implemented for Qwen2.5 VL with Hugging Face"
            )
