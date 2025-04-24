import re
from typing import Any, Dict, List

from qwen_vl_utils import (
    fetch_image,
    fetch_video,  # type: ignore
    process_vision_info,
)

from llms.providers.hugging_face.hugging_face_client_manager import get_client_manager
from llms.providers.hugging_face.parsing_utils import extract_vision_chat_msgs
from llms.types import Message
from utils.image_utils import any_to_pil


def extract_media_from_chat_completion(messages: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Extract media from a chat completion."""
    images = []
    videos = []
    for message in messages:
        content = message.get("content", None)
        if not content:
            continue

        if not isinstance(content, list):
            content = [content]

        for item in content:
            if item.get("type") == "image":
                if "image" in item:
                    images.append(item["image"])
                elif "base64" in item:
                    img = any_to_pil(item["base64"])
                    images.append(img)
                elif "url" in item:
                    img = any_to_pil(item["url"])
                    images.append(img)
            if item.get("type") == "video":
                if "path" in item:
                    videos.append(item["path"])
                elif "url" in item:
                    videos.append(item["url"])
    return images, videos


class ModelProcessor:
    """
    Encapsulates methods to process model inputs and outputs for Hugging Face models.
    """

    def __init__(self) -> None:
        pass

    # ===============================================================
    # Get model inputs
    # ===============================================================
    @classmethod
    def get_inputs(
        cls,
        provider_messages: list[list[Dict[str, Any]]],
        model_path: str,
    ) -> Any:
        """Route to the correct model input getter based on the model path."""
        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_path, flags=re.IGNORECASE):
            return cls.get_inputs_qwen2_5_vl(
                messages=provider_messages,
                model_path=model_path,
            )

        elif re.search(r"Kimi-VL", model_path, flags=re.IGNORECASE):
            return cls.get_inputs_default(
                provider_messages=provider_messages,
                model_path=model_path,
                truncation=True,
            )

        else:
            return cls.get_inputs_default(
                provider_messages=provider_messages,
                model_path=model_path,
            )

    @classmethod
    def get_inputs_default(
        cls,
        provider_messages: list[list[Dict[str, Any]]],
        model_path: str,
        add_generation_prompt: bool = True,
        padding: bool = True,
        truncation: bool = False,
    ) -> Any:
        """Default preparation for model inputs."""
        processor = get_client_manager(model_path).get_processor()

        texts = [
            processor.apply_chat_template(  # type: ignore
                message, add_generation_prompt=add_generation_prompt, return_tensors="pt"
            )
            for message in provider_messages
        ]
        images, videos = [], []
        for message in provider_messages:
            imgs_batch, videos_batch = extract_media_from_chat_completion(message)
            images.extend(imgs_batch)
            videos.extend(videos_batch)

        if len(images) > 0:
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=padding, truncation=truncation)  # type: ignore
        else:
            inputs = processor(text=texts, return_tensors="pt", padding=padding, truncation=truncation)  # type: ignore

        return inputs

    @classmethod
    def get_inputs_qwen2_5_vl(
        cls,
        messages: list[list[Dict[str, Any]]],
        model_path: str,
    ) -> Any:
        processor = get_client_manager(model_path).get_processor()

        texts = [
            processor.apply_chat_template(  # type: ignore
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages, return_video_kwargs=False)  # type: ignore

        inputs = processor(  # type: ignore
            text=texts, images=image_inputs, video=video_inputs, padding=True, return_tensors="pt"
        )
        return inputs

    # ===============================================================
    # Decoding
    # ===============================================================
    @classmethod
    def decode_outputs(
        cls,
        outputs: Any,
        model_path: str,
        start_idxs: List[int] = [],
        skip_special_tokens: bool = True,
    ) -> Any:
        outputs_trimmed = outputs
        if start_idxs:
            outputs_trimmed = [output[start_idx:] for output, start_idx in zip(outputs, start_idxs)]

        if re.match(r"Qwen/Qwen2\.5-VL-.*", model_path, flags=re.IGNORECASE):
            return cls.decode_outputs_qwen2_5_vl(outputs_trimmed, model_path, skip_special_tokens)
        else:
            return cls.decode_outputs_default(outputs_trimmed, model_path, skip_special_tokens)

    @classmethod
    def decode_outputs_default(
        cls,
        outputs: Any,
        model_path: str,
        skip_special_tokens: bool = True,
    ) -> Any:
        processor = get_client_manager(model_path).get_processor()
        decoded_outputs = processor.batch_decode(  # type: ignore
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        return decoded_outputs

    @classmethod
    def decode_outputs_qwen2_5_vl(
        cls,
        outputs: Any,
        model_path: str,
        skip_special_tokens: bool = True,
    ) -> Any:
        processor = get_client_manager(model_path).get_processor()
        decoded_outputs = processor.batch_decode(  # type: ignore
            outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

        return decoded_outputs
