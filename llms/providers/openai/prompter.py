from typing import Any, Dict, List

from llms.providers.openai.constants import ROLE_MAPPINGS, UPLOAD_IMAGES
from llms.types import ContentItem, Contents, Message
from utils.image_utils import any_to_b64


class OpenAIPrompter:
    """
    A class to encapsulate prompt adjustments for OpenAI API generation.

    This class handles:
      - Converting a content item to a genai Part (text, image, etc.)
      - Regularizing a list of content items
      - Regularizing messages (including handling system vs. non-system messages)
      - Adjusting the message role according to OpenAI's API
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def img_to_provider(cls, input: ContentItem, mode: str) -> Dict[str, Any]:
        img_b64 = any_to_b64(input.data, add_header=True)
        img_detail = input.meta_data.get("img_detail", "auto")
        if mode == "chat_completion":
            return {"type": "image_url", "image_url": {"url": img_b64, "detail": img_detail}}
        elif mode == "response":
            return {"type": "input_image", "image_url": img_b64, "detail": img_detail}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def text_to_provider(cls, input: ContentItem, mode: str) -> Dict[str, Any]:
        if mode == "chat_completion":
            return {"type": "text", "text": input.data}
        elif mode == "response":
            return {"type": "input_text", "text": input.data}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def img_text_to_provider(cls, input: ContentItem, mode: str) -> Dict[str, Any]:
        if input.type == "text":
            return cls.text_to_provider(input, mode)
        elif input.type == "image":
            return cls.img_to_provider(input, mode)
        else:
            raise ValueError(f"Unknown content item type: {input.type}")

    @staticmethod
    def func_output_to_provider(input: ContentItem, mode: str) -> Dict[str, Any]:
        if mode == "chat_completion":
            return {"type": "function_call_output", "call_id": input.meta_data["call_id"], "output": input.data}
        elif mode == "response":
            return {"type": "function_call_output", "call_id": input.meta_data["call_id"], "output": input.data}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def computer_output_to_provider(cls, input: ContentItem, mode: str) -> Dict[str, Any]:
        if mode == "chat_completion":
            raise ValueError("Computer output not supported in chat completion")
        elif mode == "response":
            # https://platform.openai.com/docs/guides/tools-computer-use

            if len(input.data) > 1:
                raise ValueError("As of March-2025: Computer output must be a single item")

            payload = {
                "type": "computer_call_output",
                "call_id": input.meta_data["call_id"],
                "acknowledged_safety_checks": input.meta_data.get("acknowledged_safety_checks", []),
                "output": cls.img_text_to_provider(input.data[0], mode),
            }
            if input.meta_data.get("url", None):
                payload["current_url"] = input.meta_data["url"]
            return payload
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @classmethod
    def convert_message(cls, message: Message, mode: str = "chat_completion") -> List[Dict[str, Any]]:
        role = cls.convert_role(message.role)
        all_msgs = []

        all_contents = []
        for content_item in message.contents:
            if not content_item.data and not content_item.raw_model_output:
                # Ignore if no data or raw model output is available
                continue

            # For image and text, accumulate contents to write a single message
            if content_item.type == "text" or content_item.type == "image":
                all_contents.append(cls.img_text_to_provider(content_item, mode))
                continue
            else:
                # Flush content once find a type that is not image or text
                if len(all_contents) > 0:
                    all_msgs.append({"role": role, "content": all_contents})
                    all_contents = []

            # For function output, write a single message
            if content_item.type == "function_output":
                all_msgs.append(cls.func_output_to_provider(content_item, mode))

            # For computer output, write a single message
            elif content_item.type == "computer_output":
                all_msgs.append(cls.computer_output_to_provider(content_item, mode))

            # Other cases of model messages, create message with data as is
            elif (
                content_item.type == "computer_call"
                or content_item.type == "function_call"
                or content_item.type == "reasoning"
            ):
                if content_item.raw_model_output is not None:
                    # Use raw model output if available
                    all_msgs.append(content_item.raw_model_output)
                elif content_item.data is not None:
                    # Try to use data if raw model output is not available
                    all_msgs.append(content_item.data)
                else:
                    # Ignore if no data or raw model output is available
                    continue
            else:
                raise ValueError(f"Unknown content item type: {content_item.type}")

        # Flush any remaining contents
        if len(all_contents) > 0:
            all_msgs.append({"role": role, "content": all_contents})

        return all_msgs

    @classmethod
    def convert_prompt(cls, prompt: List[Message], mode: str = "chat_completion") -> List[Dict[str, Any]]:
        reg_prompt = []
        for message in prompt:
            reg_prompt.extend(cls.convert_message(message, mode))
        return reg_prompt

    @staticmethod
    def reset_prompt(prompt: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Reset prompt is not implemented for OpenAI")

    @staticmethod
    def upload_all_images(prompt: List[Dict[str, Any]]) -> None:
        raise NotImplementedError("Upload all images is not implemented for OpenAI")

    @staticmethod
    def convert_role(role: str) -> str:
        return ROLE_MAPPINGS[role]
