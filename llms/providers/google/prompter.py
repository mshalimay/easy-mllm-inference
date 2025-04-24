from typing import Any, List, Optional, Tuple

from google.genai import types as genai_types
from google.genai.types import Part

from llms.providers.google.constants import ROLE_MAPPINGS, UPLOAD_IMAGES
from llms.providers.google.file_manager import GoogleFileManager
from llms.types import ContentItem, Contents, Message
from utils.image_utils import any_to_bytes, is_image


# Obs.: didn't use `self` logic for convenience as typically use the same prompter.
# This way doesnt have to instantiate the class every time.
class GooglePrompter:
    """
    A class to encapsulate prompt adjustments for Google API generation.

    This class handles:
      - Converting a content item to a genai Part (text, image, etc.)
      - Regularizing a list of content items
      - Regularizing messages (including handling system vs. non-system messages)
      - Adjusting the message role according to Google's API
    """

    upload_images = UPLOAD_IMAGES

    def __init__(self) -> None:
        pass

    @classmethod
    def convert_content_item(cls, content_item: ContentItem) -> Any:
        if content_item.type == "text":
            return Part.from_text(text=content_item.data)

        elif content_item.type == "image":
            if cls.upload_images:
                input_file = GoogleFileManager.get_upload_image_file(image=content_item.data)
                return Part.from_uri(file_uri=input_file.uri, mime_type=input_file.mime_type)  # type:ignore
            else:
                img_bytes = any_to_bytes(content_item.data, format="PNG")
                return Part.from_bytes(data=img_bytes, mime_type="image/png")

        elif content_item.type == "function_call":
            # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling#python_3
            if content_item.raw_model_output is not None:
                return content_item.raw_model_output
            else:
                # return Part.from_function_call(content_item.meta_data["name"], function_call=content_item.data)
                raise ValueError("Function call not found in content item")

        elif content_item.type == "function_output":
            return Part.from_function_response(
                name=content_item.meta_data["name"],  # must have a name
                response={"content": content_item.data},
            )

        elif content_item.type == "reasoning":
            if content_item.raw_model_output is not None:
                return content_item.raw_model_output
            else:
                raise ValueError(f"Reasoning not found in content item: {content_item}")

        else:
            raise ValueError(f"Unknown content type: {content_item.type}")

    @classmethod
    def convert_contents(cls, contents: Contents) -> List[Part]:
        parts = []
        for content in contents:
            parts.append(cls.convert_content_item(content))
        return parts

    @classmethod
    def convert_message(cls, message: Message) -> genai_types.Content:
        role = cls.convert_role(message.role)
        parts = cls.convert_contents(message.contents)
        return genai_types.Content(role=role, parts=parts)

    @staticmethod
    def convert_role(role: str) -> str:
        return ROLE_MAPPINGS[role]

    @classmethod
    def convert_prompt(cls, prompt: List[Message]) -> List[genai_types.Content]:
        reg_prompt = [genai_types.Content(role="system", parts=[])]
        for message in prompt:
            if message.role == "system":
                reg_prompt[0] = cls.convert_message(message)
            else:
                reg_prompt.append(cls.convert_message(message))
        return reg_prompt

    @staticmethod
    def reset_prompt(prompt: List[genai_types.Content]) -> List[genai_types.Content]:
        old_to_new = GoogleFileManager.reupload_all_images()
        for i, content in enumerate(prompt):
            if not content.parts:
                continue
            for j, part in enumerate(content.parts):
                if hasattr(part, "file_data") and part.file_data is not None:
                    new_file = old_to_new[part.file_data.file_uri]  # type: ignore

                    if not new_file.uri or not new_file.mime_type:
                        raise ValueError("Error uploading file: no uri or mime type")
                    content.parts[j] = Part.from_uri(
                        file_uri=new_file.uri,
                        mime_type=new_file.mime_type,
                    )
        return prompt

    @staticmethod
    def upload_all_images(prompt: List[genai_types.Content]) -> List[genai_types.Content]:
        for i, content in enumerate(prompt):
            if not content.parts:
                continue
            for j, part in enumerate(content.parts):
                if part.inline_data is not None:
                    if (
                        part.inline_data.mime_type
                        and "image" in part.inline_data.mime_type  # short-circuit to prevent has_image if not necessary
                        or is_image(part.inline_data.data)
                    ):
                        new_file = GoogleFileManager.get_upload_image_file(image=part.inline_data.data)
                        if not new_file.uri or not new_file.mime_type:
                            raise ValueError("Error uploading file: no uri or mime type")
                        content.parts[j] = Part.from_uri(
                            file_uri=new_file.uri,
                            mime_type=new_file.mime_type,
                        )
        return prompt
