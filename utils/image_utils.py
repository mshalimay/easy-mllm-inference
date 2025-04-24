import base64
import os
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import requests  # type: ignore
from numpy.typing import NDArray
from PIL import Image

from utils.types import ImageInput


# ===============================================================================
# LINK Type checking
# ===============================================================================
def is_url(img: str) -> bool:
    return img.startswith("http")


def is_image(img: Any, return_image: bool = False) -> bool | Image.Image | None:
    try:
        pil_img = any_to_pil(img)
        return pil_img if return_image else True
    except Exception:
        return False if not return_image else None


def is_b64_image(image: Any, return_image: bool = False) -> bool | Image.Image | None:
    if not isinstance(image, str):
        return False if not return_image else None
    if image.startswith("data:image/png;base64,"):
        if return_image:
            return b64_to_pil(image)
        else:
            return True
    try:
        decoded = base64.b64decode(image)
        Image.open(BytesIO(decoded))
        return True if not return_image else image
    except Exception:
        return False if not return_image else None


def is_string(obj: Any) -> bool:
    return isinstance(obj, str) and not is_image(obj)


# ===============================================================================
# LINK Image conversion
# ===============================================================================
def get_image_from_url(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Image.Image:
    if not headers:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    response = requests.get(url, stream=True, headers=headers, timeout=timeout)
    return Image.open(response.raw)


def numpy_to_pil(img_np: NDArray[Any]) -> Image.Image:
    return Image.fromarray(img_np)


def b64_to_pil(img_b64: str) -> Image.Image:
    # If there's a comma, split and decode only the base64 part
    # i.e., "data:image/png;base64,<...>"
    if img_b64.startswith("data:image"):
        _, base64_data = img_b64.split(",", 1)
    else:
        base64_data = img_b64
    decoded = base64.b64decode(base64_data)
    return Image.open(BytesIO(decoded))


def any_to_pil(img: ImageInput) -> Image.Image:
    # If image is a numpy array, convert to PIL.
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)

    elif isinstance(img, str):
        # If image is a base64 string, convert to PIL.
        if pil_img := is_b64_image(img, return_image=True):
            return pil_img

        elif os.path.exists(img):
            return Image.open(img)

        elif is_url(img):
            return get_image_from_url(img)

        else:
            raise ValueError(f"Invalid image string: {img}")

    # If image is a PIL image, return it as is.
    elif isinstance(img, Image.Image):
        return img

    # If image is a bytes object, convert to PIL.
    elif isinstance(img, bytes):
        return Image.open(BytesIO(img))

    else:
        raise ValueError(f"Invalid image type: {type(img)}")


def any_to_b64(img: ImageInput, add_header: bool = True) -> str:
    if img is None:
        raise ValueError("No image provided for conversion to base64")

    if isinstance(img, str):
        # If b64 image already, add/remove header as needed.
        if is_b64_image(img):
            # If add_header and image doesn't have a header, add it.
            if add_header and not img.startswith("data:image/png;base64,"):
                return "data:image/png;base64," + img
            # If add_header is False and image has a header, strip it.
            elif not add_header and img.startswith("data:image/png;base64,"):
                _, b64_data = img.split(",", 1)
                return b64_data
            # Otherwise, just return the image.
            return img

        # If image is a URL, download it and convert to PIL.
        elif is_url(img):
            img = get_image_from_url(img)

        # If image is a path, open it as a PIL image.
        elif os.path.exists(img):
            img = Image.open(img)

        else:
            raise ValueError(f"Invalid image string: {img}")

    # If image is a numpy array, convert to PIL.
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # If image is a bytes object, convert to PIL.
    elif isinstance(img, (bytes, bytearray)):
        img = Image.open(BytesIO(img))

    # Convert PIL image to base64.
    with BytesIO() as image_buffer:
        img.save(image_buffer, format="PNG")
        byte_data = image_buffer.getvalue()
        img_b64 = base64.b64encode(byte_data).decode("utf-8")
        if add_header:
            img_b64 = "data:image/png;base64," + img_b64
    return img_b64


def any_to_bytes(img: ImageInput, format: str = "PNG") -> bytes:
    if img is None:
        raise ValueError("No image provided for conversion to bytes")

    if isinstance(img, bytes):
        return img

    # Convert the input to a PIL Image using the existing helper function.
    pil_img = any_to_pil(img)

    # Use a BytesIO buffer to save the PIL image in PNG format and retrieve its bytes.
    with BytesIO() as image_buffer:
        pil_img.save(image_buffer, format=format)
        return image_buffer.getvalue()
