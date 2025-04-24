import os
import tempfile
from typing import Any, Dict, Optional

from google import genai
from google.genai import types as genai_types
from PIL import Image

from llms.providers.google.google_client_manager import get_client_manager
from utils.image_utils import any_to_pil


class GoogleFileManager:
    # Maps images to genai files. Why: GoogleAPI requires uploading big images to the cloud
    # to send in the prompts. This dictionary maps images previously uploaded to the genai files.
    img_to_uploaded: Dict[int, genai_types.File] = {}  # hash(image) -> genai file

    # Maps genai files to PIL images. This necessary because it is not possible to retrieve the images back from genai files.
    # and is useful to create prompt visualizations, reupload images
    uploaded_to_img: Dict[str, Image.Image] = {}  # genai file -> image

    @classmethod
    def upload_image_file(
        cls,
        image_path: str,
        client: genai.Client,
    ) -> genai_types.File:
        return client.files.upload(file=image_path)

    @classmethod
    def get_upload_image_file(cls, image: Any, force_upload: bool = False) -> genai_types.File:
        image_pil = any_to_pil(image)
        image_hash = hash(image_pil.tobytes())
        if image_hash in cls.img_to_uploaded and not force_upload:
            gen_ai_file = cls.img_to_uploaded[image_hash]
            if gen_ai_file.uri is None:
                raise ValueError("Uploaded file has no valid URI")
            cls.uploaded_to_img[gen_ai_file.uri] = image_pil
            return gen_ai_file

        client = get_client_manager().get_client()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            image_pil.save(temp.name, format="PNG")
            gen_ai_file = cls.upload_image_file(temp.name, client)
            if gen_ai_file.uri is None:
                raise ValueError("Uploaded file has no valid URI")
            cls.uploaded_to_img[gen_ai_file.uri] = image_pil
            cls.img_to_uploaded[image_hash] = gen_ai_file
        os.remove(temp.name)
        return cls.img_to_uploaded[image_hash]

    @classmethod
    def reupload_image(cls, gen_ai_filename: str) -> genai_types.File:
        original_img = cls.uploaded_to_img.pop(gen_ai_filename)
        new_gen_ai_file = cls.get_upload_image_file(image=original_img, force_upload=True)
        return new_gen_ai_file

    @classmethod
    def reupload_all_images(cls) -> Dict[str, genai_types.File]:
        old_gen_ai_files = list(cls.uploaded_to_img.keys())
        new_gen_ai_files = [cls.reupload_image(filename) for filename in old_gen_ai_files]
        return {old_gen_ai_files[i]: new_gen_ai_files[i] for i in range(len(old_gen_ai_files))}

    @classmethod
    def img_to_genai(cls, image: Image.Image) -> Optional[genai_types.File]:
        image_hash = hash(image)
        return cls.img_to_uploaded.get(image_hash)

    @classmethod
    def gen_ai_to_img(cls, gen_ai_filename: str) -> Optional[Image.Image]:
        return cls.uploaded_to_img.get(gen_ai_filename)
