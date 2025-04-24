from typing import Any, Dict, List, Union

import numpy as np
import PIL.PngImagePlugin
from PIL import Image
from PIL.ImageFile import ImageFile

ImageInput = Union[Image.Image, str, np.ndarray[Any, Any], bytes, ImageFile, ImageFile]
