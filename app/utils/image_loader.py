
from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from disk in BGR format.

    Raises:
        FileNotFoundError: Image does not exist.
        ValueError: Image cannot be decoded.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(
            f"Unable to decode image: {image_path}"
        )

    return image
