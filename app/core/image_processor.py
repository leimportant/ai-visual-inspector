
import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def resize_image(
    image: np.ndarray,
    width: int = 640
) -> np.ndarray:
    height, original_width = image.shape[:2]

    if original_width == 0:
        raise ValueError("Invalid image width")

    ratio = width / original_width
    new_height = int(height * ratio)

    return cv2.resize(
        image,
        (width, new_height),
        interpolation=cv2.INTER_AREA
    )


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(
            "Kernel size must be a positive odd number"
        )

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


def detect_edges(
    image: np.ndarray,
    threshold1: int = 100,
    threshold2: int = 200
) -> np.ndarray:
    grayscale = to_grayscale(image)

    return cv2.Canny(
        grayscale,
        threshold1,
        threshold2
    )
