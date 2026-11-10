
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


def apply_threshold(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255,
    threshold_type: int = cv2.THRESH_BINARY,
) -> np.ndarray:
    grayscale = image if len(image.shape) == 2 else to_grayscale(image)

    _, thresholded = cv2.threshold(
        grayscale,
        threshold_value,
        max_value,
        threshold_type,
    )

    return thresholded


def detect_contours(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255,
) -> list[np.ndarray]:
    thresholded = apply_threshold(
        image,
        threshold_value=threshold_value,
        max_value=max_value,
        threshold_type=cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        thresholded,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return contours


def draw_contours(
    image: np.ndarray,
    contours: list[np.ndarray],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    image_copy = image.copy()

    if len(image_copy.shape) == 2:
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)

    return cv2.drawContours(
        image_copy,
        contours,
        -1,
        color,
        thickness,
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
